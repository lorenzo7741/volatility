from scipy.interpolate import interp1d
from datetime import datetime, date
import numpy as np
import time
from utility import bisection, fixed_point
from scipy.stats import norm
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from matplotlib import cm


class DeltaVolatilitySurface:
    def __init__(self, surface: pd.DataFrame, trade_date: date = None, header_conv=lambda x: float(x), maturity_col_name='Time'):
        """
        Initializes a DeltaVolatilitySurface object.

        Parameters:
        - surface: pd.DataFrame
          DataFrame containing delta volatility surface data.
        - trade_date: date, optional
          Trade date associated with the volatility surface.
        - header_conv: function, optional
          Function to convert column headers to desired format (default is lambda x: float(x)).
        - maturity_col_name: str, optional
          Name of the column representing maturity in the DataFrame (default is 'Time').
        """
        delta_header = surface.columns[1:]
        new_header = [header_conv(col) for col in delta_header]
        self.surface = surface.rename(columns=dict(zip(delta_header, new_header)))
        self.values = surface.iloc[:, 1:].values
        self.delta = [header_conv(col) for col in surface.columns[1:]]
        self.maturity = list(surface[maturity_col_name])
        self.td = trade_date

    @classmethod
    def from_csv(cls, path: str,
                 trade_date=None,
                 date_format='%Y-%m-%d', sep=',', decimal='.',
                 maturity_col_name='Time',
                 header_conv=float #lambda x: int(x.replace(' D C', ''))/100
                 ):
        """
        Alternative constructor to create DeltaVolatilitySurface from a CSV file.

        Parameters:
        - path: str
          Path to the CSV file containing delta volatility surface data.
        - trade_date: date, optional
          Trade date associated with the volatility surface.
        - date_format: str, optional
          Date format for parsing the maturity column (default is '%Y-%m-%d').
        - sep: str, optional
          Separator used in the CSV file (default is ',').
        - decimal: str, optional
          Decimal point used in the CSV file (default is '.').
        - maturity_col_name: str, optional
          Name of the column representing maturity in the CSV file (default is 'Time').
        - header_conv: function, optional
          Function to convert column headers to desired format (default is float).

        Returns:
        - DeltaVolatilitySurface
          Instance of DeltaVolatilitySurface created from the CSV file.
        """

        conv = lambda x: datetime.strptime(x, date_format).date()
        surface = pd.read_csv(path, sep=sep, decimal=decimal, converters={maturity_col_name: conv})
        delta = [header_conv(c) for c in surface.columns if c != maturity_col_name]
        new_header = [str(int(100 * d)) for d in delta]

        surface.rename(columns=dict(zip(surface.columns[1:], new_header)), inplace=True)

        return cls(surface, trade_date=trade_date, header_conv=header_conv)

    def plot(self):
        """
        Plots the delta volatility surface in 3D.
        """
        unitary_maturity = [(mat - self.td).days/252 for mat in self.maturity]
        maturity_mg, moneyness_mg = np.meshgrid(unitary_maturity, self.delta)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Plot the surface.
        surf = ax.plot_surface(maturity_mg, moneyness_mg, self.values.transpose(),cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def convert(self, moneyness_header, r, kind='linear', algo='mixed'):
        """
        Converts the delta volatility surface to a moneyness volatility surface.

        Parameters:
        - moneyness_header: numpy array
          Array representing the moneyness values.
        - r: float
          Risk-free rate.
        - kind: str, optional
          Interpolation method for smile interpolation (default is 'linear').
        - algo: str, optional
          Algorithm for volatility conversion ('fixed', 'bisection', 'mixed') (default is 'mixed').

        Returns:
        - MoneynessVolatilitySurface
          Moneyness volatility surface obtained from the conversion.
        """

        maturity = self.maturity
        td = self.td
        surface = self.surface
        new_vol = np.zeros([len(maturity), moneyness_header.shape[0]])
        conv_counter = 0
        index_failure = []
        start = time.time()
        for i, mat in enumerate(maturity):
            day_to_maturity = mat - td
            T = day_to_maturity.days / 252
            smile_dot = surface.values[i, 1:]

            def smile_interp(delta):
                if delta <= self.delta[0]:
                    return smile_dot[0]
                elif delta >= self.delta[-1]:
                    return smile_dot[-1]
                else:
                    vol_interp_in_smile = interp1d(self.delta, smile_dot, kind=kind)
                    return vol_interp_in_smile(delta)

            d1_m = lambda m, sigma, ttm: (np.log(m) + (1 / 2) * sigma ** 2 * ttm) / (sigma * np.sqrt(ttm))  # ok

            x0 = np.exp(-r * T)
            for j, m in enumerate(moneyness_header):
                res_obj = None
                if algo == 'fixed' or algo == 'mixed':
                    fun_to_find_fixed_point = lambda delta: np.exp(-r * T) * norm.cdf(d1_m(m, smile_interp(delta), T))
                    res_obj = fixed_point(fun_to_find_fixed_point, x0)
                    if algo == 'mixed' and not res_obj['convergence']:
                        fun_to_find_zero = lambda delta: np.exp(-r * T) * norm.cdf(d1_m(m, smile_interp(delta), T)) - delta
                        res_obj = bisection(fun_to_find_zero, 0, 1000)
                        print('WARNING: failure of fixed point algorithm:. I try with bisection')
                elif algo == 'bisection':
                    fun_to_find_zero = lambda delta: np.exp(-r * T) * norm.cdf(d1_m(m, smile_interp(delta), T)) - delta
                    res_obj = bisection(fun_to_find_zero, 0, 1000)
                if res_obj['convergence']:
                    conv_counter += 1
                    vol_m_T = smile_interp(res_obj['result'])
                    new_vol[i, j] = vol_m_T
                else:
                    index_failure.append((i, j))
                    print('Not convergence in monyness ' + str(m) + ' and Time to Maturity ' + str(mat) + '\nResult object: ' + str(res_obj))

        exec = time.time() - start
        new_val_heder = [str(int(c)) for c in 100*moneyness_header]
        new_surface = pd.DataFrame(new_vol, columns=new_val_heder)
        new_surface.insert(0, 'Time', self.maturity)
        return MoneynessVolatilitySurface(new_surface, trade_date=self.td)

    def to_csv(self, path):
        """
        Writes the delta volatility surface to a CSV file.

        Parameters:
        - path: str
          Path to the CSV file for saving the data.
        """
        self.surface.to_csv(path, index=False)

    def __str__(self):
        return f"""
        Delta Volatility Surface:
        Trade Date: {self.td}
        Surface: {self.surface}
        """


class MoneynessVolatilitySurface:
    def __init__(self, surface: pd.DataFrame, trade_date: date = None, header_conv=lambda x: float(x), maturity_col_name='Time'):
        """
        Initializes a MoneynessVolatilitySurface object.

        Parameters:
        - surface: pd.DataFrame
          DataFrame containing moneyness volatility surface data.
        - trade_date: date, optional
          Trade date associated with the volatility surface.
        - header_conv: function, optional
          Function to convert column headers to the desired format (default is lambda x: float(x)).
        - maturity_col_name: str, optional
          Name of the column representing maturity in the DataFrame (default is 'Time').
        """

        val_header = surface.columns[1:]
        new_val_header = [header_conv(col) for col in val_header]
        self.surface = surface.rename(columns=dict(zip(val_header, new_val_header)))
        self.values = surface.iloc[:, 1:].values
        self.val_header = [header_conv(col) for col in surface.columns[1:]]
        self.maturity = list(surface[maturity_col_name])
        self.td = trade_date

    def convert(self):
        """
        Placeholder function for converting the moneyness volatility surface.
        To define this function, contact lorenzo.mori@hotmail.it.
        """
        print("MoneynessVolatilitySurface.convert: TO DEFINE. If you want to have this function write to lorenzo.mori@hotmail.it")

    def __str__(self):
        return f"""
        Moneyness Volatility Surface:
        Trade Date: {self.td}
        Surface: {self.surface}
        """

    def plot(self):
        """
        Plots the moneyness volatility surface in 3D.
        """
        unitary_maturity = [(mat - self.td).days/252 for mat in self.maturity]
        maturity_mg, moneyness_mg = np.meshgrid(unitary_maturity, self.val_header)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Plot the surface.
        surf = ax.plot_surface(maturity_mg, moneyness_mg, self.values.transpose(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def to_csv(self, path):
        """
        Writes the moneyness volatility surface to a CSV file.

        Parameters:
        - path: str
          Path to the CSV file for saving the data.
        """
        self.surface.to_csv(path, index=False)

