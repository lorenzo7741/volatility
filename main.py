# Implementing Fixed Point Iteration Method
import math
import numpy as np
from datetime import datetime, date
from scipy.stats import norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import time


def test_interp():
    x = np.linspace(0, 10, num=11, endpoint=True)
    y = np.cos(-x**2/9.0)
    f = interp1d(x, y) # Default is 'linear'
    f2 = interp1d(x, y, kind='cubic')

    xnew = np.linspace(0, 10, num=41, endpoint=True)
    plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
    plt.legend(['data', 'linear', 'cubic'], loc='best')
    plt.show()


# import volatility
def test_volatility(path, algo='mixed', kind='cubic', plot_smile=False, plot_vol=False):
    '''
    :arg p: path of the csv that contais volatility surface
    :param algo: 'f' for xied oint algorithm and 'b' for bisection
    :param k: can be 'linear' or 'cubic'
    :param to_plot
    :return:
    '''

    conv = lambda x: datetime.strptime(x, '%m/%d/%Y').date()
    v = pd.read_csv(path, sep=',', decimal='.', converters={'Time': conv})
    delta_header = [int(c.replace(' D C', ''))/100 for c in v.columns if c != 'Time']
    moneyness_header = np.concatenate([np.array([0.01]), np.linspace(0.05, 0.95, num=19, endpoint=True), np.array([0.99])])
    maturity = v['Time'].values
    td = date(2022, 2, 1) # td = date.today()
    new_vol = np.zeros([maturity.shape[0], moneyness_header.shape[0]])
    conv_counter = 0
    index_failure = []

    start = time.time()
    for i, mat in enumerate(maturity):
        day_to_maturity = mat - td
        T = day_to_maturity.days / 252
        r = 0.01
        smile_dot = v.values[i, 1:]

        def smile_interp(delta):
            if delta < delta_header[0]:
                return smile_dot[0]
            elif delta > delta_header[-1]:
                return smile_dot[-1]
            else:
                vol_interp_in_smile = interp1d(delta_header, smile_dot, kind=kind)
                return vol_interp_in_smile(delta)
        if plot_smile:
            delta_new = np.linspace(delta_header[0] - 0.05, delta_header[-1] + 0.05, num=500, endpoint=True)
            smile_interp_y = np.array([smile_interp(d) for d in delta_new])
            plt.plot(delta_header, smile_dot, 'o', delta_new, smile_interp_y, '-')
            plt.show()

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

    maturity = [(mat - td).days/252 for mat in maturity]
    maturity_mg, moneyness_mg = np.meshgrid(maturity, moneyness_header)

    if plot_vol:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(maturity_mg, moneyness_mg, new_vol.transpose(), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
    return {'Execution': exec, 'Total number of convergence points': conv_counter, 'total number of points': len(maturity) * len(moneyness_header)}, new_vol, index_failure, moneyness_header


#def test_bisection():
#    a = 0
#    b = 10
#    f = lambda x: math.atan(x) * math.sqrt(x) + 3 - x
#    res_obj = bisection(f, a, b)
#    print(res_obj)


#def test_fixed_point():
#    x0 = 6

#    f = lambda x: math.atan(x) * math.sqrt(x) + 3
#    fx = fixed_point(f, x0, maxiter=13)
#    print(fx)

#    f = lambda x: x * x * x + x * x - 1
#    fx = fixed_point(f, x0, maxiter=13)
#    print(fx)

#    f = lambda x: math.atan(x) + 3
#    fx = fixed_point(f, x0, maxiter=13)
#    print(fx)

#print('---fixed point---')
#test_fixed_point()
#print('-----------------')

#print('---vol conversion bisection---')
#res, vol1, index_failure1 = test_volatility(algo='b',  k='linear')
#print(res)
#print('-----------------')

print('---vol conversion fixed point---')

for (path_input, path_output) in [('vol_eex_b.csv', 'vol_eex_b_output.csv'), ('vol_peg_m.csv', 'vol_peg_m_output.csv'), ('vol.csv', 'vol_output.csv')]:
    res, vol, index_failure2, moneyness = test_volatility(path_input, algo='mixed', plot_vol=True, kind='linear')
    df_vol = pd.DataFrame(vol, columns=['Mon ' + str(x) for x in moneyness])
    df_vol.to_csv(path_output, index=False)
print('-----------------')


#print(res)
#print('-----------------')
#diff = vol1 - vol2
#diff_reshape = [diff[i, j] for i in range(diff.shape[0]) for j in range(diff.shape[1]) if not (i, j) in index_failure1 + index_failure2]
#min_diff = min(diff_reshape)
#max_diff = max(diff_reshape)

#print('Max: ' + str(max_diff) + ' Min: ' + str(min_diff))
