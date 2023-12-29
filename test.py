import numpy as np
from VolatilitySurface import DeltaVolatilitySurface as DVF
from datetime import date
from utility import bisection, fixed_point
import math

def ex_volatility_conversion():

    # Get and Plot Volatility from delta
    path_vol_delta = r'data/ex_vol_on_delta.csv'
    date_format = '%m/%d/%Y'
    dvs = DVF.from_csv(path_vol_delta, # Path that contains the csv with the volatility respect to the delta
                       trade_date=date(2022, 1, 1), # Trade Date of observation of thevolatility
                       header_conv=lambda x: int(x.replace(' D C', ''))/100, # Function to use to convert the header of the csv to a numeric delta
                       date_format=date_format) # Dates Format
    dvs.plot()

    # Conversion and plotting of the result
    path_vol_moneyness = r'data/ex_vol_on_moneyness.csv'
    moneyness_header = np.concatenate([np.array([0.01]), np.linspace(0.05, 0.95, num=19, endpoint=True), np.array([0.99])])
    interest_rate = 0.01
    mon_vol_surf = dvs.convert(moneyness_header, interest_rate)
    mon_vol_surf.plot()
    mon_vol_surf.to_csv(path_vol_moneyness)

def ex_bisection():
    a = 0
    b = 10
    f = lambda x: math.atan(x) * math.sqrt(x) + 3 - x
    res_obj = bisection(f, a, b)
    print(res_obj)

def ex_fixed_point():
    x0 = 6
    f = lambda x: math.atan(x) * math.sqrt(x) + 3
    fx = fixed_point(f, x0, maxiter=1000)
    print(fx)

    f = lambda x: math.atan(x) + 3
    fx = fixed_point(f, x0, maxiter=1000)
    print(fx)

ex_volatility_conversion()
ex_bisection()
ex_fixed_point()

