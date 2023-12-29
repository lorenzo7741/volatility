def fixed_point(f, x0, maxiter=1000, tol=1e-6):
    step, cond, reach_maxiter = 0, False, False
    while not cond and not reach_maxiter:
        x1 = f(x0)
        step += 1
        # print('Iteration-%d, x1 = %0.6f and f(x1) = %0.6f' % (iter, x1, f(x1)))
        cond = abs(x1 - x0) < tol
        x0 = x1
        reach_maxiter = step >= maxiter
    if cond:
        return {'result': x0, 'convergence': True, 'iteration': step}
    else:
        return {'result': None, 'convergence': False, 'iteration': step}


def bisection(f, a, b, maxiter=1000, tol=1e-6):
    step = 0
    xl, xr = a, b
    if abs(f(xl)) < tol:
        return {'result': xl, 'convergence': True, 'iteration': 0}
    elif abs(f(xr)) < tol:
        return {'result': xr, 'convergence': True, 'iteration': 0}
    elif f(a) * f(b) < 0:
        reach_maxiter, cond = False, False
        x_mid = None
        while not cond and not reach_maxiter:
            x_mid = (xl + xr) / 2
            step += 1
            cond = abs(f(x_mid)) < tol
            reach_maxiter = step >= maxiter
            if not cond and not reach_maxiter:
                if f(x_mid) * f(xl) < 0:
                    xr = x_mid
                elif f(x_mid) * f(xr) < 0:
                    xl = x_mid
                else:
                    raise ValueError('Probably the function is not continuos')
        if cond:
            return {'result': x_mid, 'convergence': True, 'iteration': step}
        else: # maxiter reached
            return {'result': None, 'convergence': False, 'iteration': step}
    else:
        raise ValueError('f(a) * f(b) is greater then zero')
