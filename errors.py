import numpy as np
from sympy import *
import inspect
import scipy.optimize as opt

values = np.array([np.arange(10,20),np.arange(11,21)])
errors = np.array([np.arange(1,2,0.1),np.arange(2,3,0.1)])

# Example function and derivative
def div(a, b):
    return 1.*a/b

def func_eval(function, values, errors):
    """
    Parameters
    ----------
    function: function object
        The function of N variables, e.g. 'div' if you define 'def div(a, b): return 1.*a/b'
    values: sequence
        A sequence of N equal length arrays that represent the input variables
    errors: sequence
        A sequence of N equal length arrays that represent the errors on the *values
    """
    # Evaluate the function for the given values, e.g. 1.*a/b
    evaluated = function(*values)
    
    # Get the arguments from the function and turn them into variables using symbolic logic, e.g. [a, b]
    arguments = inspect.getargspec(function)[0]
    variables = [Symbol(v) for v in arguments]

    # Define the first derivative of the function with respect to each variable, e.g. [1.0/b, -1.0*a/b**2]
    derivatives = [function(*variables).diff(v) for v in variables]
    
    # Evaluate the first derivative of the function at each value
    dfdv = lambdify(variables, derivatives, 'numpy')(*values)
    
    # For symmetric uncertainties, do linear propagation of errors
    sym_uncertainties = np.sqrt(sum(np.asarray(errors)**2*np.asarray(dfdv)**2))
    
    # For asymmetric uncertanties, get the logarithm of the likelihood function, ln L(a,c)
    # def lnlike(vars, vals, errs):
    #     a, c = vars
    #     arr_a, arr_b = vals
    #     a_dn, a_up = errs[0].T
    #     b_dn, b_up = errs[1].T
    #     A = (a-arr_a)**2/(abs(a_up)*abs(a_dn) + ((abs(a_up)-abs(a_dn))*(a-arr_a)))
    #     B = (c-a-arr_b)**2/(abs(b_up)*abs(b_dn) + ((abs(b_up)-abs(b_dn))*(c-a-arr_b)))
    #     return -0.5*(A+B)
    # 
    # # Get the uncertainties by minimizing the likelihood function for c with respect to a
    # nll = lambda *args: -lnlike(*args)
    # result = opt.minimize(nll, np.array([1.,1.]), args=(values, errors))
    # print result
    
    
    # Return the results
    return evaluated, sym_uncertainties
    