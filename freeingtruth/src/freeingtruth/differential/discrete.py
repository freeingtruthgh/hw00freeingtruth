# /// script
# requires-python = ">=3.12"
# ///

from array import array

def diff(t: array,x: array) -> array:
    """
    Computes the discrete derivative of a timeseries data

    Formula: v(t) = ( x(t_k)-x(t_(k-1)) )/( t(k)-t(k-1) )

    Args:
        t (array): time value
        x (array): signal value

    Returns:
        array: The discrete derivative of the input data
    """
    t = array('d', t)
    x = array('d', x)

    # Check for equal lengths
    if not len(t)==len(x):
        print("The datasets are not the same length!")

    v=[]
    # Iterate through the data
    for k in range(1,len(t)):
        v_value = (x[k]-x[k-1])/(t[k]-t[k-1])
        v.append(v_value)

    return v
