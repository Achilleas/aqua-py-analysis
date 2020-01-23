import math

def truncate(number, digits) -> float:
    '''
    Truncate number to n nearest digits. Set to -ve for decimal places

    Args:
        number (float) : the number to truncate
        digits (int) : nearest digits. 0 truncate to 1. 1 truncate to 10. -1
                                        truncate to 0.1
    '''
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper
