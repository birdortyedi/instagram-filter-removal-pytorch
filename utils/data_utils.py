def linear_scaling(x):
    return (x * 255.) / 127.5 - 1.


def linear_unscaling(x):
    return (x + 1.) * 127.5 / 255.