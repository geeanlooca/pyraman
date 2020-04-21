def torch_rk4(func, y0, t, *args, **kwargs):
    """Integrate ODEs with a fourth-order fixed-step Runge-Kutta solver.

    Params
    ------
    func : callable
        A function describing the differential equation.
    y0 : torch.Tensor
        The initial conditions.
    t : torch.Tensor
        The evaluation points.

    Returns
    -------
    torch.Tensor
        The solution at the last evaluation point.
    """
    y = y0.clone()

    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        k1 = h * func(y, t[i - 1], *args, **kwargs)
        k2 = h * func(y + k1 / 2, t[i - 1] + h / 2, *args, **kwargs)
        k3 = h * func(y + k2 / 2, t[i - 1] + h / 2, *args, **kwargs)
        k4 = h * func(y + k3, t[i - 1] + h, *args, **kwargs)
        y = y + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return y
