from .SSAN_M import SSAN_M
from .SSAN_R import SSAN_R


def get_model(name, max_iter):
    if name == "SSAN_R":
        model = SSAN_R(max_iter=max_iter)
        return model
    elif name == "SSAN_M":
        model = SSAN_M(max_iter=max_iter)
        return model