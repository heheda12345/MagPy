import torch
import torch.nn as nn
from frontend.compile import compile, reset

import torch


def f(a, b, c):
    return a + b + c


compiled_f = compile(f)

a = torch.full((1,), 1.0)
b = torch.full((1,), 2.0)
c = torch.full((1,), 3.0)
print(compiled_f(a, b, c))
print(compiled_f(a, b, c))