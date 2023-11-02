from frontend.compile import compile, reset
import torch


def inplace_add(a, b):
    a += b
    return a

x = 3
y = torch.full((4,), 5.0)
# y = 3
compiled = compile(inplace_add)
print(compiled(y, x))
print(compiled(y, x))
print(inplace_add(y, x))