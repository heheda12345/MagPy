import torch
from frontend.compile import compile, reset

# def f(x, y):
#     z = x + 1
#     if z > 0:
#         return y + 1
#     else:
#         return y + 2

# compiled_f = compile(f)
# print(compiled_f(0, 0), "should be 1")
# print(compiled_f(0, 0), "should be 1")
# print(compiled_f(-1, 0), "should be 2")
# print(compiled_f(-1, 0), "should be 2")


def f(x):
    return (x + 1) // 2 + 1


compiled_f = compile(f)
print(compiled_f(1), "should be 2")
print(compiled_f(1), "should be 2")
