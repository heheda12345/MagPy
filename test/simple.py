from frontend.compile import compile
from frontend.c_api import finalize


# a = 1
def f(b):
    return b + 1


def g(b):
    return b + 1


def k(b):
    return (b + 1) // 2


compiled_f = compile(f)
print(compiled_f(3), "should be 4")
print(compiled_f(3), "should be 4")
print(compiled_f(4), "should be 5")
print(compiled_f(4), "should be 5")
compiled_g = compile(g)
print(compiled_g(3), "should be 4")
print(compiled_g(3), "should be 4")
compiled_k = compile(k)
print(compiled_k(3), "should be 2")
print(compiled_k(3), "should be 2")
# FIXME: finalize crashes
finalize()
