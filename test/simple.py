from frontend.compile import compile
from frontend.c_api import finalize


# a = 1
def f(b):
    print("=============runnning f==============")
    return b + 1


def g(b):
    print("=============runnning g==============")
    return b + 1


compiled_f = compile(f)
print(compiled_f(2), "should be 3")
print(compiled_f(3), "should be 4")
compiled_g = compile(g)
print(compiled_g(2), "should be 3")
print(compiled_g(3), "should be 4")

finalize()
