
from frontend.compile import compile

a = 1
def f(b):
    print("=============runnning f==============")
    return a + b

compiled_f = compile(f)
print(compiled_f(2), "should be 3")
print(compiled_f(3), "should be 4")
a = 2
print(compiled_f(2), "should be 4")
print(compiled_f(3), "should be 5")

