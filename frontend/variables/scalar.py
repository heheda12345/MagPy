# learn from https://stackoverflow.com/questions/3238350/subclassing-int-in-python
from frontend.variables.base import Variable

class TrackedScalarMeta(type):
    _dont_wrap = {
        "__str__", "__repr__", "__hash__", "__getattribute__", "__init_subclass__", "__subclasshook__",
        "__reduce_ex__", "__getnewargs__", "__format__", "__sizeof__", "__doc__", "__class__", "__new__",
        "__getattr__", "__setattr__"
    }

    def __new__(typ, name, bases, attrs, **kwargs):
        #  Provide a call to the base class __new__
        print("new:", typ, name, bases, attrs, kwargs)
        base_type = attrs["base_type"]
        # attrs["__new__"] = typ.__class_new__

        cls = type.__new__(typ, name, bases, attrs)

        if "dont_wrap" not in attrs:
            attrs["dont_wrap"] = set()
        print("attr", type(attrs), type(typ._dont_wrap))
        attrs["dont_wrap"].update(typ._dont_wrap)

        base_members = set(dir(base_type))
        print("base_type", base_type)
        print("base_members", base_members)
        typ.wrapped = base_members - set(attrs) - attrs["dont_wrap"]

        for member in typ.wrapped:
            obj = object.__getattribute__(base_type, member)
            if callable(obj):
                print(f"Wrapping {obj.__name__} with {cls.unwrap_wrapper.__name__}")
                wrapped = cls.unwrap_wrapper(obj)
                setattr(cls, member, wrapped)
        return cls

    # def __class_new__(typ, *args, **kw):
    #     "Save boilerplate in our implementation"
    #     return typ.base_type.__new__(typ, *args, **kw)

    # def __new__(cls, name, bases, attrs):
    #     print("TrackedScalarMeta", name, bases, attrs)
    #     return super().__new__(cls, name, bases, attrs)

class TrackedInt(Variable, metaclass=TrackedScalarMeta):
    base_type = int
    def __init__(self, value):
        print("init:", value)
        super().__init__(expr=f'{value}')

    @classmethod
    def unwrap_wrapper(cls, func):
        @cls.functools.wraps(func)
        def unwrap(*args, **kw):
            print("func:", func)
            args = (x.unwrap() if isinstance(x, Variable) else x for x in args)
            kw = {k: (v.unwrap() if isinstance(v, Variable) else v) for k, v in kw.items()}
            return func(*args, **kw)
        return unwrap
    
    def unwrap_impl(self):
        return eval(self.expr)

    def __add__(self, __value: 'TrackedInt') -> 'TrackedInt':
        return TrackedInt(f'{self.expr} + {__value.expr}')
    
    def __str__(self) -> str:
        return f"TrackedInt({self.expr})"

MAGIC_NUMBER = 66666

if __name__ == '__main__':
    a = TrackedInt(10)
    print("a:", a)
    b = TrackedInt(10)
    print("b:", b)
    c = a + b
    print("c:", c)
    d = a - b
    print("d:", d)