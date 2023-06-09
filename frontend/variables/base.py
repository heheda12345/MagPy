class Variable:
    import functools

    def __init__(self, expr=None, unwrap=False):
        self.expr = expr
        self.evaled = False
        if unwrap:
            self.value = self.unwrap()
        else:
            self.value = None

    def unwrap(self):
        print("unwrap is called on", self)
        if self.evaled:
            return self.value
        result = self.unwrap_impl()
        self.evaled = True
        return result

    def unwrap_impl(self):
        raise NotImplementedError(
            "eval_impl: should be implemented by subclasses")
