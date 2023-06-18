from typing import Optional, Any


class Variable:

    def __init__(self, expr: Optional[str] = None, unwrap: bool = False):
        self.expr = expr
        self.evaled = False
        if unwrap:
            self.value = self.unwrap()
        else:
            self.value = None

    def unwrap(self) -> Any:
        print("unwrap is called on", self)
        if self.evaled:
            return self.value
        result = self.unwrap_impl()
        self.evaled = True
        return result

    def unwrap_impl(self) -> Any:
        raise NotImplementedError(
            "eval_impl: should be implemented by subclasses")
