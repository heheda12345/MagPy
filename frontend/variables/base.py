from dataclasses import dataclass
from typing import Optional


@dataclass
class Guard:
    code: list[str]
    imports: Optional[set[str]] = None

    def add(self, other: 'Guard') -> None:
        self.code.extend(other.code)
        if self.imports is None and other.imports is None:
            return
        if self.imports is None:
            self.imports = set()
        self.imports.update(other.imports)

    def get_imports(self, indent) -> str:
        imports = self.imports or set()
        print("imports:", self.imports)
        return '\n'.join(
            f'{"    " * indent}import {module_name}' for module_name in imports)
