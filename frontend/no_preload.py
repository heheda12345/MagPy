import os
from typing import Any


class NO_LD_PRELOAD_CTX:
    old_ld_preload: str = ''

    def __enter__(self) -> None:
        if 'LD_PRELOAD' in os.environ:
            self.old_ld_preload = os.environ['LD_PRELOAD']
            del os.environ['LD_PRELOAD']

    def __exit__(self, *args: Any) -> None:
        if self.old_ld_preload:
            os.environ['LD_PRELOAD'] = self.old_ld_preload
