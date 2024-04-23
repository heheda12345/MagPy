from typing import Callable, Any, Union

CONFIG = {
    "backend": "inductor",  # Union[str, Callable[..., Any]]
    "debug": True,
    "miss_threshold": 3,
    "dynshape": False,
    "enable_fallback": False,
}


def set_config(key: str, value: Any) -> None:
    CONFIG[key] = value


def get_config(key: str) -> Any:
    return CONFIG[key]