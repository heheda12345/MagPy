from frontend.no_preload import NO_LD_PRELOAD_CTX

no_ld_preload = NO_LD_PRELOAD_CTX()


# content of plugins/example_plugin.py
def pytest_configure(config):
    no_ld_preload.__enter__()


def pytest_unconfigure(config):
    no_ld_preload.__exit__()