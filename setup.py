import setuptools

setuptools.setup(
    name='frontend',
    version='0.0.0',
    packages=['frontend'],
    ext_modules=[setuptools.Extension('frontend.c_api', ['frontend/frame_evaluation.c'])],
)