import setuptools

setuptools.setup(
    name='frontend',
    version='0.0.0',
    packages=setuptools.find_packages('.', exclude=['test']),
    include_dirs=['frontend'],
    ext_modules=[
        setuptools.Extension('frontend.c_api', [
            'frontend/csrc/frame_evaluation.cpp', 'frontend/csrc/opcode.cpp',
            'frontend/csrc/parse_types.cpp'
        ],
                             language='c++',
                             define_macros=[('LOG_CACHE', 'None')])
    ],
)
