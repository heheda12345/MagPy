#!/bin/bash
set -e
# python lint check
mypy -p frontend
# python code style
yapf -r . -d
# C++ code style
clang-format --style=file -n --Werror frontend/csrc/*

echo "check passed"