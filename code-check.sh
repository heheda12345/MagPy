#!/bin/bash
set -e
# python code style
yapf -r . -d
# python lint check
mypy -p frontend
# C++ code style
clang-format --style=file -n --Werror frontend/csrc/*

echo "check passed"