#!/bin/bash
set -e
source /opt/spack/share/spack/setup-env.sh
spack unload
spack load gcc@11.3.0 /ohhhwxj
spack load python@3.9.12%gcc@=11.3.0

set -x
BUILD_DIR="${BUILD_DIR:-`pwd`/../build}"
CPYTHON_DIR=${BUILD_DIR}/cpython
if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi
if [ ! -d $CPYTHON_DIR ]; then
    git clone git@github.com:python/cpython.git $CPYTHON_DIR
fi
TAG=v3.9.12
if [ ! -f $BUILD_DIR/ldlong.${TAG}.so ]; then
    SCRIPT_DIR=`pwd`
    pushd $CPYTHON_DIR
    git checkout $TAG
    ./configure '--without-pydebug' '--enable-shared' '--without-ensurepip' '--with-openssl=/usr' '--with-dbmliborder=gdbm' '--with-system-expat' '--with-system-ffi' '--enable-loadable-sqlite-extensions' 'CFLAGS=-fPIC'
    make clean
    make -j
    git apply ${SCRIPT_DIR}/longobject.${TAG}.patch
    gcc -c -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC   -std=c99 -Wextra -Wno-unused-result -Wno-unused-parameter -Wno-missing-field-initializers -Werror=implicit-function-declaration -fvisibility=hidden -I${CPYTHON_DIR}/Include/internal -I${CPYTHON_DIR} -I${CPYTHON_DIR}/Include -DPy_BUILD_CORE -o ${BUILD_DIR}/ldlong.o ${CPYTHON_DIR}/Objects/longobject.${TAG}.c
    gcc -shared ${BUILD_DIR}/ldlong.o -o ${BUILD_DIR}/ldlong.${TAG}.so -ldl
    git checkout -- ${CPYTHON_DIR}/Objects/longobject.c
    popd
fi
