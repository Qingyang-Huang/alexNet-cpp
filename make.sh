#!/bin/bash

rm -rf build/*
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
# cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

