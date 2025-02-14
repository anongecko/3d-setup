#!/bin/bash
PYINCLUDES=$(python -m pybind11 --includes)
echo "$PYINCLUDES"
g++ -O3 -Wall -shared -std=c++11 -fPIC $PYINCLUDES mesh_processor.cpp -o mesh_processor.so -lpython3.10
