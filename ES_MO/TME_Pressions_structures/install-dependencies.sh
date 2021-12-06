#!/bin/sh

# Alex Coninx
# ISIR - Sorbonne Universite / CNRS
# 7/10/2019

# Dependecies : boost, sdl, C++11 compatible compiler, python3, pip3, python2.7,
# git
set -o errexit
set -o nounset
set -o pipefail

BASE_DIR="${PWD}"

# Create dir
mkdir -p "${BASE_DIR}/src"

# 1) Install pybind11
echo
echo "====================================="
echo "===== (1/4) Installing pybind11 ====="
echo "====================================="
echo
cd "${BASE_DIR}/src"
git clone https://github.com/pybind/pybind11.git
cd pybind11
# Install the pybind11 python module
pip3 install .
# Where we can find pybind11 (especially its includes)
PYBIND11_DIR="${BASE_DIR}/src/pybind11"
 
# 2) Install and patch fastsim
echo
echo "===================================================="
echo "===== (2/4) Patching and installing libfastsim ====="
echo "===================================================="
echo
cd "${BASE_DIR}/src"
git clone https://github.com/jbmouret/libfastsim.git
# We need to clone the pyfastsim repository now to get the patch
git clone https://github.com/alexendy/pyfastsim.git
cd libfastsim
# Patch libfastsim
patch -p1 < ../pyfastsim/fastsim-boost2std-fixdisplay.patch
# Build and install
python2.7 ./waf configure --prefix=./install
python2.7 ./waf build
python2.7 ./waf install
# Where we installed fastsim
FASTSIM_DIR="${BASE_DIR}/src/libfastsim/install"

# 3) Install pyfastsim
echo
echo "======================================"
echo "===== (3/4) Installing pyfastsim ====="
echo "======================================"
echo
cd "${BASE_DIR}/src/pyfastsim"
CPPFLAGS="-I\"${PYBIND11_DIR}/include\" -I\"${FASTSIM_DIR}/include\"" LDFLAGS="-L\"${FASTSIM_DIR}/lib\"" pip3 install .

# 1) Install fastsim_gym
echo
echo "====================================="
echo "===== (4/4) Installing fastsim_gym ====="
echo "====================================="
echo
cd "${BASE_DIR}/src"
git clone git@github.com:alexendy/fastsim_gym.git
cd fastsim_gym
# Install the fastsim_gym python module
pip3 install .
