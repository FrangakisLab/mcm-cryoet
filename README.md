# Mean curvature motion and levelset motion for cryo-ET


## Tools

### mcm_3D

Smooth a volume using mean curvature motion only. 


**<details><summary>Python wrapper</summary><p>**
```shell
mcm_3D.py -i input.em -o output.em -p 20
```
</p></details>

**<details><summary>Python code</summary><p>**
```python
import pymcm.mcm as mcm

outvol = mcm.mcm(invol, iterations=20, verbose=True)
```
</p></details>

**<details><summary>Pure C application</summary><p>**
```shell
mcm_3D input.em output.em 20
```
</p></details>

**<details><summary>CUDA-accelerated application</summary><p>**
```shell
mcm_3D_cuda input.em output.em 20
```
</p></details>

### mcm_levelset

Smooth a volume using a combination of mean curvature motion and levelset motion. 

**<details><summary>Python wrapper</summary><p>**
```shell
mcm_levelset.py -i input.em -o output.em -p 20 -a 0.5 -b 0.5
```
</p></details>

**<details><summary>Python code</summary><p>**
```python
import pymcm.mcm as mcm

outvol = mcm.mcm_levelset(invol, iterations=20, alpha=0.5, beta=0.5, verbose=True)
```
</p></details>

**<details><summary>Pure C application</summary><p>**
```shell
mcm_levelset input.em output.em 20 0.5 0.5
```
</p></details>

**<details><summary>CUDA-accelerated application</summary><p>**
```shell
mcm_levelset_cuda input.em output.em 20 0.5 0.5
```
</p></details>

### geodesic_trace

Finds the shortest geodesic path through a binary mask given a start and end point. 

**<details><summary>Python wrapper</summary><p>**
```shell
geodesic_trace.py -i mask.em -ov output_vol.em -ot output_coords -x 10,10,10 -y 30,30,30 -m 10000
```
</p></details>

**<details><summary>Python code</summary><p>**
```python
import pymcm.mcm as mcm

outvol, outtrace = mcm.trace(invol, x, y, maxstep=10000, verbose=True)
```
</p></details>

**<details><summary>Pure C application</summary><p>**
```shell
geodesic_trace mask.em output_vol.em 10 10 10 30 30 30
```
</p></details>

**<details><summary>CUDA-accelerated application</summary><p>**
```shell
geodesic_trace_cuda mask.em output_vol.em 10 10 10 30 30 30
```
</p></details>

## Installation

**<details><summary>TL;DR</summary><p>**
```shell
# Install C/C++ compilers and optionally CUDA 
# e.g. on Ubuntu
sudo apt install build-essentials
sudo apt install nvidia-cuda-dev nvidia-cuda-toolkit

# Installs all other pre-requisites (a little overkill)
conda create -n mcm -c conda-forge python=3.9 scikit-build numpy mrcfile cython cmake=3.18
conda activate mcm

# Build
git clone REPO
cd REPO
pip install .

# MCM-Levelset combi
mcm_levelset.py --help

# MCM alone
mcm_3D.py --help

# Geodesic trace
geodesic_trace.py --help
```
</p></details>


The package contains three flavours of the filters:

1. Pure C programs 
2. C++/CUDA-accelerated versions of the C code.
3. A python extension and scripts wrapping the C and C++/CUDA programs (**recommended**)

All can be built and installed independently. The pure C/C++/CUDA applications read data only in EM format, 
while the python wrappers also accept MRC-files as input and allow setting some additional 
parameters. If the CUDA-accelerated library is found, it will be preferred by the python wrappers.

### Prerequisites

**<details><summary>Python package (recommended)</summary><p>**
* CMake >= 3.18
* C/C++ compiler
* Python >= 3.9
* Python packages: 
  * skbuild >= 0.15
  * numpy 
  * mrcfile
  * Cython
* optional: CUDA toolkit
</p></details>

**<details><summary>Pure C applications</summary><p>**
* CMake >= 3.18
* C/C++ compiler
</p></details>

**<details><summary>CUDA-accelerated applications</summary><p>**
* CMake >= 3.18
* C/C++ compiler
* CUDA toolkit
</p></details>

### Build

**<details><summary>Python package (recommended)</summary><p>**
```shell
conda create -n mcm python=3.9 skbuild numpy mrcfile
conda activate mcm
git clone REPO
cd REPO
pip install .
```
</p></details>

**<details><summary>Pure C applications</summary><p>**
```shell
git clone REPO
cd REPO
mkdir build; cd build
cmake ..
make
# Executables now in build/bin/
```
</p></details>

**<details><summary>CUDA-accelerated applications</summary><p>**
```shell
git clone REPO
cd REPO
mkdir build; cd build
cmake ..
make
# Executables now in build/bin/
```
</p></details>





