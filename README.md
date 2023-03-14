# Mean curvature motion and levelset motion for cryo-ET

Three-dimensional visualization of biological samples is essential for understanding their architecture and function. However, it is often challenging due to the macromolecular crowdedness of the samples and low signal-to-noise ratio of the cryo-electron tomograms. Denoising and segmentation techniques address this challenge by increasing the signal-to-noise ratio and by simplifying the data in images. Here, mean curvature motion is presented as a method that can be applied to segmentation results, created either manually or automatically, to significantly improve both the visual quality and downstream computational handling. Mean curvature motion is a process based on nonlinear anisotropic diffusion that smooths along edges and causes high-curvature features, such as noise, to disappear. In combination with level-set methods for image erosion and dilation, the application of mean curvature motion to electron tomograms and segmentations removes sharp edges or spikes in the visualized surfaces, produces an improved surface quality, and improves overall visualization and interpretation of the three-dimensional images.

If you use these tools, please cite:

* Frangakis AS. [**Mean curvature motion facilitates the segmentation and surface visualization of electron tomograms.**](https://www.sciencedirect.com/science/article/abs/pii/S104784772200003X) J Struct Biol. 2022 Mar;214(1):107833. doi: [10.1016/j.jsb.2022.107833](https://doi.org/10.1016/j.jsb.2022.107833). Epub 2022 Jan 21. PMID: [35074502](https://pubmed.ncbi.nlm.nih.gov/35074502/).

**Contents:**

* [Examples](#examples)
  * [Smoothing a hand segmentation with staircase-artifacts and holes](#smoothing-a-hand-segmentation-with-staircase-artifacts-and-holes)
  * [Reducing noise in a membrane segmentation](#reducing-noise-in-a-membrane-segmentation)
* [Tools](#tools)
  * [mcm_3D](#mcm-3D)
  * [mcm_levelset](#mcm_levelset)
  * [mcm_close](#mcm_close)
  * [mcm_open](#mcm_open)
  * [geodesic_trace](#geodesic_trace)
* [Installation](#installation)

---

## Examples

### Smoothing a hand segmentation with staircase-artifacts and holes

<p align="center">
<img src="https://user-images.githubusercontent.com/6641113/224759832-2812e8de-c21c-4d62-a2af-fce798b3bc3d.gif" alt="Movie mcm_close.py"/>
</p>

Existing hand segmentations in EM/MRC-Format can be smoothed and filled in using [mcm_close.py](#mcm_close). The tool is conceptually similar to binary morphological closing and gauss filtering, but should yield better results with close to no parameter tuning. The program diffuses the input signal depending on the local image gradient (adjust strength with `-a`/`--alpha`) and local mean curvature (adjust strength with `-b`/`--beta`). 

The volume shown above is a binary segmemtation of size 900 x 700 x 250. The movie was generated using the following parameters (replace ITERNUM with the iteration number of your choice). The `--binary` option ensures rescaling to data range of [0 1] after diffusion.

```shell
# Run mcm_close for ITERNUM iterations.
mcm_close.py -i vol_in.mrc -o vol_out.mrc -p ITERNUM -a 0.5 -b 0.5 --binary
```

Optionally, it is possible to threshold the output image, to obtain a new binary output:

```shell
# Run mcm_close for ITERNUM iterations and threshold
mcm_close.py -i vol_in.mrc -o vol_out.mrc -p ITERNUM -a 0.5 -b 0.5 --binary --threshold 0.5
```


For non-binary input images, turn off rescaling using `--no-binary`

```shell
# Run mcm_close for ITERNUM iterations, no rescaling
mcm_close.py -i vol_in.mrc -o vol_out.mrc -p ITERNUM -a 0.5 -b 0.5 --no-binary
```

---

### Reducing noise in a membrane segmentation

<p align="center">
<img src="https://user-images.githubusercontent.com/6641113/224988573-2abdcbf3-5f6c-4de6-a3c0-b222c22fdf2b.gif" alt="Movie mcm_open1.py" width="400"/>
<img src="https://user-images.githubusercontent.com/6641113/224988664-f64e5cda-ec75-42e1-aed4-e62ee23fa259.gif" alt="Movie mcm_open2.py" width="400"/>
</p>

Noisy embrane segmentations or probability maps can be improved by [mcm_open.py](#mcm_open). Following a similar principle as in the example above, levelset-based erosion and mean curvature motion are applied first, followed by levelset based dilation coupled with mean curvature motion. 

The volume shown above is a simulated noisy segmemtation of size 900 x 700 x 250 with values scaled between 0 and 1. The movie was generated using the following parameters (replace ITERNUM with the iteration number of your choice). The `--binary` option ensures rescaling to data range of [0 1] after diffusion.

```shell
# Run mcm_open for ITERNUM iterations.
mcm_open.py -i vol_in.mrc -o vol_out.mrc -p ITERNUM -a 0.5 -b 0.5 --binary
```
As above, the tool can also produce thresholded output or proceed without any rescaling. 

```shell
# Run mcm_open for ITERNUM iterations and threshold
mcm_open.py -i vol_in.mrc -o vol_out.mrc -p ITERNUM -a 0.5 -b 0.5 --binary --threshold 0.5
```

```shell
# Run mcm_open for ITERNUM iterations, no rescaling
mcm_open.py -i vol_in.mrc -o vol_out.mrc -p ITERNUM -a 0.5 -b 0.5 --no-binary
```
---




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

---

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

---

### mcm_open

Morphological opening with levelset and mean curvature motion. 

**<details><summary>Python wrapper</summary><p>**
```shell
mcm_open.py -i input.em -o output.em -p 20 -a 0.5 -b 0.5
```
</p></details>

**<details><summary>Python code</summary><p>**
```python
import pymcm.mcm as mcm

# Alpha needs to be between 0 and 1
alpha = 0.5
assert(0 <= alpha <= 1)

# Erosion
outvol = mcm.mcm_levelset(invol, iterations=20, alpha=-1*alpha, beta=0.5, verbose=True)
invol = outvol.copy()

# Dilation
outvol = mcm.mcm_levelset(invol, iterations=20, alpha=alpha, beta=0.5, verbose=True)

```
</p></details>

---

### mcm_close

Morphological closing with levelset and mean curvature motion. 

**<details><summary>Python wrapper</summary><p>**
```shell
mcm_close.py -i input.em -o output.em -p 20 -a 0.5 -b 0.5
```
</p></details>

**<details><summary>Python code</summary><p>**
```python
import pymcm.mcm as mcm

# Alpha needs to be between 0 and 1
alpha = 0.5
assert(0 <= alpha <= 1)

# Dilation
outvol = mcm.mcm_levelset(invol, iterations=20, alpha=-alpha, beta=0.5, verbose=True)
invol = outvol.copy()

# Erosion
outvol = mcm.mcm_levelset(invol, iterations=20, alpha=-1*alpha, beta=0.5, verbose=True)

```
</p></details>

---

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
conda create -n mcm -c conda-forge python=3.9 scikit-build numpy mrcfile cython cmake=3.23
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
* CMake >= 3.23
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
* CMake >= 3.23
* C/C++ compiler
</p></details>

**<details><summary>CUDA-accelerated applications</summary><p>**
* CMake >= 3.23
* C/C++ compiler
* CUDA toolkit
</p></details>

### Build

**<details><summary>Python package (recommended)</summary><p>**
```shell
conda create -n mcm python=3.9 skbuild numpy mrcfile
conda activate mcm
git clone https://github.com/FrangakisLab/mcm-cryoet.git
cd mcm-cryoet
pip install .
```
</p></details>

**<details><summary>Pure C applications</summary><p>**
```shell
git clone https://github.com/FrangakisLab/mcm-cryoet.git
cd mcm-cryoet
mkdir build; cd build
cmake ..
make
# Executables now in build/bin/
```
</p></details>

**<details><summary>CUDA-accelerated applications</summary><p>**
```shell
git clone https://github.com/FrangakisLab/mcm-cryoet.git
cd mcm-cryoet
mkdir build; cd build
cmake ..
make
# Executables now in build/bin/
```
</p></details>





