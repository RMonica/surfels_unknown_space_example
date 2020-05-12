Surfels Unknown Space Example
=============================

Basic example to generate a results for Scenario 1 and Scenario 2, reported in the associated publication:

- R. Monica and J. Aleotti, "Surfel-Based Incremental Reconstruction of the Boundary between Known and Unknown Space", IEEE Transactions on Visualization and Computer Graphics

Dependencies
------------

- Ubuntu 16.04 or 18.04
- OpenCL C++ headers, version 2.0
- OpenCL runtime, minimum version 1.1
- PCL (Point Cloud Library)
- Eigen3
- CMake
- unzip and wget (to download the dataset)

To install PCL, Eigen3 and CMake, run the script as root:
```
  # install_dependencies.sh
```

OpenCL installation depends on your OpenCL device (preferably an NVIDIA GPU).

If you are using an Intel CPU as OpenCL device, please uncomment the line
```
  config.opencl_use_intel = true;
```
in `src/surfels_unknown_space_example.cpp` to activate a workaround.

**Note**: This software was not tested on ATI/AMD.

**Note**: on most systems, after installing the OpenCL runtime, a suitable OpenCL C++ library and headers can be installed with the APT packages `ocl-icd-opencl-dev` and `opencl-clhpp-headers`.

Build
-----

Standard CMake build is carried out by the script:

```
  build.sh
```

Dataset download
----------------

Run

```
  download_dataset.sh
```

to download (from `rimlab.ce.unipr.it`) and decompress the test dataset into the `data` folder.  
About 7 GB will be downloaded. You will need about 30 GB of free disk space.

Running the example
-------------------

Run

```
  run.sh
```

This can take from a few minutes to a few hours, depending on your system.

The following files will be created:

- `data/Scenario_1/output_cloud.pcd`: surfel-based boundary for Scenario 1, in PCL PCD format.
- `data/Scenario_1/output_cloud.ply`: surfel-based boundary for Scenario 1 as PLY mesh. Surfels are represented by hexagons.
- `data/Scenario_2/output_cloud.pcd`: surfel-based boundary for Scenario 2, in PCL PCD format.
- `data/Scenario_2/output_cloud.ply`: surfel-based boundary for Scenario 2 as PLY mesh. Surfels are represented by hexagons.

These surfel clouds are shown in the associated publication in Figure 12, top two rows.

The script 
```
  compare.sh
```
*in theory* should output the data used to generate the left three columns of Table 2 in the associated publication, for Scenario 1 and 2. Unfortunately, the current parallel implementation of the `surfels_unknown_space` algorithm is non-deterministic, so actual numbers may vary.