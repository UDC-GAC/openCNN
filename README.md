# OpenCNN
A winograd’s minimal filtering algorithm implementation in CUDA
## Requirements
 - CUDA Toolkit
 - cuDNN
 - CMake

## Build the project
```
git clone https://github.com/UDC-GAC/openCNN.git
cd openCNN
```
GPU architecture code must be specified inside the Makefile before compiling.

```
make
```

Compile time macros have been defined for testing purposes. They can be specified in the MakeFile according to the following values:
```
$OUT (builds an specific output storage and transform version):
  - BASE: baseline layout
  - OPTSTS64 (default): optSTS64 layout
  - OUTSTS64_CMP: optSTS64_compact layout
  - OUTLDS64: optLDS64 layout
```
## Run examples
(Recommended before time measurement) Lock the clocks:
```
sudo nvidia-smi -i 0 -pm 1
sudo nvidia-smi -lgc 1750 -i 0
```
1. OpenCNN benchmark
```
cd bench
./bench.sh
```
2. Instruction-level microbenchmarking (Require Turing devices). TuringAs (https://github.com/daadaada/turingas) must be installed for running instruction-level microbenchmarking.
```
cd bench/smem/STS
make
./test
```

## Citation

## License
Apache-2.0 License

-- Roberto López Castro
