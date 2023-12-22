# Sparse Matrix Pytorch Extension

## Introduction

This project is for the course project of Parallel Programming in NYCU CSIE. We implement the sparse matrix multiplication Parallel Optimization in PyTorch extension. We also provide a benchmark tool to compare the performance of different implementations. Currently, we have implemented the following methods:

1. naive implementation
2. refactor to parallel friendly structure implementation (still serial)
3. OpenMP implementation by ![Frankie](https://github.com/frankie699) and ![DandinPower](https://github.com/DandinPower)
4. OpenMP + memory efficient implementation by ![Leo](https://github.com/leo27945875) 
5. std::thread implementation by ![Frankie](https://github.com/frankie699) and ![DandinPower](https://github.com/DandinPower)

in most of the implementation, the OpenMP + memory efficient implementation is the fastest one and also the most memory efficient one. You can check the benchmark result after you run the benchmark scripts, or you can check our benchmark result in the logs folder.

Our Evaluation platform is:

1. AMD Ryzen 9 5950X 16-Core Processor (16cores)
2. Ubuntu 22.04 LTS
3. python 3.10.12

## Prerequisites

Before you can compile the PyTorch extension, you need to install the necessary requirements. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Compiling the PyTorch Extension

After installing the prerequisites, navigate to the pytorch_extension directory and run the setup file to compile the PyTorch extension:

```bash
cd pytorch_extension
bash run.sh
```

## Running the Tests

This project uses the pytest module for unit testing. To test the PyTorch extension implementation, run the following command:

```bash
pytest ./test
```

## Benchmark Tool

After you compile the PyTorch extension, and run the tests, you can use our benchmark code to compare different implementations of sparse matrix multiplication. We have provided 2 type of benchmark strategies:

1. Benchmarking the SPMM function with end-to-end time, with different density, different threads and matrix size.
    ```bash
    bash benchmark.sh
    ```

2. Benchmarking the SPMM function with MNIST test dataset, with different density, different threads.
    ```bash
    bash mnist_benchmark.sh
    ```

3. run all benchmarks
    ```bash
    bash all_benchmark.sh
    ``` 

for each benchmark, you can change the parameters in the script file. You can see the parameters description in the script file. 

**Note**: you must set the threads number fit into your CPU core number, also you need to care about the core has same performance or not. For example, in intel CPU, the performance core will faster than the efficiency core.

## Notes

### pytorch extension include issue

- for those which pytorch built-in function didn't include by torch/extension.h, you need to include the right file like
    ```c++
    at::native::StridedRandomAccessor
    ``` 
    you need to include
    ```c++
    #include <ATen/native/StridedRandomAccessor.h>
    ```

## Reference

- pytorch
    - [pytorch extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
    - [pytorch SparseTensorMath.cpp](https://github.com/pytorch/pytorch/blob/729ac7317a50a6a195b324cf6cefd748bf4f5498/aten/src/ATen/native/sparse/SparseTensorMath.cpp#L1379)
    - [pytorch NativeFunctions.yaml](https://github.com/pytorch/pytorch/blob/729ac7317a50a6a195b324cf6cefd748bf4f5498/aten/src/ATen/native/native_functions.yaml#L4073)
    - [pytorch SparseMatmul.cpp](https://github.com/pytorch/pytorch/blob/729ac7317a50a6a195b324cf6cefd748bf4f5498/aten/src/ATen/native/sparse/SparseMatMul.cpp#L89)