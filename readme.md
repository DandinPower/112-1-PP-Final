## Sparse Matrix Pytorch Extension

### Compile and Test
- install the requirements
    ```bash
    pip install -r requirements.txt
    ```

- compile the pytorch extension
    ```bash
    cd pytorch_extension
    python setup.py install
    ```

- run the test
    ```bash
    python main.py
    ```

### Notes

- for those which pytorch built-in function didn't include by torch/extension.h, you need to include the right file like
    ```c++
    at::native::StridedRandomAccessor
    ``` 
    you need to include
    ```c++
    #include <ATen/native/StridedRandomAccessor.h>
    ```

### Reference

- pytorch
    - [pytorch extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
    - [pytorch SparseTensorMath.cpp](https://github.com/pytorch/pytorch/blob/729ac7317a50a6a195b324cf6cefd748bf4f5498/aten/src/ATen/native/sparse/SparseTensorMath.cpp#L1379)
    - [pytorch NativeFunctions.yaml](https://github.com/pytorch/pytorch/blob/729ac7317a50a6a195b324cf6cefd748bf4f5498/aten/src/ATen/native/native_functions.yaml#L4073)
    - [pytorch SparseMatmul.cpp](https://github.com/pytorch/pytorch/blob/729ac7317a50a6a195b324cf6cefd748bf4f5498/aten/src/ATen/native/sparse/SparseMatMul.cpp#L89)