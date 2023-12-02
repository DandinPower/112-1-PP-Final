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