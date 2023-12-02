- install the requirements
    ```bash
    pip install -r requirements.txt
    ```

- setup the pytorch extension
    ```bash
    cd pytorch_extension
    python setup.py install
    ```

- for those which didn't appear in torch/extension.hm, you need to include the right file like
    at::native::Stride...
    you need to include <ATEN/NATIVE/Stride...>

- made a modification on #include <ATen/native/CompositeRandomAccessorCommon.h>
    C10_HOST_DEVICE
  references& data() const {
    return refs;
  }