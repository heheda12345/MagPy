# DeepVisor
DeepVisor is a JIT compiler for PyTorch programs. It can extract the operator graph from PyTorch programs and optimize the graph with a wide range of deep learning graph compilers.

# Installation
DeepVisor now supports Python 3.9. The support of other Python versions is working in progress.

1. Install CUDA. CUDA 11.8 is recommended.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    ```
3. Install DeepVisor:
    ```bash
    pip install -e .
    ```
4. Compile a shared library to disable Python integer cache by LD_PRELOAD. This script will generates a ``ldlong.v3.9.12.so'' file in build/ directory. You need to set the LD_PRELOAD environment variable to this file when running the PyTorch program.
    ```bash
    cd scripts
    ./compile_longobj.sh
    ```

# Example Usage

The following script compiles and runs a simple PyTorch program with DeepVisor.

```python
LD_PRELOAD=build/ldlong.v3.9.12.so python test/example.py
```

# Citation
If you find DeepVisor useful in your research, please consider citing the following paper:

> DeepVisor: Effective Operator Graph Instantiation for Deep Learning by Execution State Monitoring. Will be appeared in USENIX ATC'24.

