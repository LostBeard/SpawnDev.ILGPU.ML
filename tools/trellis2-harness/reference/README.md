# Golden tensors from the real TRELLIS.2 reference

CPU torch (fp32, sdpa attention), real weights, deterministic inputs. The harness compares our GPU kernels to these.

```
python -m venv venv && venv/Scripts/pip install torch numpy safetensors easydict tqdm   # CPU torch is enough
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.2 <ref>
TRELLIS2_REF=<ref> TRELLIS2_DIR=<weight dir> python golden_ss_flow.py     # writes <weight dir>/golden/ss_flow
dotnet run --project tools/trellis2-harness -c Release -- ssflow
```

The scripts import the reference model classes WITHOUT running `trellis2/__init__` (it pulls in renderers and
nvdiffrast, which need CUDA). Goldens are not committed (hundreds of MB).
