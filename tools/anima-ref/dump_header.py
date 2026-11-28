"""Print the safetensors header of the Anima DiT file (non-block names + block 0 + adapter block 0)."""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_reader import SafeTensors

path = sys.argv[1] if len(sys.argv) > 1 else r'D:\Projects\anima\anima-preview3-base.safetensors'
st = SafeTensors(path)
print('metadata:', st.meta)
print('count:', len(st.names()))
dtypes = {}
for n in st.names():
    dtypes[st.dtype(n)] = dtypes.get(st.dtype(n), 0) + 1
print('dtypes:', dtypes)
for n in sorted(st.names()):
    m = re.match(r'net\.blocks\.(\d+)\.', n)
    if m and m.group(1) != '0':
        continue
    m = re.match(r'net\.llm_adapter\.blocks\.(\d+)\.', n)
    if m and m.group(1) != '0':
        continue
    print(n, st.dtype(n), st.shape(n))
