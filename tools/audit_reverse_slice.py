# Which models contain a REVERSED slice - and were therefore silently wrong before 5.2.4?
#
#   python tools/audit_reverse_slice.py [root ...]
#
# WHY: `SliceOperator.Execute` clamped a negative step's `ends` to 0, so `x[..., ::-1]` wrote NOTHING and
# left an all-zeros buffer of the correct shape. Nothing threw, and every downstream shape check passed.
# That means the blast radius is not "the VAD" - it is every model in the tree that reverses an axis, and
# the only way to know which is to look. Found by Silero VAD; this asks what else was carrying it.
#
# A Slice is reversed when any entry of its `steps` input is negative. `steps` is input 4 and is virtually
# always an initializer or a Constant node, so it can be read statically.
import os
import sys
import onnx
import onnx.numpy_helper as nh

roots = sys.argv[1:] or ["."]

def steps_of(model, node):
    """The `steps` input of a Slice node, when it is statically knowable."""
    if len(node.input) < 5 or not node.input[4]:
        return None                      # no steps input at all -> all steps are 1
    name = node.input[4]
    for init in model.graph.initializer:
        if init.name == name:
            return nh.to_array(init).ravel().tolist()
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    return nh.to_array(a.t).ravel().tolist()
    return "dynamic"


scanned = affected = 0
findings = []
for root in roots:
    for dirpath, _, files in os.walk(root):
        # Build output holds copies; reporting each twice only makes the list harder to act on.
        if any(part in dirpath for part in (os.sep + "bin", os.sep + "obj", os.sep + "node_modules")):
            continue
        for f in files:
            if not f.endswith(".onnx"):
                continue
            path = os.path.join(dirpath, f)
            try:
                model = onnx.load(path, load_external_data=False)
            except Exception as e:
                print(f"  ?? {path}: {type(e).__name__}")
                continue
            scanned += 1
            hits = []
            for node in model.graph.node:
                if node.op_type != "Slice":
                    continue
                st = steps_of(model, node)
                if st == "dynamic":
                    hits.append((node.name or node.output[0], "dynamic steps - cannot tell statically"))
                elif st and any(v < 0 for v in st):
                    hits.append((node.name or node.output[0], f"steps={st}"))
            if hits:
                affected += 1
                findings.append((path, hits))

print(f"scanned {scanned} onnx model(s); {affected} contain a reversed or dynamic-step Slice\n")
for path, hits in findings:
    print(path)
    for name, why in hits:
        print(f"    {why:48s} {name}")
if not findings:
    print("No model in this tree reverses a slice axis.")
