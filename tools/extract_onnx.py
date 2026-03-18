#!/usr/bin/env python3
"""
Extract ONNX model graph and weights for SpawnDev.ILGPU.ML.

Produces:
  model_graph.json  — Graph structure (nodes, inputs, outputs, initializers)
  weights_fp16.bin  — FP16 weight blob (256-byte aligned)
  manifest_fp16.json — Weight tensor manifest (name → offset, shape, elements, bytes, dtype)

Usage:
  pip install onnx numpy
  python extract_onnx.py model.onnx output_dir/

The output files can be loaded by InferenceSession.CreateAsync() in the browser.
"""

import sys
import os
import json
import struct
import numpy as np

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model.onnx> <output_dir>")
        sys.exit(1)

    import onnx
    from onnx import numpy_helper

    model_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {model_path}...")
    model = onnx.load(model_path)
    graph = model.graph

    # ── Extract graph structure ──
    model_graph = {
        "name": graph.name or os.path.splitext(os.path.basename(model_path))[0],
        "inputs": [],
        "outputs": [],
        "nodes": [],
        "initializers": {},
    }

    # Initializer names (weights/constants)
    init_names = {init.name for init in graph.initializer}

    # Model inputs (exclude initializers — those are weights, not runtime inputs)
    for inp in graph.input:
        if inp.name in init_names:
            continue
        shape = []
        if inp.type.tensor_type.shape:
            for dim in inp.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else -1)
        model_graph["inputs"].append({"name": inp.name, "shape": shape})

    # Model outputs
    for out in graph.output:
        shape = []
        if out.type.tensor_type.shape:
            for dim in out.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else -1)
        model_graph["outputs"].append({"name": out.name, "shape": shape})

    # Nodes
    for node in graph.node:
        node_dict = {
            "opType": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        # Attributes
        attrs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode("utf-8")
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.TENSOR:
                # Constant tensor attribute — store as initializer
                tensor = numpy_helper.to_array(attr.t)
                tensor_name = f"_attr_{node.output[0]}_{attr.name}"
                init_names.add(tensor_name)
                # Will be added to weights below
        if attrs:
            node_dict["attributes"] = attrs
        model_graph["nodes"].append(node_dict)

    # ── Extract weights ──
    ALIGN = 128  # 256 bytes / 2 bytes per fp16 = 128 fp16 elements
    manifest = {}
    weight_data = bytearray()
    current_offset = 0

    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        shape = list(tensor.shape)

        # Align offset
        aligned = ((current_offset + ALIGN - 1) // ALIGN) * ALIGN
        weight_data.extend(b'\x00' * (aligned - current_offset))
        current_offset = aligned

        # Convert to FP16
        fp16_data = tensor.astype(np.float16).tobytes()
        weight_data.extend(fp16_data)

        elements = int(np.prod(shape)) if len(shape) > 0 else 1
        manifest[init.name] = {
            "offset": current_offset,
            "shape": shape,
            "elements": elements,
            "bytes": len(fp16_data),
            "dtype": "fp16"
        }
        model_graph["initializers"][init.name] = shape
        current_offset += len(fp16_data)

    # ── Write outputs ──
    graph_path = os.path.join(output_dir, "model_graph.json")
    with open(graph_path, "w") as f:
        json.dump(model_graph, f, indent=2)
    print(f"  Graph: {graph_path} ({len(model_graph['nodes'])} nodes, {len(model_graph['initializers'])} initializers)")

    manifest_path = os.path.join(output_dir, "manifest_fp16.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path} ({len(manifest)} tensors)")

    weights_path = os.path.join(output_dir, "weights_fp16.bin")
    with open(weights_path, "wb") as f:
        f.write(weight_data)
    print(f"  Weights: {weights_path} ({len(weight_data) / 1024 / 1024:.1f} MB)")

    print(f"\nDone! {len(model_graph['nodes'])} nodes, {len(manifest)} weights, "
          f"{sum(v['elements'] for v in manifest.values()):,} parameters")

    # Print op type summary
    from collections import Counter
    op_counts = Counter(n["opType"] for n in model_graph["nodes"])
    print("\nOperator summary:")
    for op, count in op_counts.most_common():
        print(f"  {op}: {count}")

if __name__ == "__main__":
    main()
