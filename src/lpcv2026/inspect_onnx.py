"""Inspect graph metadata and QDQ encodings without assuming an API bit label."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def inspect_graph(path):
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    initializers = {value.name: value for value in graph.graph.initializer}
    counts = Counter()
    for node in graph.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        scale = (
            numpy_helper.to_array(initializers[node.input[1]])
            if node.input[1] in initializers
            else None
        )
        zero = (
            numpy_helper.to_array(initializers[node.input[2]])
            if len(node.input) > 2 and node.input[2] in initializers
            else None
        )
        source = initializers.get(node.input[0])
        kind = "parameter" if source is not None else "intermediate"
        axis = next((int(attr.i) for attr in node.attribute if attr.name == "axis"), 1)
        granularity = (
            "unknown"
            if scale is None
            else ("per_tensor" if scale.size == 1 else f"per_axis_{axis}")
        )
        dtype = str(zero.dtype) if zero is not None else "implicit_uint8_or_dynamic"
        zero_mode = (
            "all_zero"
            if zero is not None and not np.any(zero)
            else "nonzero_or_unknown"
        )
        counts[f"{node.op_type}/{kind}/{dtype}/{granularity}/{zero_mode}"] += 1

    def interface(value):
        tensor = value.type.tensor_type
        return {
            "name": value.name,
            "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": [dim.dim_param or dim.dim_value for dim in tensor.shape.dim],
        }

    return {
        "schema_version": 1,
        "opsets": {op.domain or "ai.onnx": op.version for op in graph.opset_import},
        "inputs": [interface(v) for v in graph.graph.input],
        "outputs": [interface(v) for v in graph.graph.output],
        "operator_counts": dict(
            sorted(Counter(n.op_type for n in graph.graph.node).items())
        ),
        "qdq_encoding_counts": dict(sorted(counts.items())),
        "note": "Initializer vs intermediate classification is structural; all-zero zero-points indicate zero-centered encodings, not proof of every quantizer's range-selection policy.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = inspect_graph(args.model)
    payload = json.dumps(record, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
