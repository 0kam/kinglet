# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Print inputs, outputs, parameter count and largest tensors of ONNX conversions of Perch 2.0. Usage: python -I onnxinfo.py <model.onnx> [...]"""
import sys, numpy as np, onnx
for p in sys.argv[1:]:
    m = onnx.load(p, load_external_data=False)
    g = m.graph
    inits = [(i.name, list(i.dims), onnx.TensorProto.DataType.Name(i.data_type)) for i in g.initializer]
    tot = sum(int(np.prod(d)) for _, d, _ in inits)
    print("==", p.split("/")[-1], "| inputs", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input],
          "| outputs", [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in g.output])
    print("   n_init", len(inits), "total params", tot)
    for n, d, t in sorted(inits, key=lambda x: -int(np.prod(x[1])))[:6]:
        print("  ", n[:80], d, t)
    print("   any 14795/class-dim tensors:", [(n[:60], d) for n, d, _ in inits if 14795 in d or (len(d) == 3 and d[1] == 1536)])
