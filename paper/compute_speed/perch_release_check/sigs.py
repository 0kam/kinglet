# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Print the signatures, outputs, large variables and op counts of a Perch 2.0 SavedModel. Usage: python -I sigs.py <savedmodel_dir>"""
import sys, tensorflow as tf
d = sys.argv[1]
m = tf.saved_model.load(d)
print("signatures:", list(m.signatures.keys()))
for k, f in m.signatures.items():
    print(k, "inputs:", f.structured_input_signature)
    print(k, "outputs:", {n: (t.shape, t.dtype) for n, t in f.structured_outputs.items()})
print("object attrs:", [a for a in dir(m) if not a.startswith("_")])
vs = m.variables if hasattr(m, "variables") else []
print("n variables", len(vs))
for v in vs:
    if 14795 in v.shape or v.shape.num_elements() > 500000:
        print(v.name, v.shape)
# ops in signature graph
g = m.signatures["serving_default"].graph
ops = {}
for fn in g.as_graph_def().library.function:
    for n in fn.node_def: ops[n.op] = ops.get(n.op, 0) + 1
for n in g.as_graph_def().node: ops[n.op] = ops.get(n.op, 0) + 1
print("ops:", ops)
