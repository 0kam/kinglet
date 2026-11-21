# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""List the variables of Perch 2.0 SavedModels and any tensor with a 14,795-class dimension. Usage: python -I listvars.py <savedmodel_dir> [...]"""
import sys, numpy as np, tensorflow as tf
for d in sys.argv[1:]:
    print("=====", d)
    vs = tf.train.list_variables(d + "/variables/variables")
    tot = 0
    for n, s in vs:
        p = int(np.prod(s)) if s else 1
        tot += p
        if p > 100000 or any(x in (14795, 59180, 1536*0+59180) for x in s):
            print(n, s, p)
    print("nvars", len(vs), "total params", tot)
    hits = [(n, s) for n, s in vs if 14795 in s or 59180 in s]
    print("14795/59180 tensors:", hits)
