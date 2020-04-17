"""
Script to create `vecs.npy` and `vecs.vocab` from files with the following format:
<row_num> <dim>
<word_1> <vector_1>
...
<word_n> <vector_n>
"""

import numpy as np
import sys

fh = open(sys.argv[1])    # input file
foutname = sys.argv[2]  # output file path
first = fh.readline()
size = list(map(int, first.strip().split()))

wvecs = np.zeros((size[0], size[1]), float)

vocab = []
for i, line in enumerate(fh):
    line = line.strip().split()
    vocab.append(line[0])
    wvecs[i, ] = np.array(list(map(float, line[1:])))

np.save(foutname + ".npy", wvecs)
with open(foutname + ".vocab", "w") as outf:
   outf.write(" ".join(vocab))
