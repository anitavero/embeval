"""
Script to create `vecs.npy` and `vecs.vocab` from files with the following format:
<row_num> <dim>
<word_1> <vector_1>
...
<word_n> <vector_n>
"""

import numpy as np
import argh


def main(input_file, output_file):
    fh = open(input_file, 'r', errors='replace')    # input file  TODO: try better encoding
    foutname = output_file  # output file path
    first = fh.readline()
    size = list(map(int, first.strip().split()))

    wvecs = np.zeros((size[0], size[1]), float)

    vocab = []
    for i in range(size[0]):
        ln = fh.readline()
        line = ln.strip().split()
        vocab.append(line[0])
        wvecs[i, ] = np.array(list(map(float, line[1:])))

    np.save(foutname + ".npy", wvecs)
    with open(foutname + ".vocab", "w") as outf:
       outf.write(" ".join(vocab))


if __name__ == '__main__':
    argh.dispatch_command(main)
