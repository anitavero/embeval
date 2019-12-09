import numpy as np


def join_struct_arrays(arrays):
    new_dt = list(set(sum([a.dtype.descr for a in arrays], [])))
    joint = np.zeros(arrays[0].shape, dtype=new_dt)
    for a in arrays:
        for nm in a.dtype.names:
            joint[nm] = a[nm]
    return joint


def get_vec(word, embeddings, vocab):
    return embeddings[np.where(vocab == word)[0][0]].reshape(1, -1)


#### Terminal Font ####

PrintFont = {'PURPLE': '\033[95m',
             'CYAN': '\033[96m',
             'DARKCYAN': '\033[36m',
             'BLUE': '\033[94m',
             'GREEN': '\033[92m',
             'YELLOW': '\033[93m',
             'RED': '\033[91m',
             'BOLD': '\033[1m',
             'UNDERLINE': '\033[4m',
             'END': '\033[0m'}

def pfont(font, value):
    return PrintFont[font.upper()] + str(value) + PrintFont['END']

#### LaTeX Font ####

LaTeXFont = {'BLUE': '\color{blue}{',
             'RED': '\color{red}{',
             'BOLD': '\textbf{',
             'ITALIC': '\textit{',
             'END': '}'}

def pfont(font, value, format):
    return format[font.upper()] + str(value) + format['END']

