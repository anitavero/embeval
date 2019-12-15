import numpy as np
from typing import List
import re


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
             'ITALIC': '\33[3m',
             'UNDERLINE': '\033[4m',
             'END': '\033[0m'}

def pfont(font, value):
    return PrintFont[font.upper()] + str(value) + PrintFont['END']

#### LaTeX Font ####

LaTeXFont = {'BLUE': '\color{blue}{',
             'RED': '\color{red}{',
             'BOLD': '\\textbf{',
             'ITALIC': '\\textit{',
             'END': '}'}

def pfont(fonts: List[str], value: str, format):
    """Wrap string in font code.
    :param fonts: list of font names, eg. ['red', 'bold']
    :param value: string to wrap in font
    """
    for font in fonts:
          value = format[font.upper()] + str(value) + format['END']
    return value


def latex_table_wrapper(table):
    prefix = '\\resizebox{\\textwidth}{!}{'
    suffix = '\n}'
    return prefix + table + suffix


def latex_table_post_process(table, bottomrule_row_ids: List[int] = []):
    """Add separator lines and align width to page.
    :param bottomrule_row_ids: Row indices (without header) below which we put a separator line.
    """
    table = latex_table_wrapper(table)

    # Insert lines between rows belonging to different modalities (Ling, Vis, MM)
    if bottomrule_row_ids:
        newline = '\\\\'
        rows = table.split(newline)
        rows[0] = re.sub('\\\\hline', '\\\\toprule', rows[0])
        rows[1] = re.sub('\\\\hline', '\\\\midrule', rows[1])
        for r in bottomrule_row_ids:
            r += 1  # Omit header
            rows[r+1] = '\n\\bottomrule' + rows[r+1]
        table = newline.join(rows)

    return table

