import numpy as np
from typing import List
import re
import pickle
import json
import os
import argparse


def suffixate(s):
    if s != '' and s[0] != '_':
        s = '_' + s
    return s


def tuple_list(arg):
    """List[Tuple[str]] argument type.
        format: whitespace separated str lists, separated by |. eg. 'embs1 embs2 | embs2 embs3 embs4'
    """
    try:
        if '|' in arg:
            tplist = [tuple(t.split()) for t in arg.split('|')]
        else:
            tplist = [tuple(arg.split())]
        return tplist
    except:
        raise argparse.ArgumentTypeError("Tuple list must be whitespace separated str lists, " +
                                         "separated by |. eg. embs1 embs2 | embs2 embs3 embs4")


def hr_time(time, round_n=2):
    """Human readable time."""
    hours = time // 3600 % 24
    minutes = time // 60 % 60
    seconds = round(time % 60, round_n)
    return f'{hours}h {minutes}m {seconds}s'


def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_jl(path):
    articles = []
    for line in open(path, "r"):
        article = json.loads(line)
        articles.append(article)
    return articles


def pkl2json(pkl_file, savedir):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    fname = os.path.basename(pkl_file)
    with open(os.path.join(savedir, fname.split('.')[0] +'.json'), 'w') as f:
        json.dump(data, f)


def join_struct_arrays(arrays):
    new_dt = list(set(sum([a.dtype.descr for a in arrays], [])))
    joint = np.zeros(arrays[0].shape, dtype=new_dt)
    for a in arrays:
        for nm in a.dtype.names:
            joint[nm] = a[nm]
    return joint


def dict2struct_array(d):
    """Convert dict to structured array."""
    dtype = [(k, np.ndarray) for k in d.keys()]
    dim = len(list(d.values())[0])
    ar = np.array(np.empty(dim), dtype=dtype)
    for k, v in d.items():
        ar[k] = np.array(v)
    return ar


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


# def pfont(font, value):
#     return PrintFont[font.upper()] + str(value) + PrintFont['END']


#### LaTeX Font ####

LaTeXFont = {'BLUE': '\color{blue}{',
             'RED': '\color{red}{',
             'BOLD': '\\textbf{',
             'ITALIC': '\\textit{',
             'END': '}'}


def pfont(fonts: List[str], value: str, format):
    """Wrap string in font code.
    :param format: PrintFont or LaTeXFont
    :param fonts: list of font names, eg. ['red', 'bold']
    :param value: string to wrap in font
    """
    for font in fonts:
        value = format[font.upper()] + str(value) + format['END']
    return value


def latex_table_wrapper(table, title, fit_to_page, label):
    prefix = '\\begin{table}[]\n\centering\n'
    if fit_to_page:
        prefix += '\\resizebox{\\textwidth}{!}{\n'
        table = re.sub('\\\end{tabular}', '\\\end{tabular}}', table)
    suffix = '\n\caption{' + title + '}'
    suffix += '\n\label{t:' + label + '}\n\end{table}\n'
    return prefix + table + suffix


def latex_table_post_process(table, bottomrule_row_ids: List[int] = [], title='', fit_to_page=False, label=''):
    """Add separator lines and align width to page.
    :param bottomrule_row_ids: Row indices (without header) below which we put a separator line.
    """
    table = latex_table_wrapper(table, title, fit_to_page, label)

    newline = ' \\\\'
    rows = table.split(newline)
    rows[0] = re.sub('\\\\hline', '\\\\toprule', rows[0])
    rows[1] = re.sub('\\\\hline', '\\\\midrule', rows[1])

    # Insert lines between rows belonging to different modalities (Ling, Vis, MM)
    if bottomrule_row_ids:
        for r in bottomrule_row_ids:
            r += 1  # Omit header
            rows[r + 1] = '\n\\hline' + rows[r + 1]
        table = newline.join(rows)

    return table
