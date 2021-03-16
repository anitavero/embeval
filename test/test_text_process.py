import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/source')
from source.text_process import *
from nltk.metrics.association import _log2


def test_w_ppmi():
    pmi_measures = BigramPMIVariants()
    assert pmi_measures.w_ppmi(1.0, (3, 3), 4) == 0
    assert pmi_measures.w_ppmi(1.0, (3, 1), 4) == _log2(1/4) - _log2((3/4) * 1**0.75 / (3**0.75 + 1**0.75))