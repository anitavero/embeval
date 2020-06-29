import sys, os
sys.path.append(os.getcwd() + '/source')
from source.process_wiki import *
import json
from glob import glob


data_dir = 'test/data/'

def test_contexts_for_freqrange():
    files = glob(data_dir + '*contexts.txt')
    for f in files:
        os.remove(f)


    dist = {'a': 1, 'b': 10, 'c': 20, 'd': 40}
    distribution_file = 'dist.json'
    with open(distribution_file, 'w') as f:
        json.dump(dist, f)
    contexts = 'a b\n' +\
               'b c\n'
    with open(data_dir + 'A/test.contexts', 'w') as f:
        f.write(contexts)

    min_count = 0
    max_count = 50
    cont_file = contexts_for_freqrange(data_dir, distribution_file, min_count, max_count, filename_suffix='')
    with open(cont_file, 'r') as f:
        fq_contexts = f.read()
    assert fq_contexts == contexts


    min_count = 2
    max_count = 50
    cont_file = contexts_for_freqrange(data_dir, distribution_file, min_count, max_count, filename_suffix='')
    with open(cont_file, 'r') as f:
        fq_contexts = f.read()
    assert fq_contexts == 'b c\n'


    contexts = 'a b\n' +\
               'b c\n' +\
               'a d\n' +\
               'b d\n' +\
               'c d\n'
    with open(data_dir + 'A/test.contexts', 'w') as f:
        f.write(contexts)


    min_count = 10
    max_count = 50
    cont_file = contexts_for_freqrange(data_dir, distribution_file, min_count, max_count, filename_suffix='')
    with open(cont_file, 'r') as f:
        fq_contexts = f.read()
    assert fq_contexts == 'b c\n' +\
                          'b d\n' +\
                          'c d\n'

    min_count = 9
    max_count = 30
    cont_file = contexts_for_freqrange(data_dir, distribution_file, min_count, max_count, filename_suffix='')
    with open(cont_file, 'r') as f:
        fq_contexts = f.read()
    assert fq_contexts == 'b c\n'


    min_count = 1
    max_count = 15
    cont_file = contexts_for_freqrange(data_dir, distribution_file, min_count, max_count, filename_suffix='')
    with open(cont_file, 'r') as f:
        fq_contexts = f.read()
    assert fq_contexts == 'a b\n'


    min_count = 1
    max_count = 5
    cont_file = contexts_for_freqrange(data_dir, distribution_file, min_count, max_count, filename_suffix='')
    with open(cont_file, 'r') as f:
        fq_contexts = f.read()
    assert fq_contexts == ''
