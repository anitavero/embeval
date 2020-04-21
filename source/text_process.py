import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from unidecode import unidecode
import string
from tqdm import tqdm
from glob import glob
import json
from multiprocessing import Process
import math


hun_stopwords = stopwords.words('hungarian') + \
                ['is', 'ha', 'szerintem', 'szoval', 'na', 'hat', 'kicsit', 'ugye', 'amugy']
stopwords_lang = {'hungarian': hun_stopwords, 'english': stopwords.words('english'),
                  'hunglish': hun_stopwords + stopwords.words('english') + [unidecode(w) for w in hun_stopwords]}


def tokenize(text, lang):
    """
    Lower, tokenize, filter punctuation and stopwords.
    :param text: str
    :param lang: {hungarian|english|hunglish}
    :return: str list iterator
    """
    # TODO: stemming
    text = text.lower()
    trtab = text.maketrans(string.punctuation, ''.join([' ' for i in range(len(string.punctuation))]))
    words = text.translate(trtab).split()
    words = filter(lambda w: w not in stopwords_lang[lang], words)
    return words


def text2gensim(text, lang):
    """Tokenize and filter stop words. Return list of str lists (sdt for gensim)
        where each str list is a sentence and each text is a list of these lists."""
    sents = sent_tokenize(text)
    return iter([list(tokenize(s, lang)) for s in sents])


def hapax_legomena(text):
    """Return words that occur only once within a text.
    :param text: str list or Counter
    """
    cnt = Counter(text) if type(text) == list else text
    return [w for w, c in cnt.most_common() if c == 1]


def text2w2vf(corpus_tup, data_dir, window=5, vocab=[], processes=1):
    """Prepare contexts word2vecf using their context format:
       textual file of word-context pairs.
       each pair takes a separate line.
       the format of a pair is "<word> <context>", i.e. space delimited, where <word> and <context> are strings.
       The context is all non stop words in the same sentence or around the token if it's not sent_tokenized.
       :param corpus_tup: list with elements of: token (str) list or sentence list (list of str lists)
       :param data_dir: directory to write context pairs to
       :param window: Window for w2v. If 0 and text is a sentence list the context of all words are all the other
                      words in the same sentence.
    """
    print("vocab:", len(vocab))

    def contexts(corp_tup):
        for fn, txt in tqdm(corp_tup):
            cont_file = os.path.splitext(fn)[0] + '.contexts'
            if window > 0:
                if type(txt[0]) == str:   # space separated tokens
                    extract_neighbours(txt, cont_file, vocab, window)
                elif type(txt[0]) == list:    # list of str list format
                    for sent in txt:
                        extract_neighbours(sent, cont_file, vocab, window)
            elif type(txt[0]) == list:
                context_pairs(txt, cont_file, lang='english')
            else:
                print('Sentence context works only with list of str lists input.')

    if processes > 1:
        # Multiprocessing
        def chunks(lst, n):
            """Yield successive n chunks from lst."""
            size = math.ceil(len(lst) / n)
            for i in range(0, len(lst), size):
                yield lst[i:i + size]

        file_chunks = chunks(corpus_tup, processes)
        queue = []
        for c_tups in file_chunks:
            p = Process(target=contexts, args=(c_tups, ))
            p.start()
            queue.append(p)
        for p in queue:
            p.join()
    else:
        contexts(corpus_tup)

    # concatenate files
    print('Concatenate files')
    with open(os.path.join(data_dir, 'contexts.txt'), 'w') as cf:
        files = glob(os.path.join(data_dir, '*/*.contexts'))
        for fn in tqdm(files):
            with open(fn, 'r') as f:
                p_pairs = f.read()
                if p_pairs[-1] != '\n':
                    p_pairs += '\n'
                cf.write(p_pairs)


def extract_neighbours(tokens, contexts_file, vocab=[], window=5):
    positions = [(x, "l%s_" % x) for x in range(-window, +window + 1) if x != 0]
    with open(contexts_file, 'w') as f:
        for i, tok in enumerate(tokens):
            if vocab and tok not in vocab: continue
            for j, s in positions:
                if i + j < 0: continue
                if i + j >= len(tokens): continue
                c = tokens[i + j]
                if vocab and c not in vocab: continue
                f.write(f'{tok} {s}{c}\n')


def context_pairs(text, contexts_file, lang='english'):
    """Prepare contexts word2vecf without their context format:
       textual file of word-context pairs.
       each pair takes a separate line.
       the format of a pair is "<word> <context>", i.e. space delimited, where <word> and <context> are strings.
       The context is all non stop words in the same sentence.
    """
    if type(text) == str:   # raw text
        sents = text2gensim(text, lang)
    elif type(text) == list:    # Already in list of str list format
        sents = text
    with open(contexts_file, 'w') as f:
        for s in sents:
            for w in s:
                for c in s:
                    if w != c:
                        f.write(f'{w} {c}\n')
