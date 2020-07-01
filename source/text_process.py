import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from unidecode import unidecode
import string
from tqdm import tqdm
from glob import glob
import argh
from multiprocessing import Process
import math
from itertools import chain


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


def text2w2vf(corpus_tup, data_dir, window=5, vocab=[], processes=1, merge=False, filename_suffix=''):
    """Prepare contexts word2vecf using their context format:
       textual file of word-context pairs.
       each pair takes a separate line.
       the format of a pair is "<word> <context>", i.e. space delimited, where <word> and <context> are strings.
       The context is all non stop words in the same sentence or around the token if it's not sent_tokenized.
       :param corpus_tup: list with elements of: token (str) list or sentence list (list of str lists)
       :param data_dir: directory to write context pairs to
       :param window: Window for w2v. If 0 and text is a sentence list the context of all words are all the other
                      words in the same sentence.
       :param vocab: list of str, vocab to filter with in extract_neighbours.
    """
    print("#vocab to filter with:", len(vocab))

    def contexts(corp_tup):
        for fn, txt in tqdm(corp_tup):
            cont_file = os.path.splitext(fn)[0] + f'{filename_suffix}_window-{window}.contexts'
            if window > 0:
                if type(txt[0]) == str:   # space separated tokens
                    extract_neighbours(txt, cont_file, vocab, window)
                elif type(txt[0]) == list:    # list of str list format
                    # for sent in txt:
                    tokens = list(chain.from_iterable(txt))
                    extract_neighbours(tokens, cont_file, vocab, window)
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

    if merge:
        concatenate_files(data_dir, f'{filename_suffix}.contexts',
                          f'{filename_suffix + "_" if filename_suffix else ""}window-{window}_contexts.txt')


def concatenate_files(data_dir, file_pattern, outfile):
    """ Concatenate files into one big file. """
    whole_file = os.path.join(data_dir, outfile)
    files = glob(os.path.join(data_dir, f'*/*{file_pattern}'))
    for fn in tqdm(files, desc=f'Concatenating {file_pattern} files'):
        with open(fn) as f:
            pairs = f.read()
            if pairs and pairs[-1] != '\n':
                pairs += '\n'
        if os.path.exists(whole_file):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        with open(whole_file, append_write) as f:
            f.write(pairs)


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

# TODO: filter by a given vocab here too
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


if __name__== '__main__':
    argh.dispatch_commands([concatenate_files])
