from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from unidecode import unidecode
import string


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


def text2w2vf(text, window=5, vocab=[]):
    """Prepare contexts word2vecf using their context format:
        :return training_pairs:
                   textual file of word-context pairs.
                   each pair takes a separate line.
                   the format of a pair is "<word> <context>", i.e. space delimited, where <word> and <context> are strings.
                   The context is all non stop words in the same sentence or around the token if it's not sent_tokenized.
    """
    print("vocab:", len(vocab))
    training_pairs = ''
    if type(text[0]) == str:   # space separated tokens
        return extract_neighbours(text, vocab, window)
    elif type(text[0]) == list:    # list of str list format
        training_pairs = ''
        for sent in text:
            training_pairs += extract_neighbours(sent, vocab, window)
    return training_pairs


def extract_neighbours(tokens, vocab=[], window=5):
    training_pairs = ''
    positions = [(x, "l%s_" % x) for x in range(-window, +window + 1) if x != 0]
    for i, tok in enumerate(tokens):
        if vocab and tok not in vocab: continue
        for j, s in positions:
            if i + j < 0: continue
            if i + j >= len(tokens): continue
            c = tokens[i + j]
            if vocab and c not in vocab: continue
            training_pairs += f'{tok} {s}{c}\n'
    return training_pairs


def context_pairs(text, lang='english'):
    """Prepare contexts word2vecf without their context format:
        :return training_pairs:
                   textual file of word-context pairs.
                   each pair takes a separate line.
                   the format of a pair is "<word> <context>", i.e. space delimited, where <word> and <context> are strings.
                   The context is all non stop words in the same sentence.
    """
    training_pairs = ''
    if type(text) == str:   # raw text
        sents = text2gensim(text, lang)
    elif type(text) == list:    # Already in list of str list format
        sents = text
    for s in sents:
        for w in s:
            for c in s:
                if w != c:
                    training_pairs += f'{w} {c}\n'
    return training_pairs
