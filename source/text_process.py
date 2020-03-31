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