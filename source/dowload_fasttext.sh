
# Pre-trained word vectors learned on different sources are downloaded below:
#
# wiki-news-300d-1M.vec.zip: 1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).
# wiki-news-300d-1M-subword.vec.zip: 1 million word vectors trained with subword infomation on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).
# crawl-300d-2M.vec.zip: 2 million word vectors trained on Common Crawl (600B tokens).
# crawl-300d-2M-subword.zip: 2 million word vectors trained with subword information on Common Crawl (600B tokens).

DIR=$1

# -N will download and overwrite the file only if the server has a newer version.
wget -N "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip" -P $DIR
wget -N "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip" -P $DIR
wget -N "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip" -P $DIR
wget -N "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip" -P $DIR