DIR=$1

# -N will download and overwrite the file only if the server has a newer version.
wget -N "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip" -P $DIR
wget -N "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip" -P $DIR
wget -N "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip" -P $DIR
wget -N "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip" -P $DIR