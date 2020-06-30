import os
import argh
import subprocess
import time

import vecs2nps


def train(contexts_file, save_dir, w2v_dir, filename_suffix='', min_count=10, size=300, negative=15, threads=4):
    """Perform the stepst to train word2vecf on a given corpus:

        2. Create word and context vocabularies:

            ./myword2vec/count_and_filter -train dep.contexts -cvocab cv -wvocab wv -min-count 100

         This will count the words and contexts in dep.contexts, discard either words or contexts
         appearing < 100 times, and write the counted words to `wv` and the counted contexts to `cv`.

        3. Train the embeddings:

            ./myword2vec/word2vecf -train dep.contexts -wvocab wv -cvocab cv -output dim200vecs -size 200 -negative 15 -threads 10

         This will train 200-dim embeddings based on `dep.contexts`, `wv` and `cv` (lines in `dep.contexts` with word not in `wv` or context
         not in `cv` are ignored).

         The -dumpcv flag can be used in order to dump the trained context-vectors as well.

            ./myword2vec/word2vecf -train dep.contexts -wvocab wv -cvocab cv -output dim200vecs -size 200 -negative 15 -threads 10 -dumpcv dim200context-vecs

        4. convert the embeddings to numpy-readable format.
    """
    # # 1. Create input data, which is in the form of (word,context) pairs.
    # print('Create context pairs')
    # contexts_file = os.path.join(save_dir, f'context_pairs{filename_suffix}.txt')
    # text2w2vf(corpus, contexts_file, window, vocab, threads)

    start = time.time()

    # 2. Create word and context vocabularies
    print('Create vocabularies')
    cv = os.path.join(save_dir, f'cv_{filename_suffix}')
    wv = os.path.join(save_dir, f'wv_{filename_suffix}')
    output = subprocess.run(
        [f'{w2v_dir}/count_and_filter', '-train', contexts_file, '-cvocab', cv, '-wvocab', wv, '-min-count', str(min_count)],
        stdout=subprocess.PIPE)
    print(output.stdout.decode('utf-8'))
    with open(os.path.join(save_dir, f'trainlog{filename_suffix}.log'), 'w') as f:
        f.write(output.stdout.decode('utf-8'))

    # 3. Train the embeddings
    print('Train the embeddings')
    modelfn = os.path.join(save_dir, f'model{filename_suffix}')
    contextvecs = os.path.join(save_dir, f'context-vecs{filename_suffix}')
    output = subprocess.run(
        [f'{w2v_dir}/word2vecf', '-train', contexts_file, '-cvocab', cv, '-wvocab', wv,
         '-output', modelfn, '-size', str(size), '-negative', str(negative), '-threads', str(threads),
         '-dumpcv', contextvecs],
        stdout=subprocess.PIPE)
    print(output.stdout.decode('utf-8'))
    end = time.time()
    etime = end - start
    hours = etime // 3600 % 24
    minutes = etime // 60 % 60
    seconds = etime % 60
    print(f'Training time: {hours}h {minutes}m {seconds}s')
    with open(os.path.join(save_dir, f'trainlog{filename_suffix}.log'), 'a') as f:
        f.write('\nTrain:\n')
        f.write(output.stdout.decode('utf-8'))

    # 4. Convert the embeddings to numpy-readable format.
    print('Convert the embeddings to numpy-readable format')
    vecs2nps.main(modelfn, modelfn)


if __name__ == '__main__':
    argh.dispatch_command(train)
