import os
import task_eval
import argh
from glob import glob


def main(exp_name, filter_pattern='', pre_score_files=None, subdir=''):
  #  ######## Test #######
  #  datadir = '/Users/anitavero/projects/data'
  #  embdir = '/Users/anitavero/projects/data/wikidump/models/'
  #  savedir = embdir
  #  ######## END Test #######

    datadir = '/local/filespace/alv34/Datasets/'
    embdir = '/anfs/bigdisc/alv34/wikidump/extracted/models/' + subdir
    savedir = embdir + '/results/'

    models = glob(embdir + f'*model*{filter_pattern}*npy*')
    models = [os.path.split(m)[1].split('.')[0] for m in models]

    task_eval.main(datadir, actions=['compscores'], embdir=embdir,
                   vecs_names=['vecs3lem1', 'google_resnet152'],
                   ling_vecs_names=models,
                   mm_lingvis=True,
                   mm_padding=False,
                   savepath=savedir + exp_name,
                   pre_score_files=pre_score_files)


if __name__ == '__main__':
    argh.dispatch_command(main)
