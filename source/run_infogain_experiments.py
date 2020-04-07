import task_eval
import argh
from glob import glob


def main(exp_name):
    ######## Test #######
    # datadir = '/Users/anitavero/projects/data'
    # embdir = '/Users/anitavero/projects/data/wikidump/models/'
    # savedir = embdir
    #
    # if exp_name == 'quantity':
    #     task_eval.main(datadir, actions=['compscores'], embdir=embdir,
    #                    vecs_names=['model_n-1_0'],
    #                    ling_vecs_names=[],
    #                    mm_lingvis=False,
    #                    mm_padding=False,
    #                    savepath=savedir + exp_name)
    ######## END Test #######

    datadir = '/local/filespace/alv34/Datasets/'
    embdir = '/local/filespace/alv34/embeval/'
    savedir = embdir + '/results/infogain/'

    if exp_name == 'quantity':
        quantity_models = []

        task_eval.main(datadir, actions=['compscores'], embdir=embdir,
                       vecs_names=quantity_models,
                       ling_vecs_names=None,
                       mm_lingvis=True,
                       mm_padding=False,
                       savepath=savedir + exp_name)


if __name__ == '__main__':
    argh.dispatch_command(main)