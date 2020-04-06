import task_eval
import argh


def main(exp_name):
    ######## Test #######
    # datadir = '/Users/anitavero/projects/data'
    # embdir = '/Users/anitavero/projects/data/mmdeed/'
    # savedir=embdir
    #
    # if exp_name == 'nopadding_mitchell':
    #     task_eval.main(datadir, actions=['compbrain'], embdir=embdir,
    #                    vecs_names=['fmri-internal_m5_mm_descriptors'],
    #                    ling_vecs_names=['w2v13'],
    #                    mm_lingvis=True,
    #                    mm_padding=False,
    #                    savepath=savedir + 'test_mitchell')
    ######## END Test #######

    datadir = '/local/filespace/alv34/Datasets/'
    embdir = '/local/filespace/alv34/embeval/'
    savedir = embdir + '/results/'

    if exp_name == 'padding':
        task_eval.main(datadir, actions=['compscores', 'compbrain'], embdir=embdir,
                       vecs_names=['google_alexnetfc7', 'vecs3lem1', 'google_vgg',
                                   'men-internal', 'men-whole', 'google_resnet152'],
                       ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                       mm_lingvis=True,
                       mm_padding=True,
                       savepath=savedir + exp_name)

    if exp_name == 'nopadding':
        task_eval.main(datadir, actions=['compscores', 'compbrain'], embdir=embdir,
                       vecs_names=['google_alexnetfc7', 'vecs3lem1', 'google_vgg',
                                   'men-internal', 'men-whole', 'google_resnet152'],
                       ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                       mm_lingvis=True,
                       mm_padding=False,
                       savepath=savedir + exp_name)

    # if exp_name == 'nopadding_common_subset':
    #     task_eval.main(datadir, actions=['compscores', 'compbrain'], embdir=embdir,
    #                    vecs_names=['google_alexnetfc7', 'vecs3lem1', 'google_vgg',
    #                                'men-internal', 'men-whole', 'google_resnet152'],
    #                    ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
    #                    mm_lingvis=True,
    #                    mm_padding=False,
    #                    savepath=savedir + exp_name)


    ######## Brain experiments ########

    if exp_name == 'padding_mitchell':
        task_eval.main(datadir, actions=['compbrain'], embdir=embdir,
                       vecs_names=['fmri_google_resnet152', 'fmri_google_alexnet',
                                   'fmri-internal_m5_vs_descriptors', 'fmri-internal_m5_mm_descriptors',
                                   'fmri_combined_vs_descriptors', 'fmri_combined_mm_descriptors',
                                   'frcnn_whole_vs_descriptors', 'frcnn_whole_mm_descriptors',
                                   'vecs3lem1'],
                       ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                       mm_lingvis=True,
                       mm_padding=True,
                       savepath=savedir + exp_name)

    if exp_name == 'nopadding_mitchell':
        task_eval.main(datadir, actions=['compbrain'], embdir=embdir,
                       vecs_names=['fmri_google_resnet152', 'fmri_google_alexnet',
                                   'fmri-internal_m5_vs_descriptors', 'fmri-internal_m5_mm_descriptors',
                                   'fmri_combined_vs_descriptors', 'fmri_combined_mm_descriptors',
                                   'frcnn_whole_vs_descriptors', 'frcnn_whole_mm_descriptors',
                                   'vecs3lem1'],
                       ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                       mm_lingvis=True,
                       mm_padding=False,
                       savepath=savedir + exp_name)

    ##### Common subset #####

    if exp_name == 'padding_mitchell_commonsubset':
        task_eval.main(datadir, actions=['compbrain'], embdir=embdir,
                       vecs_names=['fmri_google_resnet152', 'fmri_google_alexnet',
                                   'fmri-internal_m5_vs_descriptors', 'fmri-internal_m5_mm_descriptors',
                                   'fmri_combined_vs_descriptors', 'fmri_combined_mm_descriptors',
                                   'frcnn_whole_vs_descriptors', 'frcnn_whole_mm_descriptors',
                                   'vecs3lem1'],
                       ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                       mm_lingvis=True,
                       mm_padding=True,
                       savepath=savedir + exp_name,
                       common_subset=True)

    if exp_name == 'nopadding_mitchell_commonsubset':
        task_eval.main(datadir, actions=['compbrain'], embdir=embdir,
                       vecs_names=['fmri_google_resnet152', 'fmri_google_alexnet',
                                   'fmri-internal_m5_vs_descriptors', 'fmri-internal_m5_mm_descriptors',
                                   'fmri_combined_vs_descriptors', 'fmri_combined_mm_descriptors',
                                   'frcnn_whole_vs_descriptors', 'frcnn_whole_mm_descriptors',
                                   'vecs3lem1'],
                       ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                       mm_lingvis=True,
                       mm_padding=False,
                       savepath=savedir + exp_name,
                       common_subset=True)


if __name__ == '__main__':
    argh.dispatch_command(main)
