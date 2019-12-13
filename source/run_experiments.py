import eval
import argh


def main(exp_name):
    datadir='/local/filespace/alv34/Datasets/'
    embdir='/local/filespace/alv34/mmdeed/'
    savedir=embdir + '/results/'

    if exp_name == 'padding':
        eval.main(datadir, actions=['compscores', 'compbrain'], embdir=embdir,
                  vecs_names=['google_alexnetfc7', 'vecs3lem1', 'google_vgg',
                              'men-internal', 'men-whole'],
                  ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                  mm_lingvis=True,
                  mm_padding=True,
                  savepath=savedir + exp_name)

    if exp_name == 'nopadding':
        eval.main(datadir, actions=['compscores', 'compbrain'], embdir=embdir,
                  vecs_names=['google_alexnetfc7', 'vecs3lem1', 'google_vgg',
                              'men-internal', 'men-whole'],
                  ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                  mm_lingvis=True,
                  mm_padding=False,
                  savepath=savedir + exp_name)

    if exp_name == 'nopadding_common_subset':
        eval.main(datadir, actions=['compscores', 'compbrain'], embdir=embdir,
                  vecs_names=['google_alexnetfc7', 'vecs3lem1', 'google_vgg',
                              'men-internal', 'men-whole'],
                  ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                  mm_lingvis=True,
                  mm_padding=False,
                  savepath=savedir + exp_name)

    if exp_name == 'padding_mitchell':
        eval.main(datadir, actions=['compbrain'], embdir=embdir,
                  vecs_names=['fmri_google_resnet-18', 'fmri_google_alexnet',
                              'fmri-internal_m5_vs_descriptors', 'fmri-internal_m5_mm_descriptors',
                              'fmri_combined_vs_descriptors', 'fmri_combined_mm_descriptors',
                              'frcnn_whole_vs_descriptors', 'frcnn_whole_mm_descriptors',
                              'vecs3lem1'],
                  ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                  mm_lingvis=True,
                  mm_padding=True,
                  savepath=savedir + exp_name)

    if exp_name == 'nopadding_mitchell':
        eval.main(datadir, actions=['compbrain'], embdir=embdir,
                  vecs_names=['fmri_google_resnet-18', 'fmri_google_alexnet',
                              'fmri-internal_m5_vs_descriptors', 'fmri-internal_m5_mm_descriptors',
                              'fmri_combined_vs_descriptors', 'fmri_combined_mm_descriptors',
                              'frcnn_whole_vs_descriptors', 'frcnn_whole_mm_descriptors',
                              'vecs3lem1'],
                  ling_vecs_names=['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
                  mm_lingvis=True,
                  mm_padding=False,
                  savepath=savedir + exp_name)


if __name__ == '__main__':
    argh.dispatch_command(main)
