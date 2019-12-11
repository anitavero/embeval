import eval


if __name__ == '__main__':

    datadir = '/local/filespace/alv34/Datasets/'
    embdir = '/local/filespace/alv34/mmdeed/'
    savedir = embdir + '/results/'

    eval.main(datadir, actions=['compscores', 'compbrain'], embdir=embdir,
              vecs_names = ['google_alexnetfc7', 'vecs3lem', 'google_vgg', 'fmri_google_resnet-18',
                            'fmri_google_alexnet', 'fmri-internal_m5_mm_descriptors',
                            'frcnn_internal_mm_descriptors'],
              ling_vecs_names = ['wikinews', 'wikinews_sub', 'crawl', 'w2v13'],
              mm_lingvis=True,
              mm_padding=True,
              savepath = savedir + 'padding')

