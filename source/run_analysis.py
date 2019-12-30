import eval
import argh


def main(action):
    datadir = '/Users/anitavero/projects/data'
    embdir = '/Users/anitavero/projects/data/mmdeed/'

    if action == 'printcorr':

        # print('\nSEMSIM\n')

        # print('\nPadding\n')
        eval.main(datadir, actions=['printcorr'],
                  loadpath=embdir + 'padding',
                  print_corr_for='gt',
                  tablefmt='latex_raw')

        # print('\nNo Padding\n')
        eval.main(datadir, actions=['printcorr'],
                  loadpath=embdir + 'nopadding',
                  print_corr_for='gt',
                  tablefmt='latex_raw')

        # print('\nPadding - common subset\n')
        eval.main(datadir, actions=['printcorr'],
                  loadpath=embdir + 'nopadding',
                  print_corr_for='gt',
                  tablefmt='latex_raw',
                  common_subset=True)

        # print('\nBRAIN\n')

        # print('\nNo padding Brain\n')
        eval.main(datadir, actions=['printbraincorr'],
                  loadpath=embdir + 'nopadding_mitchell',
                  print_corr_for='gt',
                  tablefmt='latex_raw')

        # print('\nNo padding Brain - common subset\n')
        eval.main(datadir, actions=['printbraincorr'],
                  loadpath=embdir + 'nopadding_mitchell_commonsubset',
                  print_corr_for='gt',
                  tablefmt='latex_raw')

        # ###### Test ######
        # eval.main(datadir, actions=['brainwords'],
        #           loadpath=embdir + 'test_mitchell',
        #           print_corr_for='gt',
        #           tablefmt='simple')

    if action == 'concreteness':

        with open('figs/figs.tex', 'w') as f:
            f.write('')

        for pair_agg in ['sum', 'diff']:
            print(f'\n------------ {pair_agg} ------------')

            print('\nPadding')
            eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'padding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg)

            print('\nPadding - common subset')
            eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'padding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg,
                      common_subset=True)


            print('\nNo Padding')
            eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'nopadding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg)

            print('\nNo Padding - common subset')
            eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'nopadding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg,
                      common_subset=True)

    if action == 'brainwords':

        with open('figs/figs_brain.tex', 'w') as f:
            f.write('')

        print('\nNo Padding Brain Words')
        eval.main(datadir, actions=['brainwords'],
                  loadpath=embdir + 'nopadding_mitchell')

        print('\nNo Padding Brain Words - common subset')
        eval.main(datadir, actions=['brainwords'],
                  loadpath=embdir + 'nopadding_mitchell_commonsubset')


if __name__ == '__main__':
    argh.dispatch_command(main)