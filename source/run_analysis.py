import task_eval
import argh


def main(action):
    datadir = '/Users/anitavero/projects/data'
    embdir = '/Users/anitavero/projects/data/mmdeed/'

    if action == 'printcorr' or action == 'printcorr_subsample':

        # print('\nSEMSIM\n')

        # print('\nPadding\n')
        task_eval.main(datadir, actions=[action],
                  loadpath=embdir + 'padding',
                  print_corr_for='gt',
                  tablefmt='latex_raw')

        # print('\nNo Padding\n')
        task_eval.main(datadir, actions=[action],
                  loadpath=embdir + 'nopadding',
                  print_corr_for='gt',
                  tablefmt='latex_raw')

        # print('\nPadding - common subset\n')
        task_eval.main(datadir, actions=[action],
                  loadpath=embdir + 'nopadding',
                  print_corr_for='gt',
                  tablefmt='latex_raw',
                  common_subset=True)

        # print('\nBRAIN\n')

        if action != 'printcorr_subsample':
            # print('\nNo padding Brain\n')
            task_eval.main(datadir, actions=['printbraincorr'],
                      loadpath=embdir + 'nopadding_mitchell',
                      print_corr_for='gt',
                      tablefmt='latex_raw')

            # print('\nNo padding Brain - common subset\n')
            task_eval.main(datadir, actions=['printbraincorr'],
                      loadpath=embdir + 'nopadding_mitchell_commonsubset',
                      print_corr_for='gt',
                      tablefmt='latex_raw')

        # ###### Test ######
        # eval.main(datadir, actions=['brainwords'],
        #           loadpath=embdir + 'test_mitchell',
        #           print_corr_for='gt',
        #           tablefmt='simple')

    if action == 'concreteness':

        with open('figs/figs_concreteness.tex', 'w') as f:
            f.write('')

        for pair_agg in ['sum', 'diff']:
            print(f'\n------------ {pair_agg} ------------')

            print('\nPadding')
            task_eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'padding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg)

            print('\nPadding - common subset')
            task_eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'padding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg,
                      common_subset=True)


            print('\nNo Padding')
            task_eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'nopadding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg)

            print('\nNo Padding - common subset')
            task_eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'nopadding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg,
                      common_subset=True)

    if action == 'brainwords':

        with open('figs/figs_brain.tex', 'w') as f:
            f.write('')

        print('\nNo Padding Brain Words')
        task_eval.main(datadir, actions=['brainwords'],
                  loadpath=embdir + 'nopadding_mitchell')

        print('\nNo Padding Brain Words - common subset')
        task_eval.main(datadir, actions=['brainwords'],
                  loadpath=embdir + 'nopadding_mitchell_commonsubset')


if __name__ == '__main__':
    argh.dispatch_command(main)