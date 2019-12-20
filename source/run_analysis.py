import eval
import argh


def main(action):
    datadir = '/Users/anitavero/projects/data'
    embdir = '/Users/anitavero/projects/data/mmdeed/'

    if action == 'printcorr':

        print('\n############## Padding ##############')
        eval.main(datadir, actions=['printcorr'],
                  loadpath=embdir + 'padding',
                  print_corr_for='gt',
                  tablefmt='latex_raw')

        print('\n############## No padding ##############')
        eval.main(datadir, actions=['printcorr'],
                  loadpath=embdir + 'nopadding',
                  print_corr_for='gt',
                  tablefmt='latex_raw')

        print('\n############## No padding - common subset ##############')
        eval.main(datadir, actions=['printcorr'],
                  loadpath=embdir + 'nopadding',
                  print_corr_for='gt',
                  tablefmt='latex_raw',
                  common_subset=True)

        print('\n############## No padding Brain ##############')
        eval.main(datadir, actions=['printcorr'],
                  loadpath=embdir + 'nopadding_mitchell',
                  print_corr_for='gt',
                  tablefmt='latex_raw')

        print('\n############## No padding Brain - common subset ##############')
        eval.main(datadir, actions=['printcorr'],
                  loadpath=embdir + 'nopadding_mitchell_commonsubset',
                  print_corr_for='gt',
                  tablefmt='latex_raw')

        # ###### Test ######
        # eval.main(datadir, actions=['brainwords'],
        #           loadpath=embdir + 'test_mitchell',
        #           print_corr_for='gt',
        #           tablefmt='simple')

    if action == 'concreteness':

        for pair_agg in ['sum', 'diff']:
            print(f'\n\n------------ {pair_agg} ------------\n\n')

            print('\n############## Padding ##############')
            eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'padding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg)

            print('\n############## Padding - common subset ##############')
            eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'padding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg,
                      common_subset=True)


            print('\n############## No Padding ##############')
            eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'nopadding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg)

            print('\n############## No Padding - common subset ##############')
            eval.main(datadir, actions=['concreteness'],
                      loadpath=embdir + 'nopadding',
                      concrete_num=100,
                      plot_vecs=[],
                      pair_score_agg=pair_agg,
                      common_subset=True)

    if action == 'brainwords':

        print('\n############## No Padding Brain Words ##############')
        eval.main(datadir, actions=['brainwords'],
                  loadpath=embdir + 'nopadding_mitchell')

        print('\n############## No Padding Brain Words - common subset ##############')
        eval.main(datadir, actions=['brainwords'],
                  loadpath=embdir + 'nopadding_mitchell_commonsubset')


if __name__ == '__main__':
    argh.dispatch_command(main)