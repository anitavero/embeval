import task_eval
import argh
from argh import arg


@arg('-a', '--actions', nargs='+', choices=['printcorr', 'plot_quantity', 'plot_freqrange'])
def main(actions='printcorr', name='', tablefmt='simple'):
    datadir = '/Users/anitavero/projects/data'
    embdir = '/Users/anitavero/projects/data/wikidump/models/'
    savedir = embdir

    if 'printcorr' in actions:

        task_eval.main(datadir, actions=['printcorr'],
                       loadpath=embdir + name,
                       print_corr_for='gt',
                       tablefmt=tablefmt)

    if 'plot_quantity' in actions:
        task_eval.main(datadir, actions=['plot_quantity'],
                       loadpath=embdir + name,
                       print_corr_for='gt',
                       tablefmt=tablefmt)

    if 'plot_freqrange' in actions:
        task_eval.main(datadir, actions=['plot_freqrange'],
                       loadpath=embdir + name,
                       print_corr_for='gt',
                       tablefmt=tablefmt,
                       quantity=500)


if __name__ == '__main__':
    argh.dispatch_command(main)