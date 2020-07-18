import task_eval
import argh


def main(action, name='', tablefmt='simple'):
    datadir = '/Users/anitavero/projects/data'
    embdir = '/Users/anitavero/projects/data/wikidump/models/'
    savedir = embdir

    if action == 'printcorr':

        task_eval.main(datadir, actions=['printcorr'],
                       loadpath=embdir + name,
                       print_corr_for='gt',
                       tablefmt=tablefmt)

    if action == 'plot_quantity':
        task_eval.main(datadir, actions=['plot_quantity'],
                       loadpath=embdir + name,
                       print_corr_for='gt',
                       tablefmt=tablefmt)


if __name__ == '__main__':
    argh.dispatch_command(main)