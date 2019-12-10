import eval

if __name__ == '__main__':

    # eval.main('/Users/anitavero/projects/data', actions=['concreteness'],
    #           loadpath='/Users/anitavero/projects/data/mmdeed/padding')

    eval.main('/Users/anitavero/projects/data', actions=['printcorr'],
              loadpath='/Users/anitavero/projects/data/mmdeed/full_nopadding_fmri',
              print_corr_for='gt')
