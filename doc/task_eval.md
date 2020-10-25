Module task_eval
================

Functions
---------

    
`compute_correlations(scores: (<class 'numpy.ndarray'>, <class 'list'>), name_pairs: List[Tuple[str, str]] = None, common_subset: bool = False)`
:   Computer correlation between score series.
    :param scores: Structured array of scores with embedding/ground_truth names.
    :param name_pairs: pairs of scores to correlate. If None, every pair will be computed.
                      if 'gt', everything will be plot against the ground_truth.

    
`compute_dists(vecs)`
:   

    
`compute_scores(actions, embeddings, scores, datasets, pairs, brain_scores=None, pre_score_files: str = None, ling_vecs_names=[], vecs_names=[], mm_lingvis=False, mm_embs_of: List[Tuple[str]] = None, mm_padding=False, common_subset=False)`
:   

    
`coverage(vocabulary, data)`
:   

    
`covered(dataset, vocab)`
:   

    
`dataset_vocab(dataset: str) ‑> list`
:   

    
`divide_eval_vocab_by_freqranges(distribution_file, eval_data_dir, dataset_name, num_groups=3, save=False)`
:   

    
`eval_concreteness(scores: numpy.ndarray, word_pairs, num=100, gt_divisor=10, vecs_names=None, tablefmt='simple')`
:   Eval dataset instances based on WordNet synsets.

    
`eval_dataset(dataset: List[Tuple[str, str, float]], dataset_name: str, embeddings: List[numpy.ndarray], vocabs: List[List[str]], labels: List[str]) ‑> (<class 'numpy.ndarray'>, <class 'list'>)`
:   

    
`highlight(val, conditions: dict, tablefmt)`
:   Highlight value in a table column.
    :param val: number, value
    :param conditions: dict of {colour: condition}
    :param tablefmt: 'simple' is terminal, 'latex-raw' is LaTeX

    
`latex_escape(string)`
:   

    
`main(datadir, embdir: str = None, vecs_names=[], savepath=None, loadpath=None, actions=['plotcorr'], plot_orders=['ground_truth'], plot_vecs=[], ling_vecs_names=[], pre_score_files: str = None, mm_embs_of: List[Tuple[str]] = None, mm_lingvis=False, mm_padding=False, print_corr_for=None, common_subset=False, tablefmt: str = 'simple', concrete_num=100, pair_score_agg='sum', quantity=-1)`
:   :param pair_score_agg:
    :param mm_lingvis:
    :param tablefmt: printed table format. 'simple' - terminal, 'latex_raw' - latex table.
    :param concrete_num:
    :param datadir: Path to directory which contains evaluation data (and embedding data if embdir is not given)
    :param vecs_names: List[str] Names of embeddings
    :param embdir: Path to directory which contains embedding files.
    :param savepath: Full path to the file to save scores without extension. None if there's no saving.
    :param loadpath: Full path to the files to load scores and brain results from without extension.
                     If None, they'll be computed.
    :param actions:
    :param plot_orders:
    :param plot_vecs:
    :param ling_vecs_names: List[str] Names of linguistic embeddings.
    :param pre_score_files: Previously saved score file path without extension, which the new scores will be merged with
    :param mm_embs_of: List of str tuples, where the tuples contain names of embeddings which are to
                       be concatenated into a multi-modal mid-fusion embedding.
    :param mm_padding:
    :param print_corr_for: 'gt' prints correlations scores for ground truth, 'all' prints scores between all
                            pairs of scores.
    :param common_subset: action printcorr: Print results for subests of the eval datasets which are covered by all
                          embeddings' vocabularies.
                          action compbarin: Compute brain scores for interection of vocabularies.

    
`mm_over_uni(name, score_dict)`
:   

    
`neighbors(words, vocab, vecs, n=10)`
:   

    
`plot_brain_words(brain_scores, plot_order)`
:   Plot hit counts for word in Brain data.
    :param brain_scores: brain score dict
    :param plot_order: 'concreteness' orders words for Wordnet conreteness
                       <emb_name> orders plot for an embedding's scores

    
`plot_by_concreteness(scores: numpy.ndarray, word_pairs, ax1, ax2, common_subset=False, vecs_names=None, concrete_num=100, title_prefix='', pair_score_agg='sum', show=False)`
:   Plot scores for data splits with increasing concreteness.

    
`plot_for_freqranges(scores: numpy.ndarray, gt_divisor, quantity=-1, common_subset=False, pair_num=None, split_num=None, ds_name=None)`
:   

    
`plot_for_quantities(scores: numpy.ndarray, gt_divisor, common_subset=False, legend=False, pair_num=None)`
:   

    
`plot_scores(scores: numpy.ndarray, gt_divisor=10, vecs_names=None, labels=None, colours=None, linestyles=None, title=None, type='plot', alphas=None, xtick_labels=None, ax=None, show=True, swapaxes=False)`
:   Scatter plot of a structured array.

    
`print_brain_scores(brain_scores, tablefmt: str = 'simple', caption='', suffix='', label='')`
:   

    
`print_correlations(scores: numpy.ndarray, name_pairs='gt', common_subset: bool = False, tablefmt: str = 'simple', caption='', label='')`
:   

    
`wn_concreteness(word, similarity_fn=<bound method WordNetCorpusReader.path_similarity of <WordNetCorpusReader in '/Users/anitavero/nltk_data/corpora/wordnet'>>)`
:   WordNet distance of a word from its root hypernym.

    
`wn_concreteness_for_pairs(word_pairs, synset_agg: str, similarity_fn=<bound method WordNetCorpusReader.path_similarity of <WordNetCorpusReader in '/Users/anitavero/nltk_data/corpora/wordnet'>>, pair_score_agg='sum') ‑> (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>)`
:   Sort scores by first and second word's concreteness scores.
    :param pair_score_agg: 'sum' adds scores for the two words, 'diff' computes their absolute difference.
    :return (ids, scores): sorted score indices and concreteness scores.

Classes
-------

`DataSets(datadir: str)`
:   Class for storing evaluation datasets and linguistic embeddings.

    ### Class variables

    `datasets`
    :

    `fmri_vocab`
    :

    `men: List[Tuple[str, str, float]]`
    :

    `normalizers`
    :

    `simlex: List[Tuple[str, str, float]]`
    :

`PlotColour()`
:   

    ### Static methods

    `colour_by_modality(labels)`
    :

    `get_legend()`
    :