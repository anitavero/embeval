Example Usage
==============

Print performance scores on semantic similarity and relatedness tasks:

``python source.task_eval.py datadir compscores embdir --vecs-names model1 model2 --ling-vecs-names text_model1 text_model2 --mm_padding=True``

Run Mutual Information Experiments for word frequency ranges using HSIC kernel method:

``python source.run_mi_experiments --exp-names='quantity'  cost-name='BIHSIC_IChol'  --pca_n_components=100``


Fore more examples see:

:py:mod:`source.run_infogain_analysis`

:py:mod:`source.run_infogain_experiments`

:py:mod:`emb_information`

:py:mod:`unsupervised_metrics`