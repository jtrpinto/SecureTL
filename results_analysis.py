'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: results_analysis.py
- Reads the results pickle file saved by [trait]_test_triplet_model.py or
  [trait]_test_secure_model.py and prints several performance and security
  metrics, also plotting cancelability, linkability, ROC, and DET curves.

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import numpy as np
import pickle as pk
import aux_functions as af


RESULTS_FILE = "model_results.pk"   # results file saved by [trait]_test_triplet_model.py or [trait]_test_secure_model.py
SECURE = True                       # True (model trained with SecureTL) or False (model trained with original Triplet Loss)

results = pk.load(open(RESULTS_FILE, 'rb'))

# FMR, FNMR, EER, Cancelability, and Non-Linkability:
if SECURE:
    af.plot_perf_vs_canc_curves(results, title='My Method', figsize=[7.0, 4.0], savefile='performance_curve.pdf')
    af.plot_dsys(results, title='My Method', figsize=[7.0, 4.0], savefile='dsys_curve.pdf')
else:
    af.plot_perf_curves(results[0], title='My Method', figsize=[7.0, 4.0], savefile='performance_curve.pdf')

# ROC and DET Curves:
af.plot_roc([results[0]['roc']], ['My Method'], 'ROC Curves', figsize=[7.0, 4.0], savefile='roc_curves.pdf')  # May plot for more than one method
af.plot_det([results[0]['roc']], ['My Method'], 'DET Curves', figsize=[7.0, 4.0], savefile='det_curves.pdf')  # May plot for more than one method
