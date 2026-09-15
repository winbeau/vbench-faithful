import importlib.util
from pathlib import Path
path = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_pairwise_statistics.py'
spec = importlib.util.spec_from_file_location('pairwise_statistics', path)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
EXPECTED={'dynamic_degree':(.603,.684,0,.684,.461,1,.815),'subject_consistency':(.626,.584,.000302,.585,.383,1,.961),'human_action':(.663,.553,0,.553,.265,1,.904),'spatial_relationship':(.531,.505,.344890,.525,.292,.7778,.726)}
def test_reconstructed_official_baseline():
 # Only these four dimensions have frozen results under results/e0/raw_official_scores.
 # DIMS also covers motion_smoothness, whose official scores live in the supplementary
 # output root and are selected with --official-scores-root instead.
 rows=m.calculate(iterations=10,dimensions=list(EXPECTED))
 assert {x['dimension'] for x in rows}==set(EXPECTED)
 for x in rows:
  d,t,delta,tie,tau,cov,pear=EXPECTED[x['dimension']]
  for got,want in ((x['dev_acc0'],d),(x['test_acc0'],t),(x['tie_margin_delta'],delta),(x['test_tie_aware_acc'],tie),(x['kendall_tau_b'],tau),(x['coverage'],cov),(x['model_level_pearson_n4'],pear)):assert abs(got-want)<.001,(x['dimension'],got,want)
