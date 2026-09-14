import importlib.util
from pathlib import Path
p=Path('/root/vbench-audit/scripts/evaluate_pairwise_statistics.py');s=importlib.util.spec_from_file_location('stats',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
EXPECTED={'dynamic_degree':(.603,.684,0,.684,.461,1,.815),'subject_consistency':(.626,.584,.000302,.585,.383,1,.961),'human_action':(.663,.553,0,.553,.265,1,.904),'spatial_relationship':(.531,.505,.344890,.525,.292,.7778,.726)}
def test_reconstructed_official_baseline():
 for x in m.calculate(iterations=10):
  d,t,delta,tie,tau,cov,pear=EXPECTED[x['dimension']]
  for got,want in ((x['dev_acc0'],d),(x['test_acc0'],t),(x['tie_margin_delta'],delta),(x['test_tie_aware_acc'],tie),(x['kendall_tau_b'],tau),(x['coverage'],cov),(x['model_level_pearson_n4'],pear)):assert abs(got-want)<.001,(x['dimension'],got,want)
