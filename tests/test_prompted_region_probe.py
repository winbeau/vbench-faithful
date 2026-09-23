import pytest

from scripts.counterfactual.probe_prompted_regions import main
from scripts.counterfactual.audit_prompted_regions import selected_prediction_indices


def test_prompt_probe_invalid_selection_fails_before_source_access():
    args=['prepare','--manifest','absent','--review','absent','--native-run','absent',
          '--region-run','absent','--candidate','x','--output','absent',
          '--start','-1','--lag','1','--region','0']
    with pytest.raises(SystemExit) as e:main(args)
    assert e.value.code==2


def test_rgb_only_ablation_holds_sift_out_of_sam_prompting():
    identity={'request':{'prompts':[{'origin':s} for s in
        ('native_integer_peak_0','sift_key_10','unshifted_identity_proposal')]},
        'regions':[{'request_prompt_index':i} for i in [0,0,1,1,2,2]]}
    assert selected_prediction_indices(identity,'all')==list(range(6))
    assert selected_prediction_indices(identity,'rgb-only')==[0,1,4,5]
