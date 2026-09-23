import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.counterfactual.validate_vjepa_motion import select_metadata, cluster_interval, paired_stats, human_statistics

CONFIG=json.loads((Path(__file__).parents[1]/'configs/dynamic-static-jitter/vjepa-validation-v1.json').read_text())


def test_prespecified_native_population_and_prompt_separation():
    pool=[]; reserved=[]
    for p in range(30):
        for g in range(3):
            for rep in range(5):
                row={'video_uid':f'{p}-{g}-{rep}','prompt_id':f'p{p}','split':'test',
                     'dimension':'dynamics_degree','relative_video_path':f'{p}-{g}-{rep}.mp4'}
                pool.append(row)
                if rep==0: reserved.append(row)
        reserved.append({'video_uid':f'{p}-gif','prompt_id':f'p{p}','split':'test',
                         'dimension':'dynamics_degree','relative_video_path':f'{p}.gif'})
    native,natural,excluded=select_metadata(pool,reserved,[{'prompt_id':'train'}],[{'prompt_id':'dev'}])
    assert (len(native),len(natural),len(excluded))==(90,450,30)
    assert all(r['status']=='NOT SCORED' for r in excluded)
    assert select_metadata(list(reversed(pool)),list(reversed(reserved)),[],[])==(native,natural,excluded)
    with pytest.raises(ValueError,match='leakage'):
        select_metadata(pool,reserved,[{'prompt_id':'p0'}],[])
    with pytest.raises(ValueError,match='population'):
        select_metadata(pool[:-1],reserved,[],[])


def test_source_equal_weight_signed_delta_not_positive_part():
    c=copy.deepcopy(CONFIG); c['analysis']['bootstrap_replicates']=100
    pairs=[{'prompt_id':'a','origin':[0,1,1],'joint':[.2,.3,.3],
            'natural_only':[.2,.4,.4],'joint_latent':[-1,-.9,-.9]},
           {'prompt_id':'b','origin':[0,0,0],'joint':[.8,.6,.6],
            'natural_only':[.8,.8,.8],'joint_latent':[1,.8,.8]}]
    result=paired_stats(pairs,c)
    assert result['joint']['delta']==pytest.approx(-.05)
    assert result['joint']['mae']==pytest.approx(.15)
    assert result['fixed_scale_numerical_check']['point_estimate_pass']
    assert not result['fixed_scale_numerical_check']['overall_goal_complete']
    pairs[0]['origin']=[1,1,1]
    assert paired_stats(pairs,c)['fixed_scale_numerical_check']['point_estimate_pass'] is None


def test_cluster_bootstrap_weights_source_counts():
    assert cluster_interval([1,1,1],['p','p','q'],4,100)==[1.,1.]
    a=cluster_interval([0,1,1],['p','q','q'],4,1000)
    assert a==[0.,1.]


def test_human_preference_direction_ties_are_separate():
    c=copy.deepcopy(CONFIG); c['analysis']['bootstrap_replicates']=100
    pairs=[dict(a='a',b='b',label=1,prompt_id='p'),dict(a='a',b='b',label=0,prompt_id='q'),
           dict(a='a',b='b',label=.5,prompt_id='z')]
    scores={'a':{'origin':0,'joint':.8,'natural_only':.2},'b':{'origin':0,'joint':.2,'natural_only':.8}}
    result=human_statistics(pairs,scores,c)
    assert result['origin']['predicted_ties']==2
    assert result['origin']['concordance']==.5
    assert result['joint']['strict_correct']==1 and result['joint']['incorrect']==1
    assert result['joint']['human_ties']==1
    assert result['joint']['human_tie_mean_absolute_gap']==pytest.approx(.6)
    assert human_statistics([],scores,c)['joint']['concordance'] is None
