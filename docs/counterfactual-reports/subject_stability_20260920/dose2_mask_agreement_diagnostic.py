"""Read-only model-mask agreement diagnostic; never changes score qualification."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read_rows(path): return [json.loads(line) for line in path.read_text().splitlines()]
def artifact(root, descriptor):
    path=(root/descriptor['path']).resolve()
    if root.resolve() not in path.parents or sha(path)!=descriptor['sha256']:
        raise ValueError('artifact identity changed')
    return path

def scoring_mask(root, variant, method):
    path=artifact(root,variant['scoring_masks'][method])
    with np.load(path,allow_pickle=False) as data:
        array=data['masks']; present=data['present']; metadata=json.loads(data['metadata_json'].item())
    if (array.ndim!=4 or not np.isin(array,(0,1)).all()
        or not np.array_equal(array.reshape(*array.shape[:2],-1).any(-1),present)
        or metadata['decoded_rgb_sha256']!=variant['decoded_rgb_sha256']
        or metadata['source']!='actual_variant_only' or metadata['method']!=method):
        raise ValueError('scoring masks do not match actual variant')
    return array.any(1)

def ratio(a,b): return float(a/b) if b else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','dataset','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--position',default='start');p.add_argument('--method',default='hybrid')
    args=p.parse_args();run=json.loads((args.run/'run.json').read_text());protocol=json.loads((args.run/'protocol.json').read_text())
    if (not run.get('completed') or sha(args.run/'scores.jsonl')!=run['scores_sha256']
        or sha(args.dataset/'index.jsonl')!=run['dataset_index_sha256']):raise ValueError('incomplete or changed inputs')
    records=read_rows(args.run/'scores.jsonl');index={r['video_uid']:r for r in read_rows(args.dataset/'index.jsonl')}
    if len(records)!=len(index) or {r['base']['video_uid'] for r in records}!=set(index):raise ValueError('population mismatch')
    source=Path(run.get('source_root',args.run));cases=[]
    for row in records:
        if row['construction_status']!='accepted':continue
        uid=row['base']['video_uid'];case={'video_uid':uid,'subject_en':row['base']['subject_en'],
            'review_eligible':uid not in protocol.get('input_review_exclusions',{}),'status':'missing_scoring_artifacts'}
        cases.append(case);keys=['clean',args.position+'/background_corrupt']
        if any(k not in row['variants'] or args.method not in row['variants'][k].get('scoring_masks',{}) for k in keys):continue
        descriptor=index[uid];mp=args.dataset/descriptor['manifest']
        if sha(mp)!=descriptor['manifest_sha256']:raise ValueError('dataset manifest changed')
        manifest=json.loads(mp.read_text());root=source/row['source_shard'] if 'source_shard' in row else args.run
        clean,bg=[row['variants'][k] for k in keys]
        a,b=[scoring_mask(root,v,args.method) for v in [clean,bg]]
        with np.load(artifact(args.dataset,manifest['construction_masks']),allow_pickle=False) as data:construction=data['masks'].astype(bool)
        if a.shape!=b.shape or a.shape!=construction.shape:raise ValueError('frame or resolution mismatch')
        indices=manifest['positions'][args.position]['parameters']['window_indices'];edited=np.zeros(len(a),bool);edited[indices]=True
        inter=(a&b).sum((1,2));union=(a|b).sum((1,2));asize=a.sum((1,2));protected=(a&construction).sum((1,2))
        case.update(status='diagnosed',frames=len(a),edited_frames=len(indices),
            clean_scoring_mask_outside_protected_region_fraction=ratio((asize-protected)[edited].sum(),asize[edited].sum()),
            scoring_mask_iou_edited=ratio(inter[edited].sum(),union[edited].sum()),
            scoring_mask_iou_unedited=ratio(inter[~edited].sum(),union[~edited].sum()),
            unedited_frames_with_scoring_mask_change=int(np.any(a!=b,axis=(1,2))[~edited].sum()),
            construction_mean_area=float(construction.mean()),clean_scoring_mean_area=float(a.mean()))
        for method,label in [('official','origin'),(args.method+'_exclude','repair')]:
            x,y=[v['scores'].get(method,{}) for v in [clean,bg]]
            case[label+'_abs_change']=abs(x['score']-y['score']) if x.get('status')==y.get('status')=='succeeded' else None
    result={'candidate_denominator':len(records),'constructed_denominator':len(cases),'position':args.position,
        'diagnosed':sum(r['status']=='diagnosed' for r in cases),'cases':cases,
        'interpretation':'Agreement between two predicted construction/scoring masks is not semantic ground truth. Outside-protection fraction is geometric potential exposure, not a count of actually changed pixels. No cohort, score, or model is modified.',
        'provenance':{'scores_sha256':run['scores_sha256'],'dataset_index_sha256':run['dataset_index_sha256'],'diagnostic_source_sha256':sha(Path(__file__))}}
    for population in ['full_numeric','primary']:
        rows=[r for r in cases if r['status']=='diagnosed' and (population=='full_numeric' or r['review_eligible'])]
        summaries=[]
        for lower,upper in [(0,.2),(.2,.5),(.5,1.000001)]:
            group=[r for r in rows if r['clean_scoring_mask_outside_protected_region_fraction'] is not None and lower<=r['clean_scoring_mask_outside_protected_region_fraction']<upper]
            values=[r['repair_abs_change'] for r in group if r.get('repair_abs_change') is not None]
            summaries.append({'outside_protected_region_range':[lower,upper],'cases':len(group),'repair_scored':len(values),'repair_mean_abs_change':float(np.mean(values)) if values else None})
        result[population]={'diagnosed':len(rows),'exposure_bins':summaries,
            'cases_with_unedited_frame_mask_changes':sum(r['unedited_frames_with_scoring_mask_change']>0 for r in rows)}
    if args.output.exists():raise ValueError('refusing to overwrite diagnostic')
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
