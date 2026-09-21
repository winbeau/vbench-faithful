"""Construction-only instance probe; never a scoring-mask provider."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from .common import ROOT, sha256_file
from .generate_subject_masks import deterministic_setup
from .region_discrimination import corrupt_image
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json, write_jsonl, write_npz


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--construction',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--all-frames',action='store_true',help='Inspect every frame rather than only first/middle/last.')
    args=parser.parse_args()
    import torch,torchvision
    from torchvision.models import resnet50
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.ops.misc import FrozenBatchNorm2d
    deterministic_setup();torch.set_num_threads(2)
    config=json.loads((ROOT/'configs/subject-repair/instance_construction_probe.json').read_text())
    digest=sha256_file(args.checkpoint)
    if digest!=config['weight_sha256']:raise ValueError('unexpected model checkpoint')
    if torchvision.__version__.split('+')[0]!='0.20.1':raise ValueError('probe pins torchvision 0.20.1')
    # Match torchvision's COCO_V1 construction, including FrozenBN and eps=0,
    # while loading only the explicitly provisioned local checkpoint.
    backbone=_resnet_fpn_extractor(resnet50(weights=None,norm_layer=FrozenBatchNorm2d),3)
    model=MaskRCNN(backbone,num_classes=91,min_size=config['resize_short_side'],
                  max_size=config['resize_max_side']).eval()
    model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True),strict=True)
    for module in model.modules():
        if isinstance(module,FrozenBatchNorm2d):module.eps=0.0
    output=new_output(args.output);rows=[];sections=[];sheets=[]
    for entry in read_jsonl(args.construction/'index.jsonl'):
        manifest=json.loads(artifact_path(args.construction,entry['manifest']).read_text())
        if manifest['status']!='accepted':continue
        frames=read_png_sequence(args.construction,manifest['frames'])
        with np.load(artifact_path(args.construction,manifest['mask_file']['path']),allow_pickle=False) as p:
            old_masks=p['masks']
        uid=manifest['base']['video_uid'];figures=[]
        indices=range(len(frames)) if args.all_frames else sorted({0,len(frames)//2,len(frames)-1})
        for index in indices:
            image=frames[index]
            with torch.inference_mode():prediction=model([torch.from_numpy(image).permute(2,0,1).float()/255])[0]
            take=(prediction['labels']==config['person_category']) & (prediction['scores']>=config['detection_threshold'])
            masks=(prediction['masks'][take,0]>=config['mask_threshold']).numpy().astype(np.uint8)
            scores=prediction['scores'][take].tolist();boxes=prediction['boxes'][take].tolist()
            record={'video_uid':uid,'frame_index':index,'num_person_instances':len(masks),'detection_scores':scores,
                    'boxes':boxes,'status':'single_person_candidate' if len(masks)==1 else 'ambiguous_or_missing',
                    'role':'construction_probe','human_confirmed':False,'metric_scores_used':False}
            write_npz(output/'masks'/f'{uid}_{index:06d}.npz',masks=masks,
                      metadata_json=json.dumps({'role':'construction_probe','family':'maskrcnn_resnet50',
                                               'weights_sha256':digest,'config':config},sort_keys=True))
            old=corrupt_image(image,1-old_masks[index],operator='gaussian',scale=18)
            new=image.copy()
            if len(masks)==1:
                new=corrupt_image(image,1-masks[0],operator='gaussian',scale=18)
                record['subject_area_fraction']=float(masks[0].mean())
                record['subject_pixels_changed']=int(np.count_nonzero(np.any(new!=image,axis=-1)&(masks[0]>0)))
            else:
                for box in boxes:
                    x0,y0,x1,y1=map(round,box);cv2.rectangle(new,(x0,y0),(x1,y1),(255,0,0),2)
            tiles=[]
            for title,frame in [('Original',image),('SegFormer + GrabCut',old),('Instance candidate' if len(masks)==1 else f'Ambiguous: {len(masks)} people',new)]:
                ok,data=cv2.imencode('.png',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR));assert ok
                uri=base64.b64encode(data).decode()
                tiles.append(f'<figure><img src="data:image/png;base64,{uri}"><figcaption>{title}</figcaption></figure>')
            figures.append(f'<h3>Frame {index}</h3><div class="row">'+''.join(tiles)+'</div>')
            if index==0:
                canvas=np.full((326,900,3),250,np.uint8)
                cv2.putText(canvas,manifest['base']['prompt_en'][:90],(8,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1)
                for j,frame in enumerate((image,old,new)):canvas[26:,j*300:(j+1)*300]=cv2.resize(frame,(300,300))
                sheets.append(canvas)
            rows.append(record);write_jsonl(output/'observations.jsonl',rows)
            print(uid,index,record['status'],flush=True)
        sections.append('<section><h2>'+html.escape(manifest['base']['prompt_en'])+'</h2>'+''.join(figures)+'</section>')
    document='<!doctype html><meta charset="utf-8"><title>Construction instance probe</title><style>body{font:16px system-ui;margin:2rem}.row{display:flex;gap:1rem}figure{margin:0}img{width:320px;max-width:30vw}section{border-top:1px solid #ccc}</style><h1>Construction instance probe</h1><p>Exploratory construction comparison only. No scoring masks or metric results were used. Multiple people are flagged rather than merged. Native subject pixels are preserved by each candidate edit.</p>'+''.join(sections)
    (output/'instance_review.html').write_text(document)
    cv2.imwrite(str(output/'instance_preview.png'),cv2.cvtColor(np.concatenate(sheets),cv2.COLOR_RGB2BGR))
    write_json(output/'run.json',{'torch':torch.__version__,'torchvision':torchvision.__version__,'device':'cpu',
        'checkpoint_sha256':digest,'config':config,'frames_probed':len(rows),'metric_scoring':'NOT RUN',
        'frame_sampling':'all' if args.all_frames else 'first_middle_last',
        'source_sha256':sha256_file(Path(__file__))})
    return 0


if __name__=='__main__':raise SystemExit(main())
