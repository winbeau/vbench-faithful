#!/usr/bin/env python3
"""Render two native-resolution 4x4 sheets of the exact sampled before/after frames.

Run in the pinned official decoding environment. Sheets are for independent vision
annotation only; they are never fed back into a scoring model. No boxes, predictions
or expected visibility labels are drawn on them.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vbench_prompts_compile import annotation_jobs as J
from cache_matrix_transforms import frames_for_backend, frame_hash
from cache_backend_outputs import sha


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--vbench-root',default='/root/wenbiao_zhao/VBench')
    args=p.parse_args(argv)
    sys.path.insert(0,args.vbench_root)
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from vbench.utils import load_video
    job=json.loads(Path(args.job).read_text());base=job['base'];transformed=job['transformed']
    source=Path(base['video_path'])
    if sha(source)!=base['video_sha256']:
        raise ValueError('source changed')
    frames=frames_for_backend(source,load_video)
    if len(frames)!=16 or [frame_hash(f) for f in frames]!=base['frame_sha256']:
        raise ValueError('source frame mismatch')
    after=[]
    for i,(frame,detail) in enumerate(zip(frames,transformed['frame_transform'])):
        if detail['status']!='ok' or detail.get('target_box_coverage')!=1:
            raise ValueError('endpoint geometry incomplete')
        patched=frame.copy()
        for x0,y0,x1,y1 in detail['rectangles']:
            patched[y0:y1,x0:x1]=128
        if frame_hash(patched)!=transformed['frame_sha256'][i]:
            raise ValueError('reconstructed patch differs from scored evidence')
        path=Path(detail['endpoint_image'])
        if sha(path)!=detail['endpoint_image_sha256']:
            raise ValueError('endpoint PNG changed')
        if not np.array_equal(np.asarray(Image.open(path).convert('RGB')),patched.astype('uint8')):
            raise ValueError('endpoint preview differs from reconstructed evidence')
        after.append(patched)
    if len(after)!=16:
        raise ValueError('missing endpoint frame')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    height,width=frames[0].shape[:2];header=28
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    except OSError:font=ImageFont.load_default()
    paths=[]
    for name,values in [('before',frames),('after',after)]:
        sheet=Image.new('RGB',(4*width,4*(height+header)),(255,255,255));draw=ImageDraw.Draw(sheet)
        for i,frame in enumerate(values):
            x=(i%4)*width;y=(i//4)*(height+header)
            draw.text((x+6,y+2),f'Frame {i}',fill=(0,0,0),font=font)
            sheet.paste(Image.fromarray(frame.astype('uint8')),(x,y+header))
        path=out/(name+'.png');sheet.save(path);paths.append(str(path))
    J.atomic_json(out/'contacts.json',{'images':paths,'image_sha256':[sha(p) for p in paths],
        'source_video_sha256':base['video_sha256'],'sampled_frame_count':16,
        'native_frame_size':[width,height],'display_note':'integer RGB rendering of scored frames; no spatial resizing or detector hints'})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
