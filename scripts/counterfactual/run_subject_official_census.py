"""Run the registered official Subject census without choosing inputs by score."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import ROOT, sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json


def bind_scoring(plan, dataset):
    """Only bind artifact identities/counts; scoring policy comes from the plan."""
    rows = read_jsonl(dataset / 'index.jsonl')
    accepted = {r['video_uid'] for r in rows if r['status'] == 'accepted'}
    config = copy.deepcopy(plan['scoring_template'])
    if len(rows) != plan['videos'] or len({r['video_uid'] for r in rows}) != len(rows):
        raise ValueError('construction does not cover the registered candidate population')
    known = config['input_review_exclusions']
    config['input_review_exclusions'] = {uid: why for uid, why in known.items() if uid in accepted}
    config.update(dataset_index_sha256=sha256_file(dataset / 'index.jsonl'),
                  construction_protocol_sha256=sha256_file(dataset / 'protocol.json'),
                  expected_constructed=len(accepted),
                  expected_review_qualified=len(accepted)-len(config['input_review_exclusions']))
    return config


def media_inventory(base, video_root):
    path = artifact_path(video_root, base['relative_video_path'])
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {'video_uid': base['video_uid'], 'path': str(path),
              'bytes': path.stat().st_size, 'sha256': sha256_file(path),
              'container_duration_seconds': None}
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=codec_type,nb_frames,width,height,duration', '-of', 'json', str(path)],
        capture_output=True, text=True, timeout=30)
    if probe.returncode:
        result['duration_status'] = 'ffprobe_failed'
        result['ffprobe_error'] = probe.stderr.strip()[:1000]
    else:
        info = json.loads(probe.stdout); result['streams'] = info.get('streams', [])
        raw = info.get('format', {}).get('duration')
        if raw not in (None, 'N/A'):
            result['container_duration_seconds'] = float(raw)
        result['duration_status'] = 'known' if result['container_duration_seconds'] is not None else 'unavailable'
    return result


def verify_dataset(dataset, workers):
    from .build_region_discrimination import verify
    before = time.time(); proof = verify(dataset, workers=workers)
    proof.update(index_sha256=sha256_file(dataset/'index.jsonl'), wall_seconds=time.time()-before,
                 workers=workers, verifier_source_sha256=sha256_file(ROOT/'scripts/counterfactual/build_region_discrimination.py'))
    write_json(dataset/'integrity.json', proof)
    print(json.dumps(proof), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'video-root', 'segformer-model', 'detector-checkpoint',
                 'mobilesam-checkpoint', 'dino-repo', 'dino-weight', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--source-archive-sha256', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or sha256_file(args.plan) != args.expected_plan_sha256:
        raise ValueError('invalid workers or changed preregistration')
    plan = json.loads(args.plan.read_text())
    manifest = artifact_path(ROOT, plan['scoring_template']['input_manifest'])
    class_map = artifact_path(ROOT, plan['class_map'])
    corruption = artifact_path(ROOT, plan['corruption_protocol'])
    for path, expected in [(manifest, plan['manifest_sha256']), (class_map, plan['class_map_sha256']),
                           (corruption, plan['corruption_protocol_sha256'])]:
        if sha256_file(path) != expected:
            raise ValueError('registered construction input changed')
    # A fixed census writes many lossless variants; leave existing work intact.
    if shutil.disk_usage(args.output.parent).free < 120 * 1024**3:
        raise RuntimeError('less than 120 GiB available for the full census')
    out = new_output(args.output); started = time.time()
    base_env = {**os.environ, 'VBENCH_AUDIT_WORKSPACE': str(ROOT),
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
    events = []

    def stage(name, **details):
        event = {'stage': name, 'unix': time.time(), **details}; events.append(event)
        write_json(out/'stage.json', event); write_json(out/'events.json', events)
        print(json.dumps(event, ensure_ascii=False), flush=True)

    def run_module(module, arguments, logfile):
        with logfile.open('w') as log:
            subprocess.run([sys.executable, '-m', module, *map(str, arguments)], cwd=ROOT,
                env=base_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)

    def score_family(name, dataset, config, verifier=None):
        control = new_output(out/name); protocol = control/'scoring_protocol.json'; write_json(protocol, config)
        processes = []; stage(name+'_scoring', protocol_sha256=sha256_file(protocol))
        for shard, gpu in enumerate(config['scoring_gpus']):
            command = [sys.executable, '-m', 'scripts.counterfactual.run_subject_stability',
                '--dataset', str(dataset), '--output', str(control/f'shard{shard}'), '--protocol', str(protocol),
                '--detector-checkpoint', str(args.detector_checkpoint), '--mobilesam-checkpoint', str(args.mobilesam_checkpoint),
                '--dino-repo', str(args.dino_repo), '--dino-weight', str(args.dino_weight),
                '--device', 'cuda:0', '--num-shards', str(len(config['scoring_gpus'])), '--shard-index', str(shard)]
            env = {**base_env, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
            with (control/f'shard{shard}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
            (control/f'shard{shard}.pid').write_text(str(process.pid)+'\n'); processes.append(process)
        codes = [p.wait() for p in processes]
        if verifier is not None:
            process, log_path = verifier
            if process.wait() != 0:
                raise RuntimeError('independent pixel verification failed; see '+str(log_path))
        if any(codes):
            raise RuntimeError(f'scoring failed with exit codes {codes}; all outputs preserved')
        merged, analysis = out/(name+'-merged'), out/(name+'-analysis')
        run_module('scripts.counterfactual.merge_subject_stability', ['--source', control, '--output', merged], out/(name+'-merge.log'))
        run_module('scripts.counterfactual.analyze_subject_stability', ['--run', merged, '--output', analysis,
            '--protocol', protocol, '--integrity-receipt', dataset/'integrity.json'], out/(name+'-analysis.log'))
        from .analyze_subject_stability import summarize, review_qualified_records
        records = read_jsonl(merged/'scores.jsonl')
        qualified = review_qualified_records(records, config['input_review_exclusions'])
        groups = {}
        for group in sorted({r['base']['intervention_prompt_group'] for r in records}):
            selected = [r for r in qualified if r['base']['intervention_prompt_group'] == group]
            groups[group], _ = summarize(selected, ['hybrid_exclude'], positions=config['positions'])
        write_json(analysis/'prompt_group_statistics.json', groups)
        stage(name+'_completed', run_sha256=sha256_file(merged/'run.json'))

    run = {'started_unix': started, 'plan_sha256': sha256_file(args.plan),
           'source_archive_sha256': args.source_archive_sha256,
           'controller_sha256': sha256_file(Path(__file__)), 'source_root': str(ROOT),
           'output': str(out), 'scoring_gpus': plan['scoring_template']['scoring_gpus'],
           'workers': args.workers, 'completed': False}
    write_json(out/'run.json', run); (out/'registered_plan.json').write_bytes(args.plan.read_bytes())
    try:
        stage('media_inventory')
        rows = read_jsonl(manifest)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            media = list(pool.map(lambda row: media_inventory(row, args.video_root), rows))
        known = [r['container_duration_seconds'] for r in media if r['container_duration_seconds'] is not None]
        write_json(out/'media_inventory.json', {'videos': len(media), 'known_durations': len(known),
            'unknown_durations': len(media)-len(known), 'known_duration_sum_seconds': sum(known), 'media': media})
        stage('segmentation')
        construction = out/'construction'
        run_module('scripts.counterfactual.generate_subject_masks_parallel', ['--bases', manifest,
            '--protocol', args.plan, '--class-map', class_map, '--model-dir', args.segformer_model,
            '--video-root', args.video_root, '--output', construction, '--workers', args.workers], out/'construction.log')
        from .build_region_discrimination import build
        stage('quarter_build'); dataset = out/'quarter-dataset'; before = time.time()
        summary = build(construction, new_output(dataset), workers=args.workers, protocol_path=corruption,
                        gaussian_reference_short_side=plan['gaussian_reference_short_side'])
        write_json(dataset/'build_run.json', {'started_unix': before, 'finished_unix': time.time(),
            'wall_seconds': time.time()-before, 'summary': summary, 'workers': args.workers,
            'construction_run_sha256': sha256_file(construction/'run.json'),
            'source_sha256': sha256_file(ROOT/'scripts/counterfactual/build_region_discrimination.py')})
        config = bind_scoring(plan, dataset)
        log_path = out/'quarter-verification.log'
        with log_path.open('w') as log:
            verifier = subprocess.Popen([sys.executable, '-c',
                'import sys; from pathlib import Path; from scripts.counterfactual.run_subject_official_census import verify_dataset; verify_dataset(Path(sys.argv[1]), int(sys.argv[2]))',
                str(dataset), str(args.workers)], cwd=ROOT, env=base_env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        score_family('quarter', dataset, config, (verifier, log_path))
        from .build_subject_frame_probe import build as build_probe, verify as verify_probe, POSITIONS
        stage('single_frame_build'); probe_dataset = out/'single-frame-dataset'
        proof = json.loads((dataset/'integrity.json').read_text())
        probe_plan = {'schema_version': 8, 'family': 'subject_single_frame_probe',
            'preregistration_sha256': sha256_file(args.plan), 'positions': list(POSITIONS),
            'cohort_candidates': plan['videos'], 'expected_constructed': config['expected_constructed'],
            'parent_dataset': str(dataset), 'parent_index_sha256': sha256_file(dataset/'index.jsonl'),
            'parent_integrity_sha256': sha256_file(dataset/'integrity.json'),
            'parent_verified_frames': proof['verified_corrupted_frames'],
            'declaration': plan['supplemental_probe'], 'claim_limit': plan['claim_limit']}
        probe_path = out/'single-frame-construction-protocol.json'; write_json(probe_path, probe_plan)
        before = time.time(); build_probe(dataset, new_output(probe_dataset), probe_path)
        write_json(probe_dataset/'build_run.json', {'started_unix': before, 'finished_unix': time.time(),
            'wall_seconds': time.time()-before,
            'source_sha256': sha256_file(ROOT/'scripts/counterfactual/build_subject_frame_probe.py')})
        before = time.time(); integrity = verify_probe(dataset, probe_dataset)
        write_json(probe_dataset/'integrity.json', {**integrity, 'wall_seconds': time.time()-before})
        probe_config = bind_scoring(plan, probe_dataset)
        probe_config.update(positions=list(POSITIONS), primary_intervention='first_frame/background_corrupt',
            controls=[p+'/'+s for p in POSITIONS for s in ['background_corrupt','subject_corrupt']
                      if (p,s) != ('first_frame','background_corrupt')])
        score_family('single-frame', probe_dataset, probe_config)
        stage('completed')
        run.update(completed=True, finished_unix=time.time(), wall_seconds=time.time()-started)
        write_json(out/'run.json', run)
    except Exception as exc:
        stage('failed', reason=f'{type(exc).__name__}: {exc}')
        run.update(failure_reason=f'{type(exc).__name__}: {exc}', finished_unix=time.time(), wall_seconds=time.time()-started)
        write_json(out/'run.json', run)
        raise


if __name__ == '__main__':
    main()
