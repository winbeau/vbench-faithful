#!/usr/bin/env python3
"""Stage and publish only the frozen LASIESTA/BMC numeric evidence, never media."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

PREFIX = 'dimensions/dynamic_degree/generalization/external-v1-20260923'
DATASET = 'xju-arlab/vbench-repair'
MODEL = 'xju-arlab/vbench-model'
MARKER = '## Dynamic external generalization: LASIESTA and BMC (2026-09-23)'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def stage(args):
    output = args.stage.resolve(); output.mkdir(parents=True, exist_ok=False)
    research = args.research_root.resolve()
    code_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    origins = {}

    def copy(source, relative):
        target = output / relative
        if not source.is_file() or source.is_symlink() or target.exists():
            raise ValueError(f'Missing, linked or duplicate input: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        origins[str(relative)] = sha(source)

    for cohort, selected, controllers in [
        ('lasiesta-v1', 'selection', ['prepare-build-controller', 'score-summarize-controller']),
        ('bmc-v1', 'selection-v5', ['prepare-v5-controller', 'build-v5-controller', 'score-summarize-v5-controller'])]:
        base = Path('output/dynamic-generalization') / cohort
        patterns = ['analysis/*.json', 'analysis/*.jsonl', f'{selected}/*.json', f'{selected}/*.jsonl',
                    'scores/shard-*/*.json', 'scores/shard-*/scores.jsonl',
                    'construction/shard-*/inputs.jsonl', 'construction/shard-*/completion.json',
                    'download/*.json'] + [f'{c}/*.json' for c in controllers]
        for pattern in patterns:
            for source in sorted((research / base).glob(pattern)):
                copy(source, base / source.relative_to(research / base))
        if cohort == 'lasiesta-v1':
            for source in sorted((args.extra / cohort).glob('construction/shard-*/*')):
                relative = base / source.relative_to(args.extra / cohort)
                if (output / relative).exists():
                    if sha(output / relative) != sha(source):
                        raise ValueError('Conflicting recovered evidence')
                else:
                    copy(source, relative)
        for name in (cohort + '.json', 'review.' + cohort + '.json'):
            relative = Path('configs/dynamic-generalization') / name
            copy(research / relative, relative)
        relative = Path('docs/counterfactual-reports') / ('dynamic_generalization_' + cohort[:-3] + '_20260923.md')
        copy(research / relative, relative)
    # Keep evidence byte-exact; only Markdown navigation is adapted for publication.
    for path in output.glob('docs/counterfactual-reports/*.md'):
        source_relative = path.relative_to(output)
        def link(match):
            target = match.group(2)
            if re.match(r'^[a-z]+://', target) or target.startswith('#'):
                return match.group(0)
            filename, _, anchor = target.partition('#')
            resolved = (path.parent / filename).resolve()
            if output in resolved.parents and resolved.exists():
                return match.group(0)
            repo_path = os.path.normpath(str(source_relative.parent / filename))
            suffix = '#' + anchor if anchor else ''
            return f'[{match.group(1)}](https://github.com/winbeau/vbench-repair/blob/{code_revision}/{repo_path}{suffix})'
        path.write_text(re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, path.read_text()))
    summary = read(output / 'output/dynamic-generalization/bmc-v1/analysis/summary.json')
    if summary['source_clips'] != 84:
        raise ValueError('Unexpected cohort; create a new version instead')
    (output / 'README.md').write_text(
        '# Dynamic Degree: frozen external generalization evidence\n\n'
        'LASIESTA: 43 clips from 9 recordings; BMC: 84 clips from 7 eligible recordings '
        '(9 candidates, 2 NOT SCORED). These are 127 clips, **not 127 independent videos/scenes**. '
        'The datasets are reported separately. Frozen aligned-v1; no retraining or score remapping.\n\n'
        '| Cohort / group | Clips | Origin base → 8px jitter | Repair base → 8px jitter |\n'
        '| --- | ---: | ---: | ---: |\n'
        '| LASIESTA static | 16 | 0.000000 → 0.875000 | 0.125877 → 0.163079 |\n'
        '| LASIESTA moving | 27 | 0.592593 → 0.981481 | 0.589715 → 0.610019 |\n'
        '| BMC all, primary | 84 | 0.261905 → 1.000000 | 0.282374 → 0.292002 |\n'
        '| BMC visually static, exploratory | 5 | 0.000000 → 1.000000 | 0.041793 → 0.014800 |\n'
        '| BMC moving, exploratory | 59 | 0.372881 → 1.000000 | 0.362253 → 0.369636 |\n'
        '| BMC ambiguous, retained | 20 | 0.000000 → 1.000000 | 0.106876 → 0.132283 |\n\n'
        'Each CF cell first averages two fixed seeds per clip; encoding controls are separate. '
        'The intervention is local 8px alternating texture displacement, not RGB noise or brightness flicker.\n\n'
        'LASIESTA uses official object-state annotations but its native BMP frame timing is unverified. '
        'BMC labels are score-blind agent visual review, **not official motion/static truth or independent human labels**; '
        'its 5 static clips come from only 2 recordings. BMC retains 20 ambiguous clips in its primary analysis. '
        'BMC has 37/168 CFs with absolute repair changes >0.1; LASIESTA has 5/32 static CF increases >0.1. '
        'Average robustness is not per-clip invariance. CI clustering uses original recordings, not clips.\n\n'
        'Full reports: [LASIESTA](docs/counterfactual-reports/dynamic_generalization_lasiesta_20260923.md), '
        '[BMC](docs/counterfactual-reports/dynamic_generalization_bmc_20260923.md). '
        'Machine-authoritative summaries and individual scores are under `output/dynamic-generalization/`; '
        '`configs/` preserves original protocols/review records. `manifest.json` fixes every member SHA-256 '
        'and the private code revision. Historical absolute machine paths are provenance, not download paths.\n\n'
        '**Numeric evidence and documentation only:** no source/CF videos, screenshots, weights or private code. '
        'Original media must be obtained from the authors: '
        '[LASIESTA](https://www.gti.ssr.upm.es/data/lasiesta_database.html) '
        '([Cuevas et al., CVIU 2016](https://doi.org/10.1016/j.cviu.2016.08.005), CC BY-SA 4.0) and '
        '[BMC](https://backgroundmodelschallenge.eu/) '
        '([Vacavant et al., BMC/ACCV 2012](https://doi.org/10.1007/978-3-642-37410-4_25), '
        'standardized redistribution license not verified). No new license is assigned to third-party data.\n\n'
        f'Private [code/runbook](https://github.com/winbeau/vbench-repair/blob/{code_revision}/docs/reproduction/DYNAMIC_GENERALIZATION.md). '
        'Run `scripts/verify_dynamic_generalization.py --bundle <this-directory>` to recheck hashes, '
        '508 Origin decisions, 508 repair sigmoids, 127 encoding controls, pair means/MAE, cluster CIs and AUROC. '
        'This replay does not decode media or rerun GPU inference. Original full-media audit receipts are retained. '
        'Previously failed dev gates remain; physical strength calibration and pretraining overlap are not established.\n')
    files = []
    for path in sorted(output.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix not in ('.json', '.jsonl', '.md') or path.is_symlink():
            raise ValueError('Non-metadata file in publication stage')
        relative = path.relative_to(output).as_posix()
        files.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path),
                      'source_sha256': origins.get(relative),
                      'transformation': 'markdown_links_only' if relative.startswith('docs/') else 'none' if relative in origins else 'generated_index'})
    manifest = {'schema': 'vbench-repair-dynamic-external/1', 'repo_id': DATASET, 'path': PREFIX,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'code_repository': 'https://github.com/winbeau/vbench-repair',
        'code_revision': code_revision, 'research_base_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=research, text=True).strip(),
        'research_worktree_contains_uncommitted_experiments': True,
        'model_revision': '5fe53c4c7eb1a8a4fcd5b6e22f748da1d073a78e',
        'head_sha256': '6dcfbdce2e604cdd669e54015fee34e034c8004726d06c46611532f451ca8a53',
        'source_media_included': False, 'model_weights_changed': False, 'training_updates': 0,
        'files': files}
    write(output / 'manifest.json', manifest)
    print(json.dumps({'stage': str(output), 'files': len(files) + 1, 'bytes': sum(f['bytes'] for f in files)}))


def publish(args):
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    manifest = read(args.stage / 'manifest.json')
    # This lightweight integrity check deliberately avoids importing the GPU environment.
    for item in manifest['files']:
        if sha(args.stage / item['path']) != item['sha256']:
            raise ValueError('Stage changed after verification')
    replay = read(args.replay)
    if (replay['status'] != 'PASS' or replay['manifest_sha256'] != sha(args.stage / 'manifest.json')
            or {c['cohort'] for c in replay['cohorts']} != {'lasiesta-v1', 'bmc-v1'}):
        raise ValueError('A successful local replay is required')
    api = HfApi(endpoint=args.endpoint)
    initial = api.dataset_info(DATASET)
    if any(s.rfilename.startswith(PREFIX + '/') for s in initial.siblings):
        raise ValueError('Release prefix already exists; do not overwrite historical publication')
    declared = {entry['path'] for entry in manifest['files']} | {'manifest.json'}
    actual = {p.relative_to(args.stage).as_posix() for p in args.stage.rglob('*') if p.is_file()}
    if actual != declared:
        raise ValueError('Undeclared files in stage; refusing unreviewed upload')
    additions = {PREFIX + '/' + name: (args.stage / name).read_bytes() for name in declared}
    for card in ('README.md', 'dimensions/dynamic_degree/README.md'):
        original = Path(hf_hub_download(DATASET, card, repo_type='dataset', revision=initial.sha, endpoint=args.endpoint)).read_text()
        if MARKER in original:
            raise ValueError('External release already linked')
        link = f'https://huggingface.co/datasets/{DATASET}/blob/main/{PREFIX}/README.md'
        additions[card] = (original.rstrip() + '\n\n' + MARKER + '\n\n'
            f'[Full reports, protocols and byte-exact numeric evidence]({link}) add 43 LASIESTA and 84 BMC clips '
            'from 16 source recordings. The aligned-v1 weights and the 450-source paper experiment are unchanged. '
            'LASIESTA static/moving: Origin 0.000000→0.875000 / 0.592593→0.981481; '
            'Repair 0.125877→0.163079 / 0.589715→0.610019. BMC all: Origin 0.261905→1.000000; '
            'Repair 0.282374→0.292002. BMC labels are agent review, not official motion labels; '
            'native LASIESTA timing is unverified. These are exploratory mean-robustness results, '
            'not per-clip invariance or 127 independent scenes. Only numeric evidence is released; '
            'no original/derived external media are redistributed. Earlier archive counts above remain unchanged.\n').encode()
    commit = api.create_commit(repo_id=DATASET, repo_type='dataset', parent_commit=initial.sha,
        operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=b) for p, b in sorted(additions.items())],
        commit_message='feat(dynamic): publish frozen LASIESTA and BMC generalization evidence')
    receipt = {'status': 'uploaded_pending_download_verification', 'repo_id': DATASET,
        'dataset_revision': commit.oid, 'dataset_parent_revision': initial.sha, 'path': PREFIX,
        'endpoint': args.endpoint, 'code_revision': manifest['code_revision'], 'model_weights_changed': False,
        'replay': replay, 'dataset_files': [{'path': p, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                                         for p, b in sorted(additions.items())]}
    write(args.receipt, receipt)
    print(json.dumps({'dataset_revision': commit.oid, 'status': receipt['status']}), flush=True)
    complete(args, receipt)


def complete(args, receipt):
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi(endpoint=args.endpoint)
    verify_download(api, args, receipt)
    model = api.model_info(MODEL)
    original = Path(hf_hub_download(MODEL, 'README.md', revision=model.sha, endpoint=args.endpoint)).read_text()
    if MARKER in original:
        raise ValueError('Model card already updated; inspect before retry')
    link = f'https://huggingface.co/datasets/{DATASET}/blob/{receipt["dataset_revision"]}/{PREFIX}/README.md'
    updated = (original.rstrip() + '\n\n' + MARKER + '\n\n'
        f'The unchanged aligned-v1 checkpoint was evaluated on two external real-video datasets; '
        f'[frozen results and limitations]({link}). LASIESTA contributes 43 clips/9 recordings and BMC '
        '84 clips/7 recordings. No retraining or posthoc score mapping. BMC mean repair score '
        '0.282374→0.292002 versus Origin 0.261905→1.000000 under 8px local texture jitter. '
        'BMC labels are exploratory agent review and LASIESTA native timing is unverified. '
        'Individual failures, confidence intervals and original dev failures are retained. '
        'This update changes documentation only; pinned weight revisions remain valid.\n').encode()
    model_commit = api.create_commit(repo_id=MODEL, repo_type='model', parent_commit=model.sha,
        operations=[CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=updated)],
        commit_message='docs(dynamic): link frozen external generalization evidence')
    downloaded = Path(hf_hub_download(MODEL, 'README.md', revision=model_commit.oid, endpoint=args.endpoint,
        local_dir=args.downloads / 'model-card', force_download=True))
    if downloaded.read_bytes() != updated:
        raise ValueError('Model card download mismatch')
    receipt.update(status='verified', verified_utc=datetime.now(timezone.utc).isoformat(),
        model_card_revision=model_commit.oid, model_parent_revision=model.sha,
        model_card_sha256=sha(downloaded), model_files_changed=['README.md'])
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'dataset_revision', 'model_card_revision')}), flush=True)


def verify_download(api, args, receipt):
    from huggingface_hub import hf_hub_download
    def fetch(item):
        p = Path(hf_hub_download(DATASET, item['path'], repo_type='dataset', revision=receipt['dataset_revision'],
            endpoint=args.endpoint, local_dir=args.downloads / 'dataset', force_download=True))
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError(f'Download mismatch: {item["path"]}')
        return item['path']
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(fetch, receipt['dataset_files']))
    receipt['dataset_files_fresh_download_verified'] = len(checked)
    receipt['status'] = 'dataset_verified_model_card_pending'
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'fresh_download_files_verified': len(checked)}), flush=True)


def resume(args):
    receipt = read(args.receipt)
    if (receipt['repo_id'] != DATASET or receipt['path'] != PREFIX
            or receipt['status'] not in ('uploaded_pending_download_verification', 'dataset_verified_model_card_pending')
            or not re.fullmatch('[0-9a-f]{40}', receipt['dataset_revision'])):
        raise ValueError('Not an unfinished publication receipt for this release')
    complete(args, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('stage'); p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--research-root', type=Path, required=True); p.add_argument('--extra', type=Path, required=True)
    p = sub.add_parser('publish'); p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True); p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--downloads', type=Path, required=True)
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    p = sub.add_parser('resume'); p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--downloads', type=Path, required=True)
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    args = parser.parse_args(); globals()[args.command](args)


if __name__ == '__main__':
    main()
