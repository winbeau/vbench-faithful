#!/usr/bin/env python3
"""Lexical screening of the already-frozen challenge; never remove items or change labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
from audit_training_sources import near_match, norm
from build_ablation_eval import VERSIONS, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/ablation-v1/eval-extended.jsonl')
    p.add_argument('--out', default='output/ablation-v1/source-audit.json')
    args = p.parse_args()
    test = [json.loads(s) for s in Path(args.data).read_text().splitlines()]
    pairs, summary, source_hashes = [], {}, {args.data: sha(args.data)}
    for task, version in VERSIONS.items():
        splits = {}
        for split in ('train', 'dev'):
            path = Path(f'data/formal/{version}/{task}/{split}.jsonl')
            splits[split] = [json.loads(s) for s in path.read_text().splitlines()]
            source_hashes[str(path)] = sha(path)
        task_summary = {'train_dev_group_overlap': len({r['group_id'] for r in splits['train']} & {r['group_id'] for r in splits['dev']}), 'splits': {}}
        for split, training in splits.items():
            unique = {}
            for row in training:
                unique.setdefault(norm(row['input']['prompt']), row['sample_id'])
            exact, near, planned = Counter(), Counter(), Counter()
            for row in test:
                if row['task'] != task or row['source'] == 'existing_dev':
                    continue
                group = row['source'] + '/' + row['variant']
                planned[group] += 1
                query = norm(row['input']['prompt'])
                hits = []
                for text, sample_id in unique.items():
                    ratio, kind = near_match(query, text)
                    if kind:
                        hits.append({'sample_id': sample_id, 'similarity': ratio, 'kind': kind})
                if hits:
                    near[group] += 1
                    exact[group] += any(h['kind'] == 'normalized_exact' for h in hits)
                    pairs.append({'test_id': row['id'], 'task': task, 'split': split, 'hits': hits})
            task_summary['splits'][split] = {'train_rows': len(training), 'unique_prompts': len(unique),
                                            'planned': dict(planned), 'normalized_exact_items': dict(exact),
                                            'near_or_exact_items': dict(near)}
        summary[task] = task_summary
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'tasks': summary, 'input_sha256': source_hashes,
                               'method': 'Existing fixed audit: normalized exact, character ratio >=0.90, or >=12-word containment.',
                               'limits': ['No evaluation items removed; no outcome information used.',
                                          'Lexical resemblance is neither proof nor disproof of common provenance.',
                                          'Engineered text tests have no independent new visual source identity.',
                                          'Existing dev excluded from blind-text audit because it is intentionally a seen diagnostic.']}, indent=2) + '\n')
    out.with_suffix('.pairs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pairs))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
