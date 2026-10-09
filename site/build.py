"""Validate the public showcase and assemble a dependency-free Pages artifact."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-media', action='store_true', help='Also decode media metadata with FFprobe')
    args = parser.parse_args()
    plans = {name: json.loads((ROOT / 'data' / f'{name}.json').read_text())
             for name in ('semantic', 'nuisance', 'spatial')}
    cases = [case for plan in plans.values()
             for case in plan['cases'] + plan.get('static_controls', [])]
    assert len({case['id'] for case in cases}) == len(cases) == 30
    assert len({case['dimension'] for case in cases}) == 9
    numeric = missing = 0
    for case in cases:
        for side in ('original', 'counterfactual'):
            media = case['videos'][side]
            path = ROOT / media['src']
            assert digest(path) == media['sha256'], f'Media changed: {path}'
            assert media['duration_seconds'] == 5
            assert (ROOT / media['poster']).is_file()
            if args.verify_media:
                meta = json.loads(subprocess.check_output([
                    'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                    '-show_entries', 'stream=duration,nb_read_frames,r_frame_rate,width,height',
                    '-of', 'json', str(path)], text=True))['streams'][0]
                assert float(meta['duration']) == 5, path
                assert int(meta['nb_read_frames']) == media['frames'], path
                assert [meta['width'], meta['height']] == media['resolution'], path
        provenance = case['provenance']
        assert digest(ROOT / provenance['first_frame']) == provenance['first_frame_sha256']
        for backend in ('vbench', 'ours'):
            for i, value in enumerate(case['scores'][backend]):
                row = case['score_records'][backend][i]
                assert value == row['score']
                assert case['score_status'][backend][i] == row['status']
                if value is None:
                    assert row['status'] != 'succeeded' and case['score_reasons'][backend][i]
                    missing += 1
                else:
                    assert row['status'] == 'succeeded' and math.isfinite(value)
                    numeric += 1
    out = ROOT.parent / 'output' / 'paper-site'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for folder in ('assets', 'data'):
        shutil.copytree(ROOT / folder, out / folder, dirs_exist_ok=True)
    for path in ROOT.iterdir():
        if path.suffix in {'.html', '.css', '.js'}:
            shutil.copy2(path, out / path.name)
    browser = '\n'.join(f'window.{name.upper()}_CASE_PLAN = '
                        + json.dumps(plan, ensure_ascii=False, separators=(',', ':')) + ';'
                        for name, plan in plans.items())
    (out / 'data' / 'cases.js').write_text(browser + '\n')
    (out / '.nojekyll').write_text('')
    print(f'Built {out}: {len(cases)} pairs, {numeric} measured scores, {missing} explicit missing scores.')


if __name__ == '__main__':
    main()
