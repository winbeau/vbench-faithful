"""Refresh or verify the exact static file set deployed by GitHub Actions."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    browser = '\n'.join(
        f'window.{name.upper()}_CASE_PLAN = '
        + json.dumps(json.loads((ROOT / 'data' / f'{name}.json').read_text()),
                     ensure_ascii=False, separators=(',', ':')) + ';'
        for name in ('semantic', 'nuisance', 'spatial')) + '\n'
    browser_path = ROOT / 'data/cases.js'
    if args.check:
        assert browser_path.read_text() == browser, 'Case JSON changed; run pnpm manifest.'
    else:
        browser_path.write_text(browser)
    target = ROOT / 'data/deployment.json'
    manifest = json.loads(target.read_text())
    paths = [p for p in ROOT.iterdir()
             if p.is_file() and (p.suffix in {'.html', '.css', '.js'} or p.name == '.nojekyll')]
    for folder in ('assets', 'data'):
        paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and p != target)
    files = []
    for path in sorted(paths):
        assert not path.is_symlink(), path
        files.append({'path':path.relative_to(ROOT).as_posix(), 'bytes':path.stat().st_size,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if args.check:
        assert files == manifest['files'], 'Static files changed; run pnpm manifest before committing.'
        print(f'Verified {len(files)} static files.')
    else:
        manifest['files'] = files
        target.write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'Updated deployment manifest: {len(files)} static files.')


if __name__ == '__main__':
    main()
