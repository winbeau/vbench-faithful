"""Download a predeclared real, fixed-camera LASIESTA subset from its authors.

No frames or labels are generated. Archives/receipts stay outside Git. The subset
excludes the official illumination, moving-camera, and dynamic-background cases.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

SEQUENCES = ('I_SI_01', 'I_SI_02', 'I_CA_01', 'I_OC_01', 'I_OC_02',
             'I_MB_01', 'I_MB_02', 'I_BS_01', 'I_BS_02')
BASE = 'https://www.gti.ssr.upm.es/images/Data/Downloads/LASIESTA/'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args(); root = Path(args.output).resolve()
    if not args.worker:
        root.mkdir(parents=True, exist_ok=False)
        with (root / 'download.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--output', str(root), '--worker'], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': proc.pid, 'output': str(root)}), flush=True)
        return
    def fetch(name):
        url = BASE + name + '.rar'; path = root / (name + '.rar')
        result = {'sequence': name, 'url': url}
        try:
            with urllib.request.urlopen(url, timeout=45) as response, path.open('xb') as handle:
                result['headers'] = dict(response.headers)
                h = hashlib.sha256(); size = 0
                while block := response.read(1024 * 1024):
                    handle.write(block); h.update(block); size += len(block)
            if size != int(result['headers']['Content-Length']):
                raise ValueError('download length mismatch')
            result.update(status='downloaded', bytes=size, sha256=h.hexdigest())
        except Exception as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (root / (name + '.receipt.json')).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch, SEQUENCES))
    report = {'dataset': 'LASIESTA', 'official_page': 'https://www.gti.ssr.upm.es/data/lasiesta_database.html',
              'license': 'CC BY-SA 4.0', 'sequences': results,
              'status': 'finished' if all(r['status'] == 'downloaded' for r in results) else 'incomplete',
              'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (root / 'completion.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
