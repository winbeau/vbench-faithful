"""Bounded download of all nine author-hosted BMC real recordings, not synthetic data."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

PAGE = 'https://backgroundmodelschallenge.eu/'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args(); root = Path(args.output).resolve()
    if not args.worker:
        root.mkdir(parents=True, exist_ok=False)
        with (root / 'download.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--output', str(root), '--worker'], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'output': str(root)}), flush=True)
        return
    with urllib.request.urlopen(PAGE, timeout=45) as response:
        html = response.read()
    (root / 'official-page.html').write_bytes(html)
    links = re.findall(r'href=[\"\']([^\"\']+)[\"\']', html.decode())
    urls = sorted({urllib.parse.urljoin(PAGE, link) for link in links
                   if re.fullmatch(r'(?:\./)?data/real/Video_00[1-9]\.zip', link)})
    expected = [PAGE + f'data/real/Video_{i:03d}.zip' for i in range(1, 10)]
    if urls != expected:
        raise ValueError('author page does not contain exactly the expected nine real archives')
    def fetch(url):
        name = url.rsplit('/', 1)[-1]; path = root / name
        result = {'sequence': path.stem, 'url': url}
        try:
            with urllib.request.urlopen(url, timeout=60) as response, path.open('xb') as handle:
                result['headers'] = dict(response.headers); h = hashlib.sha256(); size = 0
                while block := response.read(1024 * 1024):
                    handle.write(block); h.update(block); size += len(block)
                    if size > 2 * 1024 ** 3:
                        raise ValueError('archive exceeds task limit')
            length = result['headers'].get('Content-Length')
            if length is not None and int(length) != size:
                raise ValueError('download size mismatch')
            result.update(status='downloaded', bytes=size, sha256=h.hexdigest())
        except Exception as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (root / (name + '.receipt.json')).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(fetch, urls))
    report = {'dataset': 'BMC2012 real evaluation videos', 'official_page': PAGE,
              'page_sha256': hashlib.sha256(html).hexdigest(), 'sequences': rows,
              'status': 'finished' if all(r['status'] == 'downloaded' for r in rows) else 'incomplete',
              'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (root / 'completion.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
