import hashlib
from pathlib import Path

import pytest

from scripts.verify_dynamic_generalization import verify_files


def entry(path):
    return {'path': path.name, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_manifest_checks_every_byte_and_rejects_duplicates(tmp_path):
    path = tmp_path / 'scores.jsonl'; path.write_text('original')
    record = entry(path)
    assert verify_files(tmp_path, [record]) == 1
    with pytest.raises(ValueError, match='Duplicate'):
        verify_files(tmp_path, [record, record])
    path.write_text('modified')
    with pytest.raises(ValueError, match='Content mismatch'):
        verify_files(tmp_path, [record])


@pytest.mark.parametrize('name', ['../outside.json', '/tmp/outside.json'])
def test_manifest_rejects_escaping_members(tmp_path, name):
    with pytest.raises(ValueError, match='Unsafe'):
        verify_files(tmp_path, [{'path': name, 'bytes': 0, 'sha256': ''}])


def test_manifest_rejects_symbolic_links(tmp_path):
    source = tmp_path / 'source.json'; source.write_text('{}')
    link = tmp_path / 'link.json'; link.symlink_to(source)
    with pytest.raises(ValueError, match='Unsafe'):
        verify_files(tmp_path, [entry(link)])
