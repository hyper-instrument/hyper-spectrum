"""The vendored ACE adapter is a copy, and a copy that drifts is a fork.

``docker/ace-candidate/vendor/`` holds bytes this repository did not author.
They are ACE's half of the candidate contract, and the only reason they live
here is that a build service pointed at this GitHub repository has no way to
reach ACE's. That makes drift the whole risk: an edit made here rather than
upstream produces an image that behaves like nothing ACE will ever build, and
the digit-identical claim the dual-asset board rests on quietly stops meaning
anything.

So the manifest is the contract, and these tests are what makes it one.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DIR = ROOT / "docker" / "ace-candidate"
VENDOR_DIR = CANDIDATE_DIR / "vendor"
MANIFEST_PATH = CANDIDATE_DIR / "vendored.sha256.json"


def manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def recorded_files() -> dict[str, dict[str, object]]:
    files = manifest()["files"]
    assert isinstance(files, dict)
    return files


def test_every_vendored_file_matches_the_digest_the_manifest_records() -> None:
    """The drift check itself. Red here means someone edited a copy."""

    for name, record in recorded_files().items():
        payload = (VENDOR_DIR / name).read_bytes()
        assert sha256(payload).hexdigest() == record["sha256"], name
        assert len(payload) == record["size"], name


def test_the_manifest_records_every_vendored_file_and_no_others() -> None:
    """A file added to the directory without a digest is unvendored code.

    Checked in this direction as well as the other because the digest loop
    above passes trivially for anything it is not told about, and an
    unrecorded file in ``vendor/`` is exactly the shape a local edit takes
    when someone means well: a new helper beside the copies.
    """

    on_disk = {path.name for path in VENDOR_DIR.iterdir() if path.is_file()}

    assert on_disk == set(recorded_files())


def test_the_manifest_names_the_upstream_commit_each_copy_was_taken_from() -> None:
    """Without a resolvable source, "re-vendor from upstream" is not an action."""

    document = manifest()
    source = document["source"]
    assert isinstance(source, dict)
    assert source["repository"] == "ace-benchmark"
    assert isinstance(source["commit"], str)
    assert len(source["commit"]) == 40
    assert int(source["commit"], 16) >= 0
    for name, record in recorded_files().items():
        path = record["upstream_path"]
        assert isinstance(path, str)
        assert path.endswith(f"/{name}")
        assert not path.startswith("/")
