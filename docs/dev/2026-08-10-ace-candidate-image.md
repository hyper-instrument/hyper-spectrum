# Building the ACE XAS candidate image from this repository

## Goal

Make this repository self-sufficient to build the traditional-XAS-denoising
candidate container, so a build service handed only a GitHub URL for it can
produce the image — without reaching ACE's repository, its staging script, or
its locally-built base layer.

The image itself is not new. ACE has built it for some time from
`harness/docker/research/traditional-xas-denoising/`. What was not possible was
building it from *here*, and that is the whole of what changed.

## What changed

| File | |
|---|---|
| `docker/ace-candidate/Dockerfile` | The recipe, build context = repository root. |
| `docker/ace-candidate/vendor/{run,answer,verify,entry}.py` | ACE's adapter files, byte-identical. |
| `docker/ace-candidate/vendored.sha256.json` | Per-file SHA-256 + size + upstream path + source commit. |
| `docker/ace-candidate/hyperspectrum-sdk.rev` | The revision this image claims; byte-identical to ACE's `vendor/hyperspectrum-sdk.rev`. |
| `docker/ace-candidate/VENDORED.md` | Truth source, digest table, and the edit-upstream rule. |
| `.dockerignore` | Keeps `COPY .` to roughly what a `git archive` would carry. |
| `tests/docker/test_ace_candidate_vendor.py` | Drift check on the vendored copies. |
| `tests/docker/test_ace_candidate_image.py` | The recipe's contract, and the probe's constants against this checkout. |
| `tests/docker/test_ace_candidate_adapter.py` | The adapter driven by path against the real runtime. |
| `pyproject.toml` | Ruff excludes `docker/ace-candidate/vendor`. |

Truth source for the vendored files: `ace-benchmark` at
`e50b9da7c8acfb5b73ee4cbcf0c475ba7e42e0ef`.

## Key decisions

**The runtime is installed from the build context, and nothing else moved.**
ACE's Dockerfile copies `vendor/hyperspectrum-sdk.tar.gz` — a `git archive` of
one pinned commit, produced by `scripts/stage_hyperspectrum_sdist.py` — checks
its SHA-256, untars it to `/opt/hyperspectrum-sdk`, and installs from there.
Here, `COPY . /opt/hyperspectrum-sdk` puts the same kind of tree at the same
path, and every instruction after it is the same. That is the only intended
difference between the two images.

**Dependency pinning is reproduced exactly, not approximated.** The steps are
verbatim: `pip install "uv==0.10.10"`, then
`uv export --frozen --no-dev --no-emit-project` into a requirements lock, then
`pip install --require-hashes -r <lock>`, then `pip install --no-deps` the
project itself. This matters more than it looks. `uv export --frozen` reads
`uv.lock`, which is in this repository and was in the staged tarball — the same
file either way — so the two builds resolve the same versions *and* the same
wheel hashes. Under `--require-hashes` a dependency that changed under its own
version number fails the build rather than the comparison. NumPy 2.4.6 and SciPy
1.17.1 on Python 3.11 fall out of the lock, which is precisely what
`vendor/verify.py` refuses to run without.

The `uv export` command was run against this checkout during development and
produces an 823-line hash-pinned requirements file with those versions, so the
step is known to work here and not only in ACE's context.

**`verify.py` is vendored but not COPYed.** It is a probe fed to a *built* image
from outside — on stdin by `acebench-lb research build --verify`, or
base64-embedded into a Deploy Master `verify_commands` entry. Copying it in
would add a file to the recipe that the running program never reads.

**The probe's constants are re-used as a source guard.** `verify.py` pins the
savgol tool digest, the resolved implementation digest, and the HyperSpectrum
revision. `test_ace_candidate_image.py` recomputes the first two from this
checkout and compares, and holds `hyperspectrum-sdk.rev` against
`PINNED_REVISION`. So a change to `src/hyperspectrum/` that would make the
image's identity claim false is a red test on a laptop rather than a probe
failure twenty minutes into a remote build. This is the mechanism that keeps the
checked-in revision string honest; it cannot be derived, because a file naming
its own commit cannot exist.

**ACE's `APT_MIRROR` / `NPM_REGISTRY` build args are not reproduced.** They exist
in ACE's shared research base to make `apt-get`/`npm` reachable from a specific
build box. This image installs no system and no Node packages, so with those
args unset — which is how the candidate image is built — the resulting
filesystem is identical either way. `PIP_INDEX_URL` *is* kept: it is genuinely
load-bearing on a slow network, and it is an `ENV` in ACE's image, so keeping it
preserves environment parity too.

**Ruff excludes the vendored directory.** An import-sort fix applied to a pinned
copy is a drift, however correct.

## How to test

```bash
pytest                 # 682 (652 before this branch)
pytest tests/docker    # 30
ruff check .
mypy src
```

Building the image (needs Docker, which none of the review machines have):

```bash
docker build -f docker/ace-candidate/Dockerfile -t hyperspectrum-xas-candidate .
docker run --rm -i hyperspectrum-xas-candidate \
  python - < docker/ace-candidate/vendor/verify.py   # the acceptance probe
```

## Known follow-ups

- **`.dockerignore` is not exactly `git archive`.** It excludes everything
  ignored plus `.git`, but an *untracked, unignored* file in the working tree
  would still enter the build context. It cannot affect the installed runtime —
  `uv_build` packages `src/hyperspectrum` plus the declared license files, and
  the dependency set comes from `uv.lock` — so this is cosmetic rather than
  scientific. A build from a fresh clone has nothing untracked anyway.
- **The image has never been built from this recipe.** Everything above is
  checked statically or by driving the adapter in-process. The first real build
  is the first real test of the `COPY .` layer.
- **Two build homes, one identity.** The image can now be built from either
  repository. Identity is the built image digest either way — not which
  Dockerfile produced it — and ACE's `research/images.py` content-addresses its
  own recipe files, so a build from here does not and cannot claim ACE's recipe
  hash. Re-vendoring is the mechanism that keeps the two from diverging.
