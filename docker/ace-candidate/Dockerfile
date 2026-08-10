# The ACE XAS denoising candidate image, built from this repository.
#
#   docker build -f docker/ace-candidate/Dockerfile -t hyperspectrum-xas-candidate .
#
# The build context is the repository root, and that is the whole point of this
# file. ACE has the same recipe (`harness/docker/research/traditional-xas-
# denoising/Dockerfile`), and it installs HyperSpectrum from a tarball that
# `scripts/stage_hyperspectrum_sdist.py` exports out of a checkout it is given.
# That works for a local build and for ACE's own daemon; it does not work for a
# build service handed a GitHub URL, which has ACE's repository or this one and
# never both. Here the source *is* the context, so the URL is sufficient.
#
# The image is otherwise the same image, deliberately and in detail: the same
# pinned interpreter, the same frozen dependency lock installed under
# `--require-hashes`, the same three adapter files at the same paths, the same
# entrypoint, mounts and environment. `tests/docker/test_ace_candidate_image.py`
# holds each of those to the value ACE's side expects, because "same image" is a
# claim about bytes and there is no Docker on the machine that reviews it.
#
# What is NOT reproduced from ACE's base layer: its `APT_MIRROR` and
# `NPM_REGISTRY` build args. Both exist there to make `apt-get`/`npm` reachable
# from a particular build box, and both are no-ops when unset — this image
# installs no system and no Node packages, so with the args unset the resulting
# filesystem is identical either way.

# ACE's research base pins the interpreter by digest rather than by tag, and so
# does this: two repositories that both say `python:3.11-slim` and build a month
# apart have not built the same image, which is exactly what a digit-identical
# comparison between the two lanes cannot survive.
FROM python:3.11-slim@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Where wheels come from. Empty means PyPI, which is the right default and the
# wrong one on some build boxes: PyPI resolves there and then stalls on anything
# larger than a pure-Python package, and the failure reads as "no matching
# distribution" rather than as a timeout. Which index is reachable is a property
# of the machine doing the building, so it is configuration, not a literal.
ARG PIP_INDEX_URL=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# The CLI contract every ACE research image shares — the `--data/--out/--weights`
# parser, and in particular the `--weights` `!` handling. Vendored rather than
# rewritten: getting that wrong is the single most likely defect and it must not
# be re-solved per repository. See `VENDORED.md`.
COPY docker/ace-candidate/vendor/entry.py /opt/acebench/entry.py
ENV PYTHONPATH=/opt/acebench

# Outputs are collected with `docker cp` after the container exits, so `/out` is
# a plain directory rather than a mount point. `/data`, `/weights` and `/archive`
# exist so a misconfigured run fails on a missing *file* — which is a typed
# refusal from the adapter — rather than on a missing directory, which is a
# docker error with nothing in it.
RUN mkdir -p /out /data /weights /archive

WORKDIR /workspace

# The scientific runtime, from this checkout.
#
# `.dockerignore` is load-bearing here. ACE installs a `git archive` of one
# commit: tracked files and nothing else. A `COPY .` that also carried a local
# `.venv`, a stale `dist/` or someone's `runs/` would install from a different
# tree under the same claim of being that commit.
#
# The lock is exported and installed as a separate step from the project itself
# so that `--require-hashes` applies to every dependency. Installed any other
# way, "pinned numpy" means whatever PyPI served that morning; installed this
# way, a dependency that changed under its own version number fails the build
# instead of the comparison.
COPY . /opt/hyperspectrum-sdk
RUN pip install --no-cache-dir "uv==0.10.10" \
    && cd /opt/hyperspectrum-sdk \
    && UV_CACHE_DIR=/tmp/uv-cache uv export --frozen --no-dev --no-emit-project \
         --format requirements-txt --output-file /tmp/hyperspectrum-requirements.lock \
    && pip install --no-cache-dir --require-hashes \
         -r /tmp/hyperspectrum-requirements.lock \
    && pip install --no-cache-dir --no-deps /opt/hyperspectrum-sdk \
    && rm -rf /tmp/hyperspectrum-requirements.lock /tmp/uv-cache

# What this image claims its scientific runtime is. Read by the acceptance probe
# (`vendor/verify.py`), which refuses any other value — and held to this
# checkout's actual tool and implementation digests by
# `tests/docker/test_ace_candidate_image.py`, so the claim cannot quietly become
# a string that no longer describes `src/hyperspectrum/`.
COPY docker/ace-candidate/hyperspectrum-sdk.rev /opt/acebench/hyperspectrum-sdk.rev

COPY docker/ace-candidate/vendor/run.py /opt/acebench/run.py
# Beside `run.py`, so `import answer` resolves off the script's own directory —
# the same way `import entry` already resolves off PYTHONPATH.
COPY docker/ace-candidate/vendor/answer.py /opt/acebench/answer.py

ENTRYPOINT ["python", "/opt/acebench/run.py"]
