# Vendored ACE candidate adapter

`vendor/` holds four files this repository did not write. They are ACE
Benchmark's half of the XAS candidate contract, copied here byte for byte.

## Truth source

| | |
|---|---|
| Repository | `ace-benchmark` |
| Commit | `e50b9da7c8acfb5b73ee4cbcf0c475ba7e42e0ef` |
| Adapter directory | `harness/docker/research/traditional-xas-denoising/` |
| Shared CLI contract | `harness/docker/research/common/` |

| Vendored file | Upstream path | SHA-256 | Bytes |
|---|---|---|---|
| `run.py` | `harness/docker/research/traditional-xas-denoising/run.py` | `2bbb1b9d39ad6aea7c5385a480f7685d6462797966a4fd12ba11470f1f6af1ec` | 25434 |
| `answer.py` | `harness/docker/research/traditional-xas-denoising/answer.py` | `45b3a862da092019feab18b66208de76033f59b58c5d1bac7023d545c8ecd9ad` | 12699 |
| `verify.py` | `harness/docker/research/traditional-xas-denoising/verify.py` | `93a1ea628e6512c6d28bc6f9cf49963bb96d7d1ae5dbde2cc90f5597c5703b78` | 4303 |
| `entry.py` | `harness/docker/research/common/entry.py` | `5333e3b02820b1baa75fcbe632ef9e18eb690e6cd5c0836f8c2166e412978403` | 7826 |

The same table is in `vendored.sha256.json`, which is the machine-readable
copy and the one `tests/docker/test_ace_candidate_vendor.py` enforces. This
document exists for the reader; that file exists for the build.

## The rule

**Edit upstream. Re-vendor. Never edit here.**

An edit made in `vendor/` produces an image that behaves like nothing ACE will
ever build. The dual-asset board's claim is that two lanes compute the *same*
function of the *same* bytes and agree digit for digit; a local patch here
breaks that claim silently, because the two images still start, still write a
well-formed answer, and still disagree only in the numbers.

To take a newer upstream:

1. Land the change in `ace-benchmark`, in the paths above.
2. Copy the files across unchanged and rewrite `vendored.sha256.json` (digests,
   sizes, and the `source.commit`).
3. Update this table and run `pytest tests/docker/`.

Step 3 is not a formality. Besides the digest check, that directory holds the
tests that drive these copies against HyperSpectrum-shaped fixtures — the only
evidence available on a machine with no Docker that the copies still work with
*this* checkout of the runtime they call.

## Which of the four the image installs

Three are `COPY`d by `Dockerfile`; `verify.py` is not, and that is deliberate.

- `entry.py` → `/opt/acebench/entry.py`, on `PYTHONPATH`. The `--data/--out/
  --weights` CLI contract every ACE research image shares.
- `run.py` → `/opt/acebench/run.py`. The entrypoint.
- `answer.py` → `/opt/acebench/answer.py`, beside `run.py` so `import answer`
  resolves off the script's own directory.
- `verify.py` is a *probe*, fed to a built image from outside — by
  `acebench-lb research build --verify` on stdin, or base64-embedded into a
  Deploy Master `verify_commands` entry. Copying it into the image would put a
  file into the recipe that the running program never reads.

It is vendored anyway because it is the executable statement of what this image
is required to contain — the pinned HyperSpectrum revision, the savgol tool and
implementation digests, NumPy 2.4.6, SciPy 1.17.1, and the presence of both
execution facades. `tests/docker/test_ace_candidate_image.py` holds this
checkout against those constants, which is how a drift in `src/hyperspectrum/`
becomes a red test here rather than a probe failure after a 20-minute build.
