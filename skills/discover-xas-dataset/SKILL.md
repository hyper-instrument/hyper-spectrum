---
name: discover-xas-dataset
description: Use when an agent needs to find XAS, XANES, or EXAFS dataset candidates in a named HyperData profile and return catalog evidence for later task selection.
---

# Discover an XAS Dataset

Keep discovery catalog-only and fail closed. HyperSpectrum produces candidate and
readiness evidence; it does not download data, score models, or own a leaderboard.

## Workflow

1. Run the read-only preflight:

   ```bash
   .venv/bin/hyperspectrum doctor --json
   ```

   Parse the single stdout envelope. If `result.ready` is false, report the
   failed checks and stop. Do not inspect credential/profile files, install a
   client, open a login flow, or substitute another command.

2. Verify the exact profile through the public CLI only:

   ```bash
   hyd --profile volcano whoami
   ```

   Replace `volcano` only with the requested profile name. Require both a zero
   exit code and an explicit server-confirmed identity for the selected server.
   A locally cached identity, an identity explicitly unconfirmed by the server,
   or any other cached or unverified output cannot pass, even when the command
   exits zero. Treat that state as an authentication/connection failure and
   stop. Never inspect a profile file, credential, token, or environment secret.

3. Run discovery with the exact user-authorized profile:

   ```bash
   .venv/bin/hyperspectrum data discover --modality xas --profile volcano --json
   ```

   Replace `volcano` only with the requested profile name. Treat stdout as one
   `hyperspectrum-cli/v1` envelope; send diagnostics to stderr. Preserve each
   candidate's dataset code/version, content digest, public metadata, license,
   source queries, parser/axis evidence, ground-truth evidence, pairing evidence,
   readiness score, warnings, and blockers.

4. Return the unaltered envelope or a JSON projection that cites the source
   candidate fields. State ambiguity or no result explicitly. Never infer labels,
   ground truth, pairing, version, digest, access, or license from names alone.

## Stop Conditions

- Exit `3`: authentication/connection failure. Stop and ask the user to repair
  authentication or connectivity; do not read secrets or bypass the profile.
- Exit `4`: required client/asset is missing or unsupported. Stop and request
  authorization before any download or installation.
- Exit `2` or `5`: report the envelope error and stop; do not broaden the query,
  access files, or mutate external state as a fallback.

Discovery is read-only: do not download or open dataset assets, execute tools,
adapt models, train, publish, score, or write leaderboard claims.
