# Requested release fixes

Validation: **225 tests passed, 12 skipped**, including five live Docker tests. A fresh all-stage run exported 17 records and verified unchanged resume output. The largest selected pairwise cosine was 0.68267 against a 0.92 threshold, and the semantic duplicate fixture was rejected. See the [machine-readable report](results/release-fixes-validation.json).

## Global semantic deduplication

Semantic deduplication is optional and **disabled by default**, including in the smoke config. Enable `stage4_sample.semantic_dedup_enabled: true` to use the existing embeddings and `dedup_cosine_threshold`. When enabled, Stage 4 normalizes vectors and searches an exact FAISS `IndexFlatIP` containing all retained prompts, regardless of cluster, source, domain, or difficulty. A candidate at or above the threshold is rejected, with a conservative 1e-6 float32 tolerance. Lower thresholds reject more candidates; higher thresholds retain more.

Candidates follow the existing centroid/periphery round-robin preference. After rejecting a candidate, sampling continues within that quota cell. Once a cell is exhausted, its shortfall is reported rather than transferring its budget to another domain or difficulty. Configured quota order resolves cross-cell duplicates. Comparisons only use retained prompts, so a rejected intermediate point does not cause transitive over-removal.

When deduplication is enabled, missing embeddings and selected nonfinite/zero vectors are errors. With it disabled, sampling performs no duplicate comparisons and does not need embeddings unless recovering missing centroid scores. `stage4/sampling_report.json` records whether deduplication was enabled, its threshold, considered duplicate rejections, selected counts, and per-cell shortfalls. The smoke driver checks pairwise similarities and duplicate removal only when enabled. The local fixture includes a differently capitalized version of the rectangle problem for this opt-in check.

Exact comparison avoids approximate-search recall gaps, but requires a float32 index of retained vectors and potentially quadratic comparison work. Query and within-batch matrices are bounded to 512 candidates. This change establishes the threshold behavior; multi-million-record throughput remains unmeasured. Existing run invalidation and rerun publication behavior are deferred: use a fresh run directory.

## Prompt formatting

Stages 1 and 2 preserve multiline prompts, including leading indentation, tabs, blank lines, repeated spaces, and Unicode literals. Only line endings are canonicalized to LF; plain single-line source fields still trim their outer whitespace. IDs hash the resulting preserved text. Collapsing whitespace even just for IDs would incorrectly merge Python programs whose indentation changes their meaning.

Conversation handling remains first-user-turn extraction; preserving full multi-turn/tool conversations is outside this change.

## Restricted Docker execution

Docker is the default backend in the model, all runnable shipped configs, and the calibration harness. Each worker needs a reachable Docker daemon and a pre-pulled image:

```bash
docker pull python:3.12-slim
```

No Python Docker SDK is needed. Configuration supports `docker_image`, `docker_memory_mb` (256 by default), `docker_pids_limit` (32), `timeout_seconds`, `max_code_bytes` (65536), and `max_output_bytes` (4096 per stream). Pin an approved image digest for repeatable deployments. Custom images must provide Python and `/usr/bin/timeout`.

Each container has no host bind mounts or daemon socket, no network, UID/GID 65534, no Linux capabilities, no-new-privileges, a read-only root filesystem, and a 64 MiB temporary filesystem. It receives no caller environment variables. Memory/swap, PID, CPU, open-file, source-size, and captured-output limits bound execution. A container-side deadline complements the host timeout; cleanup forcibly removes that specific container and its descendants. Docker's default seccomp profile remains enabled. These controls use Docker's documented [execution options](https://docs.docker.com/engine/containers/run/) and [seccomp policy](https://docs.docker.com/engine/security/seccomp/).

The Python image contains the standard library, not arbitrary scientific packages. Missing dependency and illustrative-code outcomes retain the filter's existing uncertainty behavior; successful execution is still not proof that a program solves the task. Each snippet gets a fresh container, which favors simplicity and isolation over throughput.

Creation, startup, and cleanup errors raise an infrastructure error; there is no automatic image pull or host fallback. E2B also fails explicitly when unavailable and requests disabled internet access. Its cloud execution remains untested here. The old `subprocess` backend is retained only as an explicit trusted-input option.

These are container isolation controls, not a proof against host-kernel vulnerabilities. Run hostile workloads on a dedicated worker/VM and keep the daemon/kernel and trusted image maintained. Live tests check the actual network, filesystem, environment, UID, capability, seccomp, cgroup, output, memory, and timeout behavior on the compute node.

## Math

The math filter implementation, configuration, tests, and calibration hooks have been removed, along with direct SymPy/ANTLR dependencies. The separate [proposal](math-verification-proposal.md) is deferred; no dedicated math filtering runs.
