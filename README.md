# Multilingual output and model language preference

Set `NMESH_LANG=ja` (or `en`) to localize nmesh's runtime, planner, and CLI
output. Argparse help text remains English and is intentionally outside this
translation layer.

Use `nmesh plan --lang ja,en` or `nmesh up --lang ja,en` to softly prioritize
models that claim those languages. This is not a hard filter: nmesh still
selects a runnable model when no candidate claims every requested language.
Catalog language lists are publisher/vendor claims, not measured benchmarks.
# nmesh

nmesh は、ローカル LLM のためのハードウェア自動検出、モデル選択、メモリ見積もり、
実行管理ツールです。ノート PC から複数 GPU のワークステーションまで構成を自動で計画し、
OpenAI 互換 API を提供します。

## クイックスタート

```text
pip install -e .
nmesh doctor
nmesh plan
nmesh up
```

Gateway は `http://127.0.0.1:18000/v1` の OpenAI 互換エンドポイントで待ち受けます。

## ハードウェア tier

| Tier | 条件 |
|---|---|
| T0_CPU | GPU なし、または VRAM 2 GiB 未満 |
| T1_LOW | 2–6 GiB |
| T2_MID | 6–12 GiB |
| T3_HIGH | 12–24 GiB |
| T4_WORKSTATION | 24–48 GiB |
| T5_SERVER | 48 GiB 以上、または GPU 2 枚以上 |

## バックエンドとモデル

Ollama、llama.cpp、vLLM、MLX-LM をサポートします。未インストールでも
`nmesh plan` は候補と具体的なインストール方法を表示します。
モデルの追加や上書きは `~/.nmesh/models.yaml` に bundled catalog と同じ形式で記述します。
同じ `id` のモデルはユーザー定義が優先されます。

## CI workflow

GitHub の権限がある利用者は `ci/github-workflow-ci.yml` を
`.github/workflows/ci.yml` にコピーして使用してください。

## Gateway service

`nmesh serve --port 18000` runs only the OpenAI-compatible gateway in the
foreground when backend services are already running. The gateway exposes
`GET http://127.0.0.1:18000/metrics`, which returns per-service live telemetry
including sample count, median decode throughput, TTFT, and total latency.
`nmesh reload --port 18000` asks a running gateway to reload the latest
`~/.nmesh/plan.json` without restarting it.
The gateway server enables a 15-second watchdog that revives failed services;
use `create_app(..., watchdog=False)` when embedding it without supervision.
Set `NMESH_QUEUE_TIMEOUT` to control how long chat requests wait for a backend
concurrency slot before receiving a retryable 503 response.

Runtime state is persisted atomically in `~/.nmesh/state.json`; `status` checks
PID liveness and `down` can terminate services owned by another process.
Supervisor `atexit` cleanup is armed only after a child is actually launched,
so read-only commands such as `status` and `bench` do not erase live runtime
state.

For llama.cpp services, `nmesh up` enumerates the GGUF files published by the
configured Hugging Face repository and resolves the planned quantization to a
real filename. Split GGUF files are downloaded as a complete set; if only a
lower quantization is published, the runtime records that safe substitution in
`status` and the persisted state.

## Concurrency slots

Plans automatically size concurrency slots from memory left after placement;
`Policy.parallel_slots` can force a specific slot count.

## Memory budget controls

`nmesh plan --budget free --explain` plans against currently free VRAM and RAM
instead of the machine's total capability. The default `--budget total` keeps
the persisted plan stable as a capability description.

At launch, `nmesh up` proactively checks current free memory and replans when
the saved plan no longer fits. Use `nmesh up --ignore-free-memory` to skip that
check and rely on the normal runtime fallback ladder. `nmesh doctor` displays
each GPU's total/free VRAM and the resulting free VRAM/RAM budgets.

## State and port environment variables

`NMESH_HOME` relocates all nmesh-owned persistent and downloaded state:
the plan, runtime state, benchmark cache, capability cache, telemetry, token
calibration (`tokens.json`), user catalog, gateway log, and downloaded GGUFs. Set it to a scratch directory
before starting nmesh processes to run isolated experiments without touching
the real `~/.nmesh` state or catalog.

`NMESH_SERVICE_PORT_BASE` changes the base port used for planned backend
services. The default is `18010`; nmesh assigns subsequent service ports from
that base. Use a free base (as the E2E harness does) and ensure the gateway
port selected for `nmesh up --port` does not conflict with it.

## Hardware profile simulation

`nmesh doctor --json` output can be saved on a GPU machine and consumed
elsewhere with `nmesh plan --profile mine.json`. The profile input exercises
planning decisions only; it does not verify runtime behavior or measure GPU
performance. A simulated plan is marked as such and never writes saved
`plan.json` state, so it cannot become the local machine's runtime plan.

The repository's `profiles/` files are synthetic examples for planner
regression testing. To use a real machine's description:

```text
nmesh doctor --json > mine.json
nmesh plan --profile mine.json
```

## GPU detection and honest VRAM reporting

nmesh first tries the specialized NVIDIA and ROCm detectors: NVML or
`nvidia-smi` for NVIDIA, and `rocm-smi` for AMD. The generic detector runs only
when all of those specialized paths return no GPUs. On Windows it reads
`Win32_VideoController` and the display-adapter registry; on Linux it reads
DRM device vendor IDs and AMD's `mem_info_vram_total`. Apple Silicon keeps its
existing unified-memory path rather than going through generic discrete-GPU
detection.

`nmesh doctor` shows each detected adapter and includes the `vram_source`
provenance beside its VRAM amount (`nvml`, `smi`, `registry`, `sysfs`, or
`unknown`). An Intel integrated GPU can therefore be visible in `doctor` while
the plan remains `T0_CPU`: shared system memory is not counted as dedicated
VRAM, and the tier threshold is intentionally not inflated. Such an adapter
gets a warning explaining that its dedicated VRAM is below the placement
threshold.

Registry- and sysfs-derived adapters do not provide measured free VRAM.
nmesh conservatively reports `free = total` for those adapters; consequently,
`--budget free` is an upper bound rather than a measurement for them (the
normal display reserve is still subtracted). NVML and SMI values are the
sources that provide runtime free-memory measurements.

## Backend GPU capability checks

GPU-layer flags in a backend's help output are not proof that the backend can
execute on a GPU. In particular, the CPU-only llama.cpp build b10734 advertises
`-ngl` in `--help` but reports:

```text
Available devices:
  (none)
```

nmesh therefore asks llama.cpp to enumerate its devices. A successful empty
result, `()`, forces CPU placement and honest RAM accounting instead of
emitting GPU-layer arguments. An unknown result, `None` (for example when the
probe fails), preserves the previous behavior rather than making an
unsupported CPU fallback assumption. To use a detected GPU with llama.cpp,
install or build a backend with a Vulkan, CUDA, HIP, or SYCL GPU backend.

## Real-backend end-to-end harness

The optional real-backend harness runs the complete planner, runtime, gateway,
completion, metrics, benchmark, and teardown workflow without using the
user's normal state directory:

```text
python scripts/e2e.py
```

It creates a temporary `NMESH_HOME`, selects free ports, and uses a local
`llama-server` plus GGUF model. When possible, it reuses an already-downloaded
GGUF from the real `~/.nmesh/models` directory read-only, by hard-linking,
symlinking, or copying it into the scratch directory. Set `NMESH_E2E_MODEL`
to choose a different local model. This is intentionally standalone rather
than a pytest test because it requires a real model and can take several
minutes.

Exit codes:

* `0` — every real-backend step passed and all selected ports were released.
* `1` — a required step failed.
* `77` — the required real backend or local GGUF model is unavailable.

## OpenAI-compatible API surface

In addition to `/v1/chat/completions`, the gateway provides the legacy
`/v1/completions` endpoint. Its `prompt` may be one string or a list of
strings; nmesh concatenates that input for context-length routing. Both
completion endpoints honor explicit `nmesh-<service>` model IDs, streaming,
backend slot limits, swap ordering, and telemetry.

Set `NMESH_API_KEY` before starting the gateway to require
`Authorization: Bearer <key>` on `/v1/*` and `/metrics*`. `/health` remains
unauthenticated for readiness probes. When the variable is unset, authentication
is disabled. The key is compared securely and is never returned in errors,
metrics, or logs.

The existing `/metrics` endpoint remains JSON for compatibility.
`/metrics/prometheus` adds Prometheus 0.0.4 text exposition for the telemetry
aggregates and concurrency values already present in that JSON. Telemetry
series carry an `approximate` label so estimated values are not presented as
measurements. The decode-throughput family is named
`nmesh_telemetry_decode_tokens_per_second_median`; time values use the
`_seconds` base unit.
Only telemetry samples recorded with one request in flight feed the planner's
decode-rate overlay. Samples observed under load remain visible in
`nmesh status` telemetry but never override the benchmark value. On one CPU
machine with llama.cpp and one model, four and eight overlapping requests
reduced the per-request decode rates to 36.64 and 23.30 tok/s versus about
46.2 tok/s alone, while aggregate server throughput rose; this scope does not
generalize.
HTTP response bodies remain English because `/v1/*` errors and authentication
details are machine-facing API contracts for clients.

## Resident operation and restart recovery

`nmesh autostart --install` writes a launcher and, when absent, a
`gateway.env` file under `NMESH_HOME`. The launcher reads `KEY=VALUE` entries
from that sibling file, sets `NMESH_HOME` to its own directory before Python
starts, and runs `python -m nmesh.gateway.server --port 18000` (with the
configured port). The API key belongs only in `gateway.env`; it is not embedded
in a service unit or Task Scheduler command line. Existing `gateway.env` files
are never overwritten. On POSIX systems the launcher is mode `0700` and the
environment file is mode `0600`.

After a restart, the gateway watchdog reloads the saved plan and recovers
resident services, services recorded in `state.json`, and already-listening
services. Other services remain on demand until a request needs them.

The generated service definitions intentionally reflect platform differences:

* systemd uses `Restart=on-failure`, so the gateway is restarted after a
  failure;
* launchd uses both `RunAtLoad` and `KeepAlive`, so it is loaded at login and
  restarted after a crash;
* Windows output uses `schtasks /sc onlogon`: it does not start until a user
  logs on. Boot startup requires changing this to `/sc onstart` and choosing
  SYSTEM or saved credentials. The current simple Task Scheduler setup does
  not restart the gateway after a self-crash.

### Prompt token accounting

Context-length routing needs a prompt-token count. The built-in heuristic
(CJK characters count as 1.0 tokens, other characters as 0.25) is only a
conservative guess: measured against the real Qwen2.5-1.5B-Instruct tokenizer
through llama.cpp `POST /tokenize`, the heuristic divided by the real count is
1.71-1.75 for Japanese prose, 1.12 for English prose, 1.04 for Korean, 1.38 for
Chinese, and 0.82 for Python code. It errs in both directions and, worst of
all, under-counts code, which is exactly the content routed to the code
service.

nmesh therefore calibrates the two coefficients per model. Whenever an upstream
response reports an exact `usage.prompt_tokens`, the character counts and that
exact token count are accumulated in `tokens.json` under `NMESH_HOME`, keyed by
model id because the tokenizer is a property of the model rather than of the
service. Coefficients are fitted by least squares and are only used once a
model has at least 20 exact samples and the fit is well-conditioned and within
sane bounds; otherwise the defaults above are used. Estimates are never fed
back into the calibration.

The calibration is exposed on `/metrics` and on `/metrics/prometheus` as
`nmesh_token_calibration_cjk_per_char`, `nmesh_token_calibration_other_per_char`
and `nmesh_token_calibration_samples`, labelled with `service`, `model`, and
`kind="chat|text"`, and with `measured="true|false"`. `measured="false"` means
you are looking at
the default guess, not a measurement.

Near the routing boundary only - when the calibrated estimate lies between half
and twice the remaining context threshold - and only when the chat service is a
running llama.cpp service, nmesh asks that backend for the exact count through
`POST /tokenize` with a short timeout. Any failure or timeout falls back to the
estimate and never fails the request. The vLLM tokenizer endpoint is
deliberately not used because its shape is unverified here.

### Model quality micro-evaluation

`nmesh eval` runs a deterministic 16-task micro-evaluation covering instruction
following, output format, value extraction, and translation direction. Strict
output discipline is measured separately by the `compliance` family. It is not a
knowledge benchmark and its pass rate must not be compared with MMLU-style
scores. The catalog `quality` field remains an unverified prior used for
planning; measured results are saved in `eval.json`. When an evaluation
contradicts that prior, `nmesh plan` keeps the original ranking but emits a
warning naming both measurements and catalog claims.

Evaluation results are specific to the tuple `(model, quant, backend)`, not
just the model ID. On one CPU machine, the same Q4_K_M label for the same
Qwen 1.5B model scored 12/16 with llama.cpp and 13/16 with Ollama; the
disagreeing `extraction.date` result was verbose under llama.cpp
(`The date in YYYY-MM-DD form is: 2024-03-03`) and bare under Ollama
(`2024-03-03`). Injecting Ollama's system prompt into the llama.cpp requests
did not change that result, so the system-prompt hypothesis was rejected.
These measurements compare different weight artifacts and were taken on one
model on one CPU machine; they do not generalize.

The same limitation applies within fp16: on that machine, Qwen2.5 0.5B
scored 9/16 with both llama.cpp and Ollama, while disagreeing on
`arithmetic.subtract` (`767` vs `747`) and `multilingual.ja_translate`
(Ollama left "sleeping" untranslated). These are opposite directions and
both configurations are fp16, so the divergence is not a quantization
artifact. Equal pass rates therefore do not imply equivalent behaviour.
This is one model on one CPU machine and does not generalize; no underlying
cause is established by this measurement.

Each evaluation record also carries a grader digest. Records graded by an
earlier rule version are not used for planning and must be re-measured; this
exclusion is per record, not per task, including tasks whose grading did not
change. The two core email/date tasks contain a single candidate value, so
passing them proves that the value is right, not that the model selected it
between candidates. The generated extraction family adds decoys and tests that
selection directly.

Further controlled checks scoped those eliminations to `arithmetic.subtract`:
with the same fully expanded 48-token raw ChatML prompt, `top_k=1`,
`repeat_penalty` pinned to both 1.0 and 1.1, and cold or warm llama.cpp
prompt-cache state, the answer still differed. A single llama.cpp binary with
the same `-c 4096 -t 8` flags and prompt, changing only the weights file,
answered `767` with the official GGUF and `747` with the Ollama blob. This
establishes the weights file as the variable that changes that answer, but not
which in-file difference does so; the separate or missing `output.weight` is
not asserted as the mechanism. With that same raw prompt, both artifacts
answered `猫は sleeping です。` for `multilingual.ja_translate`, so that
task's earlier divergence is not an artifact effect and remains unexplained at
the chat layer. This remains one model on one CPU machine and does not
generalize; eval records now carry an artifact fingerprint.

### Evaluation uncertainty and resolving power

The evaluation gate and its resolving power were checked with exact
binomial/Fisher calculations:

- The contradiction gate of `0.05` equals `0.8` tasks out of 16, while one
  task is `0.0625`; a single flipped task can trip the gate.
- At `n=16`, unpaired exact Fisher at `α=0.05` cannot resolve a pass-rate
  difference below `0.3125`; against `12/16`, the opponent must be at most
  `5/16` (a `0.44` gap) to be significant.
- The measured comparisons were non-significant: `12/16` versus `9/16`
  yielded `p=0.4578`, and `13/16` versus `12/16` (the backend comparison)
  yielded `p=1.0`.
- Wilson 95% intervals are much wider than the gate: `9/16 = 0.562`
  [`0.332`, `0.769`], `12/16 = 0.750` [`0.505`, `0.898`], and
  `13/16 = 0.812` [`0.570`, `0.934`].
- For paired task results, exact McNemar requires at least six discordant tasks
  all in one direction for `p<0.05`; the measured `1`-versus-`1` discordance
  gave `p=1.0`.
- At 80% power and `α=0.05`, the suite needs at least 62 tasks to resolve a
  `0.20` gap and 294 tasks to resolve a `0.10` gap.

Model selection is unchanged: these results only gate the contradiction
warning and report underpowered evidence. A measured quality *floor* for the
planner's score is not yet possible at this suite size.

#### Extending the evaluation suite

The extension is enumerated without RNG and uses code-only verifiers: 16 core
tasks plus 88 generated tasks make 104 tasks. Extraction tasks grade the
extracted value, while strict output discipline is measured by `compliance`.
Every evaluation record carries a grader digest, and records graded by different
rules are never compared.

The default remains the `core` suite, so historical records and runtime
behaviour are unchanged; `nmesh eval --suite extended` is opt-in. Tasks within
one family are not independent samples, so the effective sample size is below
104 and the exact tests are optimistic to that extent.

The extended suite was run on one CPU machine, both models on llama.cpp so the
backend and artifact effects found earlier are not in the way. Under the first
grader version, which required extraction answers to be emitted alone, Qwen2.5
1.5B Q4_K_M scored `72/96` and Qwen2.5 0.5B fp16 scored `62/96`; the unpaired
exact Fisher test on those totals was not significant (`p=0.1569`) and only
exact McNemar on the paired outcomes was (13 versus 3 discordant, `p=0.0213`).
The `extraction.*` category rates were `0.348` and `0.174`: both models failed
those tasks, so they were concordant and carried no discriminating power.

Those failures were not wrong extracted values but correct values wrapped in
prose, so grader version 2 splits the two measurements: `extraction.*` grades
the extracted value and `compliance.*` grades strict output discipline. Re-run
on the same two artifacts and the same machine, the 104-task suite measures:

| model | result | Wilson 95% |
| --- | --- | --- |
| Qwen2.5 1.5B Q4_K_M | `89/104` = `0.856` | [`0.776`, `0.911`] |
| Qwen2.5 0.5B fp16 | `70/104` = `0.673` | [`0.578`, `0.756`] |

- unpaired exact Fisher is now significant: `p=0.0030` (it was `p=0.1569`);
- exact McNemar on the 104 paired outcomes: 24 tasks passed only on the 1.5B,
  5 only on the 0.5B, 75 concordant, `p=0.0005` (it was `p=0.0213`);
- the same paired comparison restricted to the 16 core tasks gives `5` versus
  `0` discordant tasks and `p=0.0625`, still not significant at `α=0.05`.

Fixing what the tasks measured bought more resolution than adding 80 tasks did.
`extraction.*` went from the least informative family to the most informative
one (`0.739` versus `0.435`, 10 versus 3 discordant), and the discipline that
used to be conflated with it is now visible on its own: `compliance.*` is
`1.000` versus `0.250` with 6 versus 0 discordant tasks. `instruction.*` and
`multilingual.*` are saturated at `1.000` for both models and contribute no
discordant pairs, so they carry no discriminating power at this model pair.

This is the first catalog quality ranking in this repository backed by a
significant measurement rather than a hand-written prior, by both a paired and
an unpaired exact test. It is two models on one CPU machine and does not
establish a ranking for any other model pair.

### Periodic external watch

`nmesh watch` treats external posts as claims and pointers, not evidence.
Zenn RSS: 120 items = 41,405 chars total → 0 CLI flags, 0 `/v1/` routes
extractable, which is why bodies are fetched through the article API + HTML.
The `--limit` value applies independently to every source tag or topic;
per-URL deduplication still applies across the combined result.

Qiita API bodies: 100 items = 1,052,406 chars → 190 distinct flags, of which
132 (69%) are unknown to this machine's 331-flag llama-server. The top
unknowns are vLLM flags:
`--gpu-memory-utilization`, `--tensor-parallel-size`, `--max-model-len`,
`--max-num-seqs`, `--enforce-eager`, and `--kv-cache-dtype`. Unknown means
not this backend's flag, not that the feature is missing.

41 distinct Hugging Face repo ids were mentioned across Zenn+Qiita; 0 of them
appear among the 51 repo ids in the bundled catalog. The catalog is Qwen2.5-era
while current discussion includes Qwen3.x, gemma-4, MiniMax-H3, llm-jp-4, and
Nemotron. Mention counts indicate popularity, not quality — quality still
requires `nmesh eval`, so drafts carry `quality: null`.

Noise IDs such as `docs/hub`, `papers/2504.13181`, `datasets/leemeng`, and
`blog/nvidia` are filtered by the Hugging Face 401/404 gate. GGUF mirror
repositories commonly return 404 for `config.json`, including
`Qwen/Qwen3-14B-GGUF` and `ggml-org/...-GGUF`; architecture numbers then come
from the base repository recorded in `config_repo`. GitHub's API returned 403
from this box, so there is deliberately no releases source. X is unavailable
without `NMESH_X_BEARER_TOKEN`. `--offline` accepts saved source items for
reproducible extraction and verification, and bounded state prevents repeated
findings from growing without limit. No finding is auto-applied to the product.

### Speed preference saturation

The planner's simulated bundled profiles show that the speed term is already
at its 30 tok/s ceiling for 82 of 83 candidates on
`t3-rtx4090-24gb`, 67 of 72 on `t2-rtx3060-12gb`, and 68 of 89 on
`t4-rtx6000ada-48gb`. On t3 with `--prefer speed`, it ranks
`qwen2.5-32b-instruct` `q4_k_m` at 38.49 tok/s (score 140.25) above
`phi-4-14b` `q5_k_m` at 73.68 tok/s (score 138.00), so the slower model wins
because both receive the same maximum speed points and the unvalidated quality
prior decides. These are the planner's own `_throughput` estimates on bundled
simulated profiles, not wall-clock measurements.

Changing the speed reference does not solve that degeneracy: with
`--prefer quality` and references 30/60/120/240, the selected candidates were
32B `q4_k_m` at 38.5, `phi-4-14b` `q6_k` at 63.6, 32B `q4_k_m` at 38.5,
and 32B `q5_k_m` at 9.0 tok/s, respectively. Removing the ceiling makes the
fastest tiny candidate, `qwen2.5-0.5b-instruct` `q2_k` at 3654.6 tok/s, win
regardless of quality. The formula was deliberately left unchanged: a
scale-free speed term requires a quality floor, and the catalog quality prior
is unvalidated.

CLI commands return `0` only when the requested operation succeeds. A failed
plan, unavailable gateway/backend, failed benchmark, missing plan, or non-zero
foreground server exit returns `1`. `status` and `down` remain idempotent:
querying status or stopping an already-stopped runtime is a successful
operation.

## GPU detection and honest VRAM reporting

nmesh first tries the specialized NVIDIA and ROCm detectors: NVML or
`nvidia-smi` for NVIDIA, and `rocm-smi` for AMD. The generic detector runs only
when all of those specialized paths return no GPUs. On Windows it reads
`Win32_VideoController` and the display-adapter registry; on Linux it reads
DRM device vendor IDs and AMD's `mem_info_vram_total`. Apple Silicon keeps its
existing unified-memory path rather than going through generic discrete-GPU
detection.

`nmesh doctor` shows each detected adapter and includes the `vram_source`
provenance beside its VRAM amount (`nvml`, `smi`, `registry`, `sysfs`, or
`unknown`). An Intel integrated GPU can therefore be visible in `doctor` while
the plan remains `T0_CPU`: shared system memory is not counted as dedicated
VRAM, and the tier threshold is intentionally not inflated. Such an adapter
gets a warning explaining that its dedicated VRAM is below the placement
threshold.

Registry- and sysfs-derived adapters do not provide measured free VRAM.
nmesh conservatively reports `free = total` for those adapters; consequently,
`--budget free` is an upper bound rather than a measurement for them (the
normal display reserve is still subtracted). NVML and SMI values are the
sources that provide runtime free-memory measurements.

## Backend GPU capability checks

GPU-layer flags in a backend's help output are not proof that the backend can
execute on a GPU. In particular, the CPU-only llama.cpp build b10734 advertises
`-ngl` in `--help` but reports:

```text
Available devices:
  (none)
```

nmesh therefore asks llama.cpp to enumerate its devices. A successful empty
result, `()`, forces CPU placement and honest RAM accounting instead of
emitting GPU-layer arguments. An unknown result, `None` (for example when the
probe fails), preserves the previous behavior rather than making an
unsupported CPU fallback assumption. To use a detected GPU with llama.cpp,
install or build a backend with a Vulkan, CUDA, HIP, or SYCL GPU backend.
