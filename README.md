# Multilingual output and model language preference

Set `NMESH_LANG=ja` (or `en`) to localize nmesh's runtime, planner, and CLI
output. Argparse help text remains English and is intentionally outside this
translation layer.

Use `nmesh plan --lang ja,en` or `nmesh up --lang ja,en` to softly prioritize
models that claim those languages. This is not a hard filter: nmesh still
selects a runnable model when no candidate claims every requested language.
Catalog language lists are publisher/vendor claims, not measured benchmarks.

`nmesh plan --roles chat,embed` and `nmesh up --roles chat,embed` make the
requested roles strict. Without `--roles`, a role with no local candidate is
reported as a warning and the remaining runnable services can still start.
# nmesh

## Quickstart（3 コマンド）

```text
pip install -e ".[gateway]"
nmesh doctor            # ハードウェアと使えるバックエンドを表示
nmesh up --detach       # 機械に合うモデルを選んで起動（llama.cpp が無ければ自分で取得）
# → http://127.0.0.1:18000/v1 が OpenAI 互換で使えます
```

nmesh は、ローカル LLM のためのハードウェア自動検出、モデル選択、メモリ見積もり、
実行管理ツールです。ノート PC から複数 GPU のワークステーションまで構成を自動で計画し、
OpenAI 互換 API を提供します。

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

### Managed llama.cpp engine

If no llama.cpp, Ollama, or LM Studio installation is available, a single
`nmesh up` can acquire a suitable managed llama.cpp server and the planned
model itself:

```text
nmesh up
```

Use `nmesh engine install` when you want to install the engine explicitly.
Passing `--no-download` opts out of automatic acquisition. In that mode:

```text
nmesh up --no-download
```

leaves the plan non-runnable and prints the command or package hint needed to
install the missing backend.

Engine archives are selected for the local operating system and accelerator,
stored under `$NMESH_HOME/engines/llamacpp`, and the active build is used by
hardware detection. Use `nmesh engine list --available` to inspect build tags,
`nmesh engine use <tag>` to switch installed builds, and
`nmesh engine remove <tag>` to delete one. GGUF weights remain under
`$NMESH_HOME/models`; inspect or remove them with `nmesh models local` and
`nmesh models rm`.

The llama.cpp release assets used here do not publish checksum files. The
manifest SHA-256 is therefore the hash nmesh observed while downloading the
archive. It detects later local corruption, but it does **not** verify
publisher provenance.

### KV-cache precision measurement

Speculative decoding is opt-in and evidence-gated. Measure n-gram or draft
decoding against the active plan's target service with:

```text
nmesh spec measure --kind ngram --repeats 2
nmesh spec measure --kind draft --draft C:\path\to\draft.gguf --repeats 2
nmesh spec show
```

The planner emits speculation flags only when the stored record matches the
target artifact, draft identity, and llama.cpp engine, and its decision is
`allow`. Use `nmesh plan --spec ngram` or `--spec draft` to request a
configuration; missing or losing evidence leaves speculation disabled.

KV-cache precision is part of benchmark identity. On one Windows CPU x64
machine, one llama.cpp build, one model, and one context, the controlled
measurement used engine `b10831` (`0.4.0-dev`, commit `8fe90e1fb`), Qwen2.5
1.5B-Instruct Q4_K_M (SHA-256
`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`),
`-c 32768 --parallel 1 --threads 8`, and an 8,309-token prompt with prompt
caching defeated with both `--no-cache-prompt` and request
`"cache_prompt": false`. Each arm used three repetitions and reports the
median:

| metric | f16 KV, no flash | q8_0 KV, no flash | f16 KV, `--flash-attn on` | q8_0 KV, `--flash-attn on` |
| --- | ---: | ---: | ---: | ---: |
| prefill median | 273.60 tok/s | 73.04 tok/s | 255.12 tok/s | 71.56 tok/s |
| decode median | 24.33 tok/s | 19.23 tok/s | 27.77 tok/s | 20.13 tok/s |

The load-time private bytes were 1,792,454,656 B for f16 KV and
1,348,911,104 B for q8_0 KV.

The private-byte difference was 444,104,704 B, versus 469,762,048 B in nmesh
accounting. Supplying `--flash-attn on` did not recover the prefill loss:
the corresponding medians were 255.12 tok/s for f16 KV and 71.56 tok/s for
q8_0 KV. The hard-130 quality comparison was 100/130 versus 98/130, with
discordant cells 4/2 and exact McNemar `p=0.6875`; this does not establish
equivalence, only that no difference was detectable at this sample size. No
stderr line in any arm mentioned flash attention, cache type, or an attention
fallback.

The default remains `f16`, and KV precision is not part of the automatic
fallback ladder. On this build, the measured memory saving costs prefill
speed.

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

## Runtime logs

Backend output is captured per service in
`$NMESH_HOME/logs/<service>.log`. Each log keeps one rotated generation at
`<service>.log.1`; set `NMESH_LOG_MAX_BYTES` to change the size cap. Set
`NMESH_BACKEND_LOG=0` to keep backend stdout and stderr inherited by the
launching process. Use `nmesh logs` to list services or inspect a log tail.
Resident gateways also expose `GET /logs/{service}`; when `NMESH_API_KEY` is
configured, this endpoint is protected by that key.

## Idle keep-alive

Set `NMESH_KEEP_ALIVE` to an idle timeout in seconds. The default `0` disables
idle unloading. Processes nmesh started in this `NMESH_HOME` remain unloadable
even when a different nmesh process, such as a detached gateway, re-attaches to
them. Shared Ollama daemons and servers nmesh only found listening on a port
are never stopped. Ollama's own model retention is not managed by nmesh. An
idle service is revived before the next request reaches it.

Use `nmesh unload` to unload planned services manually. The gateway also exposes
`POST /admin/unload`, `POST /admin/unload/{service}`, and
`GET /admin/running`.

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
The benchmark uses a nominal 512-token prefill parameter that tokenises to
roughly 336 real prompt tokens. Served samples deeper than 1024 real prompt
tokens remain recorded but are not used as planning evidence.
`nmesh bench` measures decode speed only; embedding services have no decode
path and must not be selected as the benchmark service.
Embedding-only services have no decode throughput, so the planner neither gates
nor ranks them on decode speed; `nmesh plan` displays `—` instead of an estimate.
On this host, Ollama 0.33.2 silently caps bge-m3 embedding input at 2048 tokens
regardless of `num_ctx`, while llama.cpp served the full planned window. nmesh
now measures the served cap, clamps the planned embedding context to it, and
warns when the backend silently truncates longer inputs; these observations are
specific to this host, artifact, and Ollama version.
For a proven cap, the gateway confirms saturation for a single string (or a
one-element input list) and returns an OpenAI-shaped `context_length_exceeded`
error when the backend would silently truncate it. Multi-element input lists
remain a known gap because aggregate usage cannot identify which element was
truncated; saturated requests receive
`X-Nmesh-Embedding-Truncation: unverified` instead.
Served length is not the same as usable retrieval length. On the measured
llama.cpp b10831 `--embeddings --pooling cls -c 8192 -b 8192 -ub 8192`
environment with the bge-m3 Q8_0 artifact, the opt-in `--retrieval` ladder
measured:

```text
~166 tokens   rank1 4/4
~320 tokens   rank1 4/4
~603 tokens   rank1 4/4
~1177 tokens  rank1 4/4
~2346 tokens  rank1 7/8
~2921 tokens  rank1 5/8
~3512 tokens  rank1 2/8
~4370 tokens  rank1 1/8
```

This is one host, one artifact, and one backend/build measurement, not a
general law about bge-m3. The chunk-remedy confirmation measured:

```text
whole document        rank1 1/8   ranks=[4,4,2,4,5,6,7,1]
200-word chunks       rank1 8/8
200-word, 20% overlap rank1 8/8
400-word chunks       rank1 8/8
400-word, 20% overlap rank1 8/8
800-word chunks       rank1 8/8
800-word chunks, normalized mean rank1 8/8
400-word chunks, normalized mean rank1 8/8
200-word chunks, normalized mean rank1 8/8
400-word chunks, raw mean         rank1 8/8
```

Overlap made no difference at these measured sizes. The recommended chunk
size is the ladder's usable rung (800 words, approximately 1177 served
tokens in this measurement). The pooled confirmation on the same vectors
measured 800-, 400-, and 200-word normalized means at rank1 8/8, and a
400-word raw mean also at 8/8 because this backend returns unit vectors.
Pooled cosine scores were lower than max-over-chunks scores due to dilution;
that dilution was not shown to be harmless in general. The planner warns
rather than clamps when single-vector retrieval degrades, and the gateway
autochunk remedy is opt-in with `NMESH_EMBED_AUTOCHUNK=1`. It applies only
where pooled recovery is proven for the matching model, quantization,
backend, and evidence identity, and uses the measured chunk size. Such
responses include `X-Nmesh-Embedding-Chunked` and
`X-Nmesh-Embedding-Chunk-Words`, plus the configured character bound in
`X-Nmesh-Embedding-Chunk-Chars`. If the arm does not recover it, chunking is
not presented as a verified remedy. This evidence remains scoped to one host,
one artifact, and one backend build.
For non-whitespace-segmented input such as Japanese, Chinese, or Thai, the
gateway splits by characters when the input exceeds the measured usable token
count in characters. Because a token covers at least one character, that count
is a conservative character bound; over-splitting is measured safe (200-word
chunks recovered 8/8), while under-splitting is measured unsafe. On one host,
one artifact, one backend build, and one synthetic corpus, Japanese measured
`6000 chars ≈ 4540 served tokens`: single-vector retrieval was 1/8, while
character-split pooled retrieval was 8/8. English word splitting remains
primary because it scored 8/8 versus 7/8 for hard character splitting;
boundary-aware splitting was measured and rejected at pooled 6/8. The
character path is deliberately language-neutral and does not add a tokenizer
or probe request.
`nmesh bench --service <embed-service> --retrieval` is opt-in because it takes
432 ladder requests (6 rungs × 8 seeds × 8 documents plus a query each) plus
about 72 chunk-recovery requests, and its wall time scales with this host's
measured encode
throughput — the estimate printed before the ladder comes from a calibration
request on the workload's own text, and on a CPU-only host measured at
387 tok/s the run took 33 minutes.
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

`nmesh eval --depth N` records requested and served prompt depth separately and
keeps deterministic single-needle context probes outside the suite score and
digest. Planner depth evidence comes only from these controlled probes: the
hard suite is answerable from the question without reading the context, so a
padded suite run cannot license a context length. The planner warns when
advertised context exceeds quality evidence but does not exclude the candidate.
The `probe_rules` value in `nmesh --version` is a fixed depth-0, `core`-seed
probe-rule identity marker; it is not expected to match the per-depth
`probe_digest` values stored in `context.json`.
The timeout budget includes the requested depth; this matters because the
bench's `512` is a nominal prefill parameter that tokenises to roughly 336 real
prompt tokens, not a real-token depth. The shipped probe families are literal,
latent, multi, and update. Context probes run identical-needle native-depth
controls before and after the deep run; a deep task counts as evidence only
when its own control task passed in both control runs, and an attributable
failure is reported as a measured-broken context warning without clamping or
excluding the candidate.
Multi-needle ordered retrieval held to about 15k tokens; two-hop retrieval had
4/4 then 2/4 shallow controls across instances and is not shipped, while count
failed at about 200 tokens and is also not shipped. Probe grading considers
only standalone six-character codes, so an eight-character case id cannot be
mistaken for an answer. Attribution is per task pair rather than rejecting a
whole family after one control miss. The controls are a cheap safeguard
against host incidents, not a new measurement source: 12 product repetitions
produced 288 unchanged grades. Changing a probe rule changes its digest, so
older `context.json` records become stale and must be re-measured.

`nmesh evidence` inventories saved benchmark, evaluation, and depth records,
including the reason codes that make a record unusable and the command needed to
remeasure it. Benchmark reasons include `harness_mismatch`, `unstable`,
`epoch_degraded`, and `unconfirmed`; evaluation reasons include `suite_unknown`,
`grader_digest_mismatch`, `unscorable`, `transport_errors`, and `depth_scoped`;
depth reasons include `probe_digest_mismatch`, `control_failed`, and
`depth_lost`; `superseded` marks an older valid record replaced by a newer
record for the same configuration. Depth values show served depth and each
probe family's passed/total and control counts; `verified` or `lost` is shown
only when the existing depth-evidence selection establishes that status. Rows
that need new evidence point to `nmesh bench`,
`nmesh eval --suite <suite>`, or `nmesh eval --depth <requested_depth>`.

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

#### Deleting saturated tasks cannot buy resolution

The obvious next step looked like deleting the two families that carried no
discriminating power. It is arithmetically void, and the measurement says so:
exact McNemar depends only on the discordant counts, so removing the 24
`instruction`/`multilingual` tasks that both models passed leaves 80 tasks,
the same `24`-versus-`5` discordance, and the same `p=0.000546`. Only the
unpaired Fisher number moves (`0.003025` to `0.001860`) — because the rate gap
looks larger once the shared successes are gone, which is a reporting artifact
and not new evidence. A suite grows resolution only by adding tasks the two
configurations answer *differently*.

So the saturated families are kept and 26 harder tasks are added as an opt-in
third tier, `nmesh eval --suite hard` (130 tasks; `core` stays the default and
the 104-task `extended` digest is unchanged, so existing records stay valid).
All 26 were authored before measuring anything and all 26 shipped: selecting
tasks after seeing which ones separate the two models would tune the instrument
to the answer it is supposed to test. Measured in one run on the same two
artifacts, same binary, same machine:

| model | extended (104) | hard (130) |
| --- | --- | --- |
| Qwen2.5 1.5B Q4_K_M | `89/104` = `0.856` | `100/130` = `0.769` [`0.690`, `0.833`] |
| Qwen2.5 0.5B fp16 | `70/104` = `0.673` | `76/130` = `0.585` [`0.499`, `0.666`] |
| paired discordance | `24`/`5`, `p=0.000546` | `29`/`5`, `p=0.000039` |

The 26 new tasks contribute `5` discordant pairs (all favouring the 1.5B),
`6` both-pass and **`15` both-fail**: they broke the saturation but 58% of them
are floored instead, which is the same zero-power condition seen from below.
`p` improved by an order of magnitude off 5 tasks, and the remaining 15 are the
next thing to fix, not a result.

#### Two of the 26 tasks were measuring the instrument, not the model

Reading the actual outputs of the 16 tasks both models failed found two
defective items rather than two hard items. `multilingual.kanji_number.17`
accepted only `十七`, so the 1.5B's `壹拾柒` — a correct kanji numeral — was
scored wrong; the item measured which kanji form a model happens to pick.
`multilingual.lang_lock.seven` demanded `七` while its prompt only forbade
Latin letters, so both models' `7` satisfied the stated constraint and failed
anyway. Both were repaired (wider accepted set; prompt that forbids digits
explicitly), which is why the 1.5B moved `99` → `100`.

That repair exposed a hole in suite identity: `suite_digest` hashed only
`(id, prompt, max_tokens)` plus a suite-wide `GRADER_VERSION`, so widening one
verifier changes pass/fail without changing the digest, while bumping
`GRADER_VERSION` invalidates every stored record including the 104-task ones
the evidence override depends on. Tasks now carry a `rule` string that is
hashed when set, so a repaired verifier invalidates only the suites containing
it; the repaired task rules in this PR change the affected suite digests, so
the earlier `extended` and `hard` records are superseded. The repaired
digests are `extended` `v2:86e41db848586057` and `hard`
`v2:22237baff8c47e35`.

The remaining floored tasks are also not one thing. On
`instruction.initials.quick_amber_fox` the 1.5B answered `Q A F` — the right
letters in a forbidden form — while the 0.5B answered `QWEN`; a single bit of
pass/fail calls those the same failure. Tasks may now carry an optional
`value_check` that grades the value while ignoring the required output form.
It never affects pass/fail, the digest, or anything stored; `nmesh eval`
reports how many failures were value-correct and form-wrong. On this run that
was `3` of the 1.5B's `5` value-checked failures (`Q A F`, `Paris`, `7`) and
`3` of the 0.5B's `10`. Those failures measure output discipline, not
capability, and reporting them as capability is what PR #38 had already
corrected one level up. These splits were produced by value rules superseded
by this PR. On q2_k, `multilingual.katakana.model` answered `モード` but was
counted value-correct through the Latin `model` copied from the prompt;
`multilingual.kanji_number.17` and `.30` answered the prompt's own Arabic
`17` and `30`.

Both tests now report the power they actually realised. `nmesh eval` prints,
per compared configuration, how many of the shared tasks disagreed, in which
direction, and which families contributed nothing; the planner's underpowered
note on paired data no longer quotes a Fisher-derived "minimum resolvable
difference" from `n`, because on paired outcomes that number is meaningless:
104 shared tasks with one discordant pair give `p=1.0` regardless of `n`. It
states the discordant counts and that exact McNemar cannot reach `p<0.05` below
`6` disagreeing tasks at any suite size.

The eight-rung outputs show why this distinction matters. Under the repaired
graders, the `q4_k_m` → `q4_0` step has 10 lost tasks: 9 wrong answers and 1
correct value in the wrong output form. The `q3_k_m` → `q2_k` step has 36 lost
tasks: 28 wrong answers and 8 correct values in the wrong output form. On the
same rungs, the superseded value rules reported q2_k as 50 wrong / 15 form and
q3_k_m as 29 wrong / 6 form. The repair moved three q2_k tasks
(`multilingual.katakana.model`, `multilingual.kanji_number.17`, and
`.30`) and one q3_k_m task (`multilingual.kanji_number.17`) out of
“value-correct, form-wrong” and into wrong answers. This split is one model on
one machine with one binary and one suite; it does not generalize to other
quantization ladders or hardware.

#### Suite grader audit

Before this PR, 2 of 130 hard-suite pass-rate graders and 23 of 130
value graders accepted the prompt verbatim. The invariant test now forbids
every shipped task grader from accepting the prompt, a refusal, or the prompt
next to a wrong value. `extraction.max.4` was structural only: real answers
were `3000`.

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

Before the candidate path was added, all 28 verified `catalog_gap` drafts were
rejected by the catalog loader. A hand-completed candidate with
`quality: null` was rejected too: `float(None)` raised `TypeError` and the
entry was silently dropped. There was no `--model` path, so measuring a
candidate before planning required inventing quality.

The same 28 candidates included 20 with no primary GGUF. The eight smallest
complete weight sets measured 0.08 / 0.64 / 6.19 / 14.25 / 17.65 / 72.55 /
93.09 / 397.26 GB. Per-file minima are invalid because repositories contain
`mmproj`, MTP/draft heads, imatrix files, vocabulary files, and multi-part
shards; `dzannotti/Qwen3.8-Flash-Next-MTP-GGUF` and
`cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF` are auxiliary-only examples.
Watch groups primary GGUF shards before reporting feasibility.

Unmeasured entries are never ranked automatically. Explicit `--model` selection
is the path to planning one, and `nmesh eval` is the only path to a legitimate
quality value. Quality is never invented.

Noise IDs such as `docs/hub`, `papers/2504.13181`, `datasets/leemeng`, and
`blog/nvidia` are filtered by the Hugging Face 401/404 gate. GGUF mirror
repositories commonly return 404 for `config.json`, including
`Qwen/Qwen3-14B-GGUF` and `ggml-org/...-GGUF`; architecture numbers then come
from the base repository recorded in `config_repo`. GitHub's API returned 403
from this box, so there is deliberately no releases source. X is unavailable
without `NMESH_X_BEARER_TOKEN`. `--offline` accepts saved source items for
reproducible extraction and verification, and bounded state prevents repeated
findings from growing without limit. No finding is auto-applied to the product.

### Answerless truncation is not a failure

A suite task budget (16 to 48 tokens) is an answer budget. A model that emits
separate reasoning output spends that budget before the answer, returns
`finish_reason=length` with empty `content`, and grading the empty string
measures the budget. Measured on this box with
`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL` on llama.cpp: **0/104** at the suite budget
with every task returning empty content, and **104/104** at
`--reasoning-allowance 464`. Such responses are now counted as `unscorable`
instead of failed, unscorable runs are never used as planning evidence or for
task-level divergence, and the allowance is part of the record identity
(`model|quant|backend|suite|digest|a464`), so a wider budget never overwrites a
narrower one.

The 104/104 also fixes the suite's upper limit: with a ceiling of 1.000 the
extended suite cannot rank two models that both saturate it, exactly as
`instruction`/`multilingual` saturated for the smaller Qwen2.5 pair.

### Measurement can outrank the catalog prior

Until now `nmesh eval` could prove the catalog `quality` prior wrong and the
planner would only say so: the ranking still came from the prior, and a
regression test asserted that the selection stayed the same. Measured pass
rates now decide instead, under a gate that never mixes the two scales:

- the planned candidate and the alternative must each have a measured pass rate
  for **their own** planned `(quant, backend)` under the same grader digest and
  reasoning allowance;
- the alternative's rate must be higher, and the exact test must reach
  `p < 0.05` — paired McNemar over the shared task ids when task-level results
  exist for both, otherwise unpaired Fisher;
- a measured candidate never outranks an unmeasured one, and aggregate-only
  records (a bare rate with no task counts) never participate.

Nothing else changed: with no eval cache the plans are bit-identical, and
`--ignore-eval-evidence` on `plan`/`up` restores prior-based ranking while
keeping the contradiction warning.

#### Quantization evidence

Pass rates are keyed by `(model, quant, backend, suite, digest, allowance,
cache_prompt)`, so measurements distinguish quantizations of the same model and
never mix the two prompt-cache conditions described below. The completed
eight-rung measurement for `qwen2.5-1.5b-instruct` used the repaired hard suite
and the same llama.cpp configuration throughout. The pre-repair q3_k_m count
included one task that passed on a wrong answer: the shipped
`extraction.email` prompt elicited `ops@nmesh-ops@example.com`, which the old
grader accepted because the expected address was merely a substring. With the
repaired prompt, the same model answered `The email address to contact is:
ops at nmesh-ops@example.com`, where the address is a standalone run and the
task passes legitimately. The rung total remains 95, so the false positive was
masked inside the total. The repaired suite digests are `extended`
`v2:86e41db848586057` and `hard` `v2:22237baff8c47e35`; stored pre-repair
records remain superseded. Every rung was measured twice with prompt-cache
reuse disabled; both repeats agreed on every one of the 130 task outcomes, with
no unscorable answers and no transport failures:

Pass rates are compared only when the suite, grader digest, reasoning allowance,
and prompt-cache condition match; the measured q4_k_m/q4_0 pair moved from
exact `p=0.0352` with cache reuse to `p=0.1796` without it.

| quant | `QUANT_PENALTY` | file GiB | passed/130 | passed/130 with reuse (superseded graders) |
| --- | ---: | ---: | ---: | ---: |
| f16 | 0.0 | 3.32 | 100 | 100 |
| q8_0 | 0.5 | 1.76 | 99 | 99 |
| q6_k | 1.0 | 1.36 | 100 | 100 |
| q5_k_m | 2.0 | 1.20 | 101 | 101 |
| q4_k_m | 3.5 | 1.04 | 98 | 100 |
| q4_0 | 5.0 | 0.99 | 92 | 91 |
| q3_k_m | 9.0 | 0.86 | 95 | 93 |
| q2_k | 16.0 | 0.70 | 65 | 63 |

Adjacent exact McNemar results in penalty order (high-only / low-only / p) are:
f16→q8_0 `1/0`, `p=1.0`; q8_0→q6_k `0/1`, `p=1.0`;
q6_k→q5_k_m `3/4`, `p=1.0`; q5_k_m→q4_k_m `5/2`, `p=0.4531`;
q4_k_m→q4_0 `10/4`, `p=0.1796`; q4_0→q3_k_m `8/11`,
`p=0.6476`; and q3_k_m→q2_k `36/6`, `p=2.83e-06`. Only
q3_k_m → q2_k is significant; every other adjacent step is indistinguishable
under this suite.

Under the repaired graders, the per-rung failure kinds (wrong answer /
correct value in the wrong form) are: f16 `25/5`, q8_0 `26/5`, q6_k `26/4`,
q5_k_m `23/6`, q4_k_m `25/7`, q4_0 `32/6`, q3_k_m `30/5`, and q2_k `53/12`.

On this model, machine, binary, and suite, the ladder collapses into two
indistinguishable bands: `{f16, q8_0, q6_k, q5_k_m, q4_k_m, q4_0, q3_k_m}` at
92–101/130 and `{q2_k}` at 65/130. The 9.0 penalty points between f16 and
q3_k_m buy no measurable quality here while costing 2.46 GiB of weights. This is
one model on one machine under one suite, not a general claim about
quantization. The `QUANT_PENALTY` values were not changed: one model's ladder
cannot set a constant that applies to every model.

The right-hand column is what the same suite reported before eval controlled
prompt-cache reuse, and it is not the same ladder. With reuse the q4_k_m/q4_0
step measured `0.035156`, which is how this document previously reported
q4_k_m → q4_0 as a measurable step; disabling reuse moves the same pair to
`0.179565`. Reuse inflated the gap in both re-measured pairs — q4_k_m/q3_k_m
went from 7 tasks (`0.143463`) to 3 tasks (`0.607239`) — because llama.cpp
reuses the slot's KV prefix, so a measurement depends on what the server
processed before it. Each arm is internally repeatable (two to three repeats per
arm agreed on every task, whether the server was fresh or had 130 to 260 tasks
of history in front of it), which is why the earlier single-run numbers looked
solid: the contamination is systematic, not noisy, and no spread check can see
it. `nmesh eval` now sends `cache_prompt: false` to llama.cpp services and
records the condition in the measurement key (`|c0`), so records made under
reuse are kept but never compared against cache-clean ones. Records carrying no
condition — everything measured before this change, including the right-hand
column — are legacy and are never compared against `|c0` records; on llama.cpp
they have to be re-measured. Sampling fields stay out: adding `top_k: 1` and
`seed: 0` on top of greedy decoding also moved single tasks (q4_k_m 100 → 101,
q3_k_m 93 → 94), which says those tasks sit at the decision boundary, not that
the fields make the measurement more reproducible.

Measurement-to-plan identity is case-insensitive on model id, quant, and
backend because record quants come from artifact names (`Q4_K_M`,
`UD-Q4_K_XL`) while the catalog spells them lowercase. On this box the
existing 89/104 record was keyed `Q4_K_M` and was silently treated as a
different configuration before this change.

With llama.cpp visible to the probe (see below), the gate has now fired on real
measurements: with a local `models.yaml` that rates Qwen2.5 0.5B above 1.5B
(100 versus 50), the prior plans 0.5B `f16` and the measured pair (1.5B
`q4_k_m` 89/104 against 0.5B `f16` 70/104, paired exact `p=0.0005` over 104
tasks) plans 1.5B instead; `--ignore-eval-evidence` restores the 0.5B plan and
the contradiction warning. With the bundled priors the two orders agree, so no
reversal is visible there.

#### Benchmark prompt-cache exposure

`nmesh bench` sends no `cache_prompt` either, so the same question was measured
on this box rather than inferred from the eval result. On llama.cpp
(qwen2.5-1.5b-instruct q4_k_m, 512-token prefill, three runs per arm) reuse
stops at the chat template: `cache_n` was 0, 24, 25 tokens out of ~337, the
processed count stayed at 310–337, and the numbers do not move when reuse is
disabled (prefill median 423.8 versus 414.1 tok/s, decode 44.1 versus 43.2
tok/s — both inside the 6.4% spread this host shows when healthy). Running the
same arm after a 2048-token unrelated prompt changed nothing (`cache_n` 24,
24, 24). So bench was not contaminated, and the reason is the prompt shape:
the prompt is `f"{nonce} " + filler`, so a fresh uuid leads every measurement.

That protection is load-bearing rather than incidental. Moving the same nonce
behind the filler — a shape any refactor could produce — let llama.cpp reuse
304–305 of ~338 prompt tokens, dropped the processed count to 34, and inflated
the ttft-derived rate from ~370 to 1605–1671 tok/s (4.3x). `nmesh bench` uses
that ttft-derived rate whenever the upstream reports no llama.cpp `timings`,
and Ollama is exactly that case: its OpenAI-compatible stream carries
`choices`, `created`, `id`, `model`, `object`, `system_fingerprint`, `usage`
and nothing else — no `timings`, no `prompt_tokens_details.cached_tokens`.
Measured against Ollama 0.33.2 (qwen2.5:0.5b-instruct), a leading nonce reports
385–391 tok/s while a trailing nonce reports 1500–1556 tok/s (4.0x), and no
field in the response says a single token was reused. On that backend the
prompt shape is the only available defense, so it is now stated in the code and
locked by a test.

Two changes follow from the measurements. Bench sends `cache_prompt: false` to
llama.cpp services so the measurement stops depending on the prompt shape
there; the measured cost is nothing, which is why bench records are not split
by cache condition the way eval records are (unlike eval, no measured value
moves, so splitting identity would invalidate stored evidence and buy nothing).
And where the upstream does report cached tokens, the ttft fallback now rates
only the tokens the server actually processed instead of the whole prompt, and
reports the source as `cached` when fewer than 16 tokens were processed.

Decode throughput is computed over the decode steps, `predicted_n - 1`, because
llama.cpp's `predicted_ms` excludes the prefill token. The old numerator recorded
1,000,000.0 tok/s at one token, 93.14 at two, and 62.34 at four, although steady
state was about 47 tok/s; the stored sessions reached
`[62.34, 93.14, 1000000.0, 47.75, 46.99]` and the median was inflated by 31%.
A bracketed ladder against a 128-token reference measured arm/reference ratios of
0.9761 (n=2), 0.9988 (n=4), 0.9994 (n=16), 1.0117 (n=32), 1.0093 (n=64), and
0.9911 (n=128). This is why `benchmark_key()` does not include generation
length or prompt depth: the corrected rate is length-independent, and bench
prompt depth is fixed rather than user-selected. `--tokens` is an upper bound;
bench records the served decode length and warns when EOS stops before the
requested length, while a one-token response is not measurable.
Saved measurements from an older benchmark harness are not used as planner
evidence; run `nmesh bench` again. On the measured machine, a legacy 0.5B
record at 102.5 tok/s had incorrectly appeared as `"estimated": false`.

#### GGUF artifact identity

The former GGUF resolver selected by filename substring rather than by a
canonical artifact label. On `bartowski/gemma-2-9b-it-GGUF`, a planned
`q8_0` could download `gemma-2-9b-it-Q3_K_L-Q8.gguf` (5,132,452,800 bytes)
and report it as `q8_0`; a planned `f16` could download
`gemma-2-9b-it-Q4_K_M-fp16.gguf` (6,843,425,696 bytes) and report it as
`f16`. The same resolver selected `gemma-2-9b-it-Q6_K-Q8.gguf` for planned
`q6_k` instead of the plain `gemma-2-9b-it-Q6_K.gguf`, and selected
`Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf` for planned `q4_0` in
`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`. The latter is an unverified
CPU-repacked artifact.

GGUF names are now parsed into canonical labels across the published
vocabulary, including IQ, K-family, tensor, and full-precision variants.
Mixed labels remain distinct from plain labels, and the CPU-repack labels
`q4_0_4_4`, `q4_0_4_8`, and `q4_0_8_8` are excluded. Artifact metadata
reports the resolved label and file bytes rather than the planned label.

The resolver orders safe fallbacks using llama.cpp's nominal bpw table; that
table is not measured nmesh data. Actual bytes can differ substantially:
the measured actual/estimated ratios were 1.641 for a 0.5B `q4_k_m` file,
2.007 for its `q2_k` file, and 1.271 for `bge-m3` `q4_k_m`; 7B and larger
models were within approximately ±5%. Planning still budgets from the
nominal parameter-count × bpw estimate, so nmesh warns when the resolved
artifact differs from that estimate by more than 10%.

Resolver recognition does not imply planner support: labels that nmesh can
identify but cannot plan remain outside `BPW`, and `nmesh watch` continues to
report them as unknown quantizations.

The planner now structurally accounts for vocabulary and output-head tensors
for the catalog models whose GGUF headers provide that metadata. Its nominal
weight estimate keeps those tensors at a 5.5 bpw floor; measured artifact byte
totals cached by repository and resolved label override the estimate when
available, and acquisition records those totals for later plans. The estimate
remains an estimate: the measured 0.5B `q4_k_m` and `q2_k` ratios above were
1.641 and 2.007, while the structural estimate brings the corresponding
planner ratios into the observed 0.79–1.30 envelope.

### An installed backend the probe cannot see

Backend availability was decided by `shutil.which` alone, so a llama.cpp build
that is not on `PATH` did not exist as far as planning was concerned: on this
box `llama-server` lives in `C:\Users\Administrator\llamacpp\`, every eval and
bench number in this README was produced by it, and the planner still reported
llama.cpp as missing and planned the `ollama` backend for chat. That silently
costs more than a hint: memory falls back to the Ollama quant estimate,
`SLOT_CAPS["ollama"]` is 1, and every llama.cpp measurement is discarded as
belonging to a different configuration - which is why the override above could
not be reached with real data.

Binaries are now resolved as `NMESH_LLAMACPP_BIN` / `NMESH_OLLAMA_BIN` /
`NMESH_VLLM_BIN`, then `PATH`, then `$NMESH_HOME/bin` (with `.exe` on Windows),
and the resolved path becomes `argv[0]` of the launch command, so a plan runs
the binary the probe actually inspected. An override pointing at something that
is not executable leaves the backend unavailable and says so instead of quietly
falling back to `PATH`. No vendor install directories are guessed.

The recorded version is now the first `--version` line that mentions a version
rather than the first line of output. `ollama --version` on this box prints
`Warning: could not connect to a running Ollama instance` before the version,
and nmesh had been storing that warning as Ollama's version string.

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
