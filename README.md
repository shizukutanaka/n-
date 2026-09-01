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
`llama-server` plus GGUF model. Set `NMESH_LLAMA_SERVER` and
`NMESH_E2E_MODEL` when the executable or model is not discoverable
automatically. This is intentionally standalone rather than a pytest test
because it requires a real model and can take several minutes.

Exit codes:

* `0` — every real-backend step passed and all selected ports were released.
* `1` — a required step failed.
* `77` — the required real backend or local GGUF model is unavailable.

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
