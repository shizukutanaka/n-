"""Small, failure-tolerant translation layer for nmesh output."""

from __future__ import annotations

import locale
import os
import re

SUPPORTED = ("en", "ja")

MESSAGES = {
    "en": {
        "warn.embeddings_unsupported": "{model}: llama.cpp embedding flags are unsupported; /v1/embeddings may be unavailable.",
        "warn.embeddings_pooling_unknown": "{model}: pooling metadata is unknown to nmesh or unsupported by this llama.cpp build; publisher/vendor may use pooling type 'none', which the OpenAI-compatible embedding endpoint rejects.",
        "warn.rerank_unsupported": "{model}: this llama.cpp build lacks --reranking; the rerank service will not be able to answer /v1/rerank.",
        "warn.embeddings_batch_limit": "{model}: embedding inputs above 512 tokens may be rejected because the physical batch-size flags are unavailable (planned context {context}).",
        "warn.embeddings_backend_unsupported": "{model}: mlx_lm.server does not provide an embedding endpoint.",
        "warn.embeddings_backend_unverified": "{model}: vLLM embedding and pooling behavior is not verified by nmesh; nmesh does not guess or inject settings.",
        "install.ollama": "Install Ollama: https://ollama.com/download",
        "install.llamacpp": "Install llama.cpp with nmesh engine install, or use winget / brew / build from source",
        "engine.installed": "Installed engines",
        "engine.active": "Active engine",
        "engine.available": "Available builds",
        "engine.tag": "Tag",
        "engine.variant": "Variant",
        "engine.active_column": "Active",
        "engine.path": "Path",
        "engine.install": "Installed {tag} ({variant})",
        "info.engine_autoinstall": "No llama.cpp backend found; installing the managed engine {tag} ({variant}).",
        "engine.use": "Using {tag} ({variant})",
        "engine.remove": "Removed {tag}",
        "engine.active_cleared": "Active engine cleared",
        "models.removed": "Removed {path}",
        "models.planned_refusal": "Refusing to remove a model referenced by the plan: {path}; use --force",
        "models.local": "nmesh local models",
        "models.scan": "nmesh model inventory",
        "models.scan_store": "{store}: {files} file(s), {gib:.2f} GiB",
        "models.scan_total": "Machine total: {files} file(s), {gib:.2f} GiB",
        "models.scan_reclaimable": "Reclaimable duplicate storage: {gib:.2f} GiB",
        "models.scan_variants": "Variant groups: {count}",
        "err.unload_not_running": "{service} was not unloaded: it is not running.",
        "err.unload_idle": "{service} was not unloaded: it is already idle.",
        "err.unload_shared": "{service} was not unloaded: it is a shared daemon; nmesh does not stop it.",
        "err.unload_external": "{service} was not unloaded: it is an external server nmesh only found listening to; nmesh does not stop it.",
        "err.unload_not_owned": "{service} was not unloaded: nmesh will not stop a running process it does not own; use nmesh down.",
        "err.unload_unknown": "{service} was not unloaded.",
        "install.vllm": "Install vLLM: pip install vllm",
        "install.mlx": "Install MLX-LM: pip install mlx-lm",
        "warn.parallel_unsupported": "llamacpp: --parallel unsupported; using one slot and plain context",
        "warn.gpu_layers_unsupported": "llamacpp: GPU-layer flags unsupported; using CPU placement",
        "warn.tensor_split_unsupported": "llamacpp: --tensor-split unsupported; omitting tensor split",
        "warn.tensor_split_proportional": "{service}: splitting the model across GPUs in proportion {split} (--tensor-split) matched to each GPU's usable VRAM.",
        "warn.kv_quant_unsupported": "{service}: {backend} does not support requested KV precision {requested}; KV cache is accounted at f16.",
        "warn.kv_quant_speed_unmodeled": "{service}: planned throughput does not model KV cache type; {kv_quant} KV measured slower in prefill and decode on the reference build.",
        "warn.gguf_mixed_precision": "{service}: the GGUF resolved for {planned} is {label} ({filename}) — a mixed-precision artifact, not a plain {planned} quantization; measurements recorded under this quant label describe that artifact.",
        "warn.gguf_size_mismatch": "{service}: resolved GGUF is {actual} bytes versus {estimated} estimated bytes; the plan's fit decision used the parameter-count × nominal-bpw estimate, not the file.",
        "warn.gguf_corrupt": "{service}: cached GGUF {filename} is {actual} bytes but {expected} bytes were recorded at download — the artifact is corrupt and will be re-acquired.",
        "warn.quant_fallback_skipped": "{service}: quantization fallback was skipped because the running artifact label {quant} is outside the planned quantization ladder.",
        "warn.gpu_over_budget": "{service}: GPU placement needs {committed:.0f} bytes, over the {budget:.0f}-byte budget",
        "warn.backend_placement_estimate": "{service} uses {backend}, which manages its own placement; this RAM fallback is nmesh's estimate, not a controllable setting.",
        "warn.gpu_layers_cpu_fallback": "{service}: GPU layers did not fit on any card; it will run on the CPU.",
        "warn.selection_capacity_tradeoff": "{role}: chose {chosen_model} {chosen_quant} instead of {empty_model} {empty_quant} because {reserved_vram_gib:.1f} GiB of GPU and {reserved_ram_gib:.1f} GiB of RAM are already committed; this is a capacity-forced tradeoff.",
        "warn.layers_reduced": "{service}: reduced GPU layers to {layers}; {previous} layers did not fit in RAM",
        "warn.slots_clamped": "{service}: parallel_slots {requested} exceeds available memory; using {slots}",
        "warn.slots_unsupported": "{service}: {backend} cannot serve {requested} parallel slots here, so the plan uses 1.",
        "warn.engine_substituted": "{service}: the planned engine binary {old} is gone; launching with installed engine {tag} instead.",
        "warn.slots_tradeoff": "{service}: {slots} parallel slots raise aggregate throughput but lower the per-request rate; the {tps} tok/s shown is a single-stream figure (measured on one CPU machine: 46.2 tok/s alone, 36.84 with 4 in flight, 23.30 with 8, aggregate 125 and 148).",
        "warn.speed_saturated": "{role}: --prefer speed could not discriminate — {chosen} {chosen_quant} at {chosen_tps} tok/s was ranked above {other} {other_quant} at {other_tps} tok/s because the score's speed term saturates at {reference} tok/s, so the unvalidated quality prior decided.",
        "warn.no_source": "{model}: no Hugging Face or Ollama source is configured",
        "warn.no_candidate": "No runnable model found for role {role}",
        "warn.model_unknown": "Requested model {model} is not in the catalog.",
        "warn.nvidia_unavailable": "NVIDIA detection unavailable",
        "warn.parallel_clamped": "llama.cpp: --parallel unsupported; using one slot instead of {requested}",
        "warn.rocm_parse": "Unable to parse rocm-smi output",
        "warn.mlx_check": "Unable to check mlx_lm",
        "warn.system_probe": "System probe failed: {error}",
        "warn.backend_no_gpu": "llama.cpp binary reports no GPU backend; using CPU placement. Install a Vulkan, CUDA, HIP, or SYCL build to use the detected GPU",
        "warn.backend_binary_missing": "{backend}: the configured binary {path} is not executable; treating the backend as unavailable instead of falling back to PATH.",
        "warn.low_vram": "{gpu} has less than 2 GiB dedicated VRAM; it remains visible but is not used for GPU placement",
        "warn.download_budget": "Planned downloads total {total_gb:.1f} GiB, exceeding the {limit_gb:.1f} GiB download budget (policy.allow_download_gb).", 
        "warn.language_coverage": "{model} does not claim coverage for {languages}; language coverage is a publisher/vendor claim, not a measured benchmark.",
        "warn.simulated_profile": "Simulated profile: planning decisions only. Nothing is run, and this machine's throughput and embedding measurements are ignored because they describe this machine, not the profile; speeds shown are estimates.",
        "hint.language": "Locale is {language}; use --lang {language} to prioritize models claiming {language}.",
        "warn.free_admission": "Free-memory admission: planned usage {need_gpu:.2f} GiB VRAM / {need_cpu:.2f} GiB RAM exceeds available {vram:.2f} GiB VRAM / {ram:.2f} GiB RAM.",
        "warn.real_artifact_replanned": "{service}: real artifact bytes exceeded the estimate; re-planned against free memory before launch.",
        "warn.free_admission_fallback": "Free-memory admission failed; using the original plan. Reason: {error}",
        "warn.runtime_fallback": "Runtime fallback attempt {attempt}",
        "warn.admission_skipped": "Free-memory admission skipped: {error}",
        "warn.health_failed": "Service failed health check: {service}",
        "warn.worker_not_coresident": "The worker role was requested, but no strictly smaller generative model fits alongside the lead; delegation remains unavailable.",
        "warn.spec_unsupported": "{service}: speculation is unsupported by the detected llama.cpp flags; no speculation flags were emitted.",
        "warn.sleep_idle_unsupported": "{model}: this llama.cpp build lacks --sleep-idle-seconds; services will stay resident.",
        "warn.moe_cpu_offload": "{service}: llama.cpp will run {model} with all layers on the GPU while {layers} MoE layers keep their expert tensors on the CPU (--n-cpu-moe); the shown decode speed is a modeled estimate until measured.",
        "warn.moe_cpu_unsupported": "{model}: this llama.cpp build lacks --n-cpu-moe; the planned MoE expert offload was dropped.",
        "warn.vllm_sleep_unsupported": "{service}: vLLM {version} lacks sleep mode (needs >= 0.9); the engine will be restarted on each swap switch.",
        "warn.cache_reuse_unsupported": "{model}: this llama.cpp build lacks --cache-reuse; prompt cache will not be reused across requests.",
        "warn.context_shift_enabled": "{model}: context shift is enabled; once generation exceeds the context window the oldest tokens are silently dropped (prompts larger than the window are still refused).",
        "warn.context_shift_unsupported": "{model}: this llama.cpp build lacks --context-shift; oversized input will still be rejected.",
        "warn.up_flags_saved_plan": "{flags}: a saved plan is already in effect, so these options are ignored; run nmesh plan to apply them.",
        "warn.plan_stale": "Saved plan was generated by an older nmesh (launch flags have changed since). Re-run `nmesh plan` to pick up improvements.",
        "warn.spec_refused": "{service}: speculation evidence refused ({reason}); speedups: {speeds}.",
        "warn.spec_override": "{service}: speculation evidence was bypassed by policy override.",
        "warn.spec_draft_no_fit": "{service}: draft artifact ({bytes} bytes) does not fit the available memory budget; speculation is disabled.",
        "warn.spec_draft_gpu_unmodeled": "{service}: draft placement on GPU is not modeled; speculation is disabled.",
        "warn.spec_draft_kv_unmodeled": "{service}: draft GGUF lacks attention layout metadata; its KV cache is unbudgeted.",
        "warn.spec_draft_missing": "{service}: draft artifact {draft} was not found as a local GGUF file; speculation is disabled.",
        "warn.spec_degraded": "Speculation evidence was measured during a degraded host epoch; the record is saved but cannot enable a positive speed claim.",
        "warn.spec_demoted": "{count} stored speculation measurements were invalidated because this host is now measurably faster than when they were taken; evidence must be re-measured.",
        "warn.bench_spec_demoted": "{count} stored speculation measurements were invalidated by a faster shared reference epoch; evidence must be re-measured.",
        "warn.quality_prior": "Model ranking uses unverified catalog quality claims; nmesh eval measures them.",
        "note.quality_measured_selected": (
            "{model} at {quant} on {backend} was measured at {rate} on the "
            "{suite} suite here, so this choice rests on measurement; "
            "unmeasured alternatives were ranked by catalog claims and only "
            "outrank it once nmesh eval measures them too."
        ),
        "warn.quality_unmeasured": (
            "Unmeasured quality excluded these models from automatic ranking: "
            "{models}{remaining}; nmesh eval measures it."
        ),
        "warn.quality_unmeasured_selected": (
            "{model} was explicitly selected without a quality measurement; "
            "its ranking used the speed term only."
        ),
        "warn.quality_contradiction": "{role}: measured pass rate ranks {other} ({other_rate:.0%}) above {selected} ({selected_rate:.0%}) while the catalog prior does the opposite ({other_prior} < {selected_prior}); exact p={p_value:.4f} across {compared} tasks.",
        "warn.eval_incomparable_conditions": "{model}: stored pass rates were measured under different conditions ({left} vs {right}), so they are not compared; measurement conditions change this verdict (a real pair moved from exact p=0.0352 to p=0.1796 between prompt-cache conditions). Re-run nmesh eval for both under the same suite, grader digest, reasoning allowance and cache condition.",
        "warn.eval_evidence_override": "{role}: measurement outranks the catalog prior \u2014 {other} ({other_rate:.0%}) beats {selected} ({selected_rate:.0%}) with exact p={p_value:.4f} across {compared} tasks, so {other} is planned even though the prior ranks it lower ({other_prior} < {selected_prior}). Both rates were measured on the planned quant and backend under the same grader. Pass --ignore-eval-evidence to rank by the prior instead.",
        "warn.eval_evidence_override_quant": "{role}: measurement outranks the quantization penalty \u2014 {model} {other_quant} ({other_rate:.0%}) beats {selected_quant} ({selected_rate:.0%}) with exact p={p_value:.4f} across {compared} tasks, so {other_quant} is planned even though QUANT_PENALTY ranks it lower ({other_penalty} > {selected_penalty}). Both rates were measured on the same backend under the same grader. Pass --ignore-eval-evidence to rank by the penalty instead.",
        "warn.quant_penalty_contradiction": "{role}: measured pass rate ranks {model} {other_quant} ({other_rate:.0%}) above {selected_quant} ({selected_rate:.0%}) while QUANT_PENALTY does the opposite ({other_penalty} > {selected_penalty}); exact p={p_value:.4f} across {compared} tasks. The penalty table is an unmeasured estimate.",
        "note.quant_indistinguishable": "{role}: {model} {other_quant} measured {other_rate:.0%} against {selected_quant}'s {selected_rate:.0%} across {compared} tasks (exact p={p_value:.4f}), so this suite cannot distinguish them, yet QUANT_PENALTY separates them by {penalty_gap} points ({selected_penalty} vs {other_penalty}). The minimum difference this comparison can resolve is {minimum:.1%}; the ranking between these two quantizations is neither confirmed nor contradicted.",
        "note.eval_underpowered": "The {tasks}-task suite cannot resolve this observed ranking: {other_rate:.1%} versus {selected_rate:.1%} (exact p={p_value:.4f}); its minimum resolvable difference at α=0.05 is {minimum:.1%}, so the ranking is neither confirmed nor contradicted. The {upgrade_tasks}-task extended suite resolves down to {upgrade_minimum:.1%}.",
        "note.eval_underpowered_full": "The {tasks}-task suite cannot resolve this observed ranking: {other_rate:.1%} versus {selected_rate:.1%} (exact p={p_value:.4f}); its minimum resolvable difference at α=0.05 is {minimum:.1%}, so the ranking is neither confirmed nor contradicted.",
        "warn.bench_excluded": "{model} {quant}: measured throughput ({tps} tok/s) excluded this model below the {threshold} tok/s threshold, although the estimate would admit it; re-benchmarking with more runs may change the plan.",
        "warn.bench_reproducibility": "Measured throughput is not reproducible on this machine (min {minimum:.2f}, max {maximum:.2f}, spread {spread:.1%}); the planner will treat the median as fact. Re-run with --runs higher.",
        "warn.bench_control": "Benchmark passes disagreed (control ratio {ratio}); this host was not stable enough to measure. {kept}",
        "warn.bench_no_control": "No control pass was run; this measurement is displayed but not stored as evidence.",
        "warn.bench_unconfirmed": "{model} {quant}: measured throughput ({tps} tok/s) is below {threshold} tok/s but needs a second agreeing measurement before exclusion.",
        "warn.bench_epoch": "Benchmark measured while the host was at {ratio:.0%} of its best observed reference speed; {kept}.",
        "warn.bench_no_reference": "Reference workload unavailable; this measurement cannot be compared across epochs.",
        "warn.bench_demoted": "{count} stored benchmark measurements were invalidated because this host is now measurably faster than when they were taken; evidence must be re-measured.",
        "warn.bench_harness_mismatch": "{count} stored benchmark measurements use an older harness and are not used as evidence; re-run nmesh bench.",
        "warn.bench_decode_unmeasurable": "bench requested {requested} decode tokens but the model served {served}; a decode rate needs at least {minimum} served tokens (llama.cpp reports predicted_ms over the steps after prefill), so no decode measurement was recorded.",
        "warn.bench_decode_short": "bench requested {requested} decode tokens but the model stopped after {served}; max_tokens is an upper bound, so the recorded decode rate is the rate at {served} tokens, not at {requested}.",
        "warn.orchestrate_degraded": "Delegation timing was measured during a degraded host epoch; the quality gate remains valid, but the cost claim is stale.",
        "warn.orchestrate_demoted": "{count} stored delegation measurements were invalidated because this host is now measurably faster than when they were taken; cost evidence must be re-measured.",
        "warn.bench_orchestrate_demoted": "{count} stored delegation measurements were invalidated by a faster shared reference epoch; cost evidence must be re-measured.",
        "info.resolved_gguf": "resolved GGUF: {name}",
        "err.no_plan": "No plan found",
        "err.no_active_plan": "No active plan",
        "err.unknown_service": "Unknown service: {service}",
        "err.restart_budget": "Restart budget exhausted: {service}",
        "err.artifact_missing": "{service}: {error} — run `nmesh up` to fetch it.",
        "err.service_unhealthy": "Service did not become healthy: {service}",
        "err.service_port_in_use": "port {port} is still in use by another process; stop it or run nmesh down to clear it.",
        "err.service_vcredist": "On Windows this can mean the Microsoft Visual C++ Redistributable is missing or outdated — install the latest vc_redist.x64.exe and retry.",
        "err.service_log_tail": "Recent backend output ({path}):\n{tail}",
        "err.no_log": "No log found for service: {service}",
        "err.runtime_start": "Runtime startup failed",
        "err.gateway_reload": "gateway reload failed: {error}",
        "err.gateway_extras": "Gateway requires the 'gateway' extra: pip install \"nmesh[gateway] @ git+https://github.com/shizukutanaka/n-.git\"",
        "err.gateway_unload": "gateway unload failed: {error}",
        "err.gateway_unavailable": "gateway unavailable: {error}",
        "err.gateway_http": "gateway request failed (HTTP {code}): {detail}",
        "err.gateway_unavailable.hint": "start services with `nmesh up` first, or check the port with `nmesh status`.",
        "err.doctor": "doctor failed: {error}",
        "err.plan": "plan failed: {error}",
        "err.profile_load": "failed to load profile: {error}",
        "err.plan_empty": "plan produced no runnable services",
        "err.plan_save": "plan failed to save: {error}",
        "err.gateway_start": "gateway failed to start: {error}",
        "err.up": "up failed: {error}",
        "err.gateway_not_ready": "gateway did not become ready; see {path}",
        "err.bench_up": "Service is not running; run nmesh up first.",
        "err.bench_measure": "Benchmark failed: {error}",
        "err.bench_embedding": "Benchmark measures decode speed, but {service} is an embedding service with no decode path; choose a decode service such as --service chat.",
        "err.bench_http": "Benchmark request for {service} failed at {url} with HTTP status {status}.",
        "label.embed_measurement": (
            "Embedding served cap: {cap} tokens; encode throughput: {tps:.1f} tok/s."
        ),
        "warn.embed_truncated": (
            "Inputs longer than {cap} tokens are silently truncated by the backend."
        ),
        "label.embed_refused": (
            "The backend refused a {tokens} token input instead of truncating it, "
            "so no silent cap follows from that probe."
        ),
        "label.retrieval_estimate": (
            "Retrieval ladder: {requests} embedding requests, measured on "
            "this host at about {minutes} minutes; the rate scales with the "
            "measured encode throughput."
        ),
        "label.retrieval_measurement": (
            "Retrieval usability: usable through {usable} tokens; "
            "degraded beyond {degraded} tokens.\n{rungs}\n"
            "Chunk recovery: {chunk}\n"
            "Pooled single vector: {pool}"
        ),
        "warn.retrieval_control": (
            "The retrieval control rung failed, so this run proves nothing "
            "about usable length."
        ),
        "warn.retrieval_degraded": (
            "Single-vector retrieval degraded beyond {degraded} tokens while "
            "the service plans context {context}. This is a single-host, "
            "single-artifact measurement; chunk long inputs instead of "
            "raising context."
        ),
        "warn.embed_retrieval_degraded": (
            "{model} at {quant} on {backend}: the backend serves planned "
            "context {context}, but single-vector rank-1 retrieval was "
            "measured at <=50% beyond {degraded} tokens on this host. "
            "Chunk long inputs instead of raising context."
        ),
        "warn.embed_retrieval_recovered": (
            "{model} at {quant} on {backend}: single-vector retrieval degraded "
            "beyond {degraded} tokens, but measured chunking recovers {hits}/{trials} "
            "at about {chunk} tokens on this host; split long inputs accordingly. "
            "For input without whitespace word boundaries, use {chunk_chars} "
            "characters as a conservative bound."
        ),
        "warn.embed_retrieval_pooled": (
            "{model} at {quant} on {backend}: single-vector retrieval degraded "
            "beyond {degraded} tokens, but measured chunking recovers {hits}/{trials} "
            "at about {chunk} tokens; the gateway can fold the pieces into one "
            "vector with measured pooled recovery "
            "{pool_hits}/{pool_trials} on this host. Set NMESH_EMBED_AUTOCHUNK=1 "
            "to opt in. Input without whitespace word boundaries is split at "
            "{chunk_chars} characters as a conservative bound."
        ),
        "warn.embed_retrieval_client": (
            "{model} at {quant} on {backend}: client-side chunking recovers "
            "{hits}/{trials} at about {chunk} tokens, but folding the pieces into "
            "one vector did not recover retrieval here. Chunking is not a verified "
            "remedy through the gateway; it will not autochunk, so chunk inputs "
            "in the client. For input without whitespace word boundaries, chunk "
            "at {chunk_chars} characters as a conservative bound."
        ),
        "warn.embed_retrieval_unrecovered": (
            "{model} at {quant} on {backend}: single-vector retrieval degraded "
            "beyond {degraded} tokens, and chunking to the measured usable length "
            "did not restore retrieval here. Chunking is not a verified remedy "
            "on this host."
        ),
        "err.spec_draft_required": "--draft is required when --kind draft is used.",
        "err.spec_repeats": "--repeats must be at least 3.",
        "err.spec_no_generative_service": "No generative service (chat, code, or worker) is available for speculation measurement.",
        "err.spec_transport": "The speculation measurement did not complete at the transport level; nothing was recorded.",
        "label.spec_control": "control A/A: worst ratio {ratio}, identical {identical}",
        "label.spec_ref_spread": "ref spread",
        "label.spec_cand_spread": "cand spread",
        "label.bench_passes": "benchmark passes ({passes}): {values} tok/s",
        "label.bench_control": "control ratio: {ratio}",
        "label.bench_kept": "previous evidence was kept",
        "label.bench_nothing_stored": "nothing was stored as evidence",
        "err.bench_save": "benchmark failed to save: {error}",
        "err.eval_up": "Service is not running; run nmesh up first.",
        "err.eval_run": "Evaluation failed: {error}",
        "label.eval_progress": "eval {index}/{total}: {task}",

        "err.eval_save": "evaluation failed to save: {error}",
        "err.eval_unknown_categories": "unknown categories for this suite: {categories}",
        "err.eval_categories": "No evaluation tasks match the requested categories.",
        "err.autotune_measure": "Autotune failed: {error}",
        "err.autotune_restore": "failed to restore original autotune configuration: {error}",
        "err.autotune_winning": "failed to restore winning autotune configuration: {error}",
        "label.item": "Item",
        "label.value": "Value",
        "label.free": "free",
        "label.vram_source": "VRAM source",
        "label.gateway_log": "Gateway log: {path}",
        "label.none": "none",
        "label.not_found": "not found",
        "label.unknown": "unknown",
        "label.backends": "Backends",
        "label.backend": "Backend",
        "label.binary": "Binary",
        "label.version": "Version",
        "label.flags": "Flags",
        "label.models": "Models",
        "label.selected_models": "Selected models",
        "label.service": "Service",
        "label.model": "Model",
        "label.languages": "Languages",
        "label.roles": "Roles",
        "label.context": "Context",
        "label.slots": "Slots",
        "label.gpu_layers": "GPU layers",
        "label.tps": "tok/s",
        "label.saved_to": "Saved to: {path}",
        "label.free_budgets": "Budgets use currently-free memory.",
        "label.telemetry_overlay": "Live telemetry overlay: {count} benchmark key(s)",
        "label.telemetry_under_load": "{count} telemetry samples overlapped other requests (or predate concurrency recording) and are excluded from the single-stream decode overlay; a per-request rate measured under load is a different quantity.",
        "label.telemetry_off_reference": "{count} live samples were measured deeper than {tokens} prompt tokens and are not used as planning evidence",
        "label.telemetry_unknown_depth": "{count} live samples predate prompt-depth recording and are not used as planning evidence",
        "label.install": "Install: {hint}",
        "label.warning": "Warning: {warning}",
        "label.memory": "Memory",
        "label.weights": "Weights",
        "label.kv": "KV",
        "warn.ollama_quant_estimate": "{service} uses the Ollama tag's own quantization; nmesh cannot control it, so this service's memory estimate is an estimate.",
        "warn.ollama_context_default": "Ollama context could not be set for {service}; it will run at Ollama's default context, not the planned {context}.",
        "label.gpu_cpu": "GPU / CPU",
        "label.acquisition_note": "{service}: acquisition note: {note}",
        "label.telemetry": "Telemetry",
        "label.samples": "Samples",
        "label.decode_median": "Decode median",
        "label.ttft_median": "TTFT median",
        "label.ttft_p95": "TTFT p95",
        "label.total_median": "Total median",
        "label.reloaded": "Reloaded: {services} (created_at={created_at})",
        "label.unloaded": "Unloaded: {services}",
        "label.service_idle": "{service}: idle",
        "label.service_sleeping": "{service}: asleep (wake on next request)",
        "label.service_failed": "{service}: failed — {reason}",
        "label.services": "Services",
        "label.state": "State",
        "label.port": "Port",
        "status.running": "running",
        "status.failed": "failed",
        "status.stopped": "stopped",
        "status.none": "no services running",
        "jobs.title": "Jobs",
        "jobs.empty": "No queued or running jobs on this gateway",
        "jobs.counts": "{service}: jobs running={running} queued={queued}",
        "jobs.cancelled": "Cancelled {job}",
        "err.jobs_missing": "No such job: {job} (finished jobs are pruned; nmesh jobs lists recent ones)",
        "err.jobs_cancel": "Cannot cancel {job} — it already started or finished (only queued jobs can be cancelled)",
        "label.job": "Job",
        "label.endpoint": "Endpoint",
        "label.age_s": "Age (s)",
        "err.jobs_gateway": "Gateway not reachable on port {port} — is the stack up? (nmesh up)",
        "err.jobs_old_gateway": "Gateway on port {port} is running an older build without job tracking — restart it (nmesh down && nmesh up)",
        "label.median_decode": "median decode: {marker}{value} tok/s",
        "label.decode_range": "decode range: {minimum:.2f}–{maximum:.2f} tok/s (spread {spread:.1%})",
        "label.prefill": "prefill:       {marker}{value} tok/s",
        "label.ttft": "TTFT:          {marker}{value} s",
        "label.backend_detail": "{service}: backend={backend} model_ref={model_ref} port={port} context={context} slots={slots} n_gpu_layers={layers} argv={argv}",
        "label.launcher_written": "Launcher written to: {path}",
        "label.gateway_env": "Gateway environment file: {path}",
        "label.unit_written": "Unit written to: {path}",
        "label.autostart_limitation": "Limitation: {text}",
        "autostart.windows_limitations": "Windows: /sc onlogon starts only after a user logs on. Boot startup requires /sc onstart and SYSTEM or saved credentials. The current simple Task Scheduler setup has no gateway self-crash restart.",
        "note.gpu_pinned": "{service}: pinned to GPU(s) {indices} (estimated placement; enforced via device visibility)",
        "note.tps_estimate": "~ = catalog estimate, not yet measured on this machine; `nmesh bench` replaces it with measured throughput.",
        "note.eval_scope": "{tasks}-task deterministic micro-eval: instruction following, output format, extraction, translation direction. Not a knowledge benchmark; do not compare with MMLU-style scores.",
        "note.eval_value_vs_discipline": "Of the {checked} failures that have a value check, {value_only} produced the correct value and failed only the required output form; those failures measure output discipline, not capability.",
        "note.eval_failure_kinds": "{failures} failures: {value} wrong answers, {form} correct answers in the wrong output form. A form failure is recoverable by constraining the output; a wrong answer is not.",
        "note.eval_underpowered_paired": "The two configurations were compared on {tasks} paired tasks; only {discordant} disagreed ({better_only} one way, {worse_only} the other; exact p={p_value:.4f}). The exact paired test cannot reach p < 0.05 with fewer than {required} disagreeing tasks regardless of suite size, so adding tasks that both configurations pass or both fail buys nothing (minimum imbalance at this discordant total: {imbalance}).",
        "note.eval_paired_power": "Only the {discordant} disagreeing tasks carry paired information ({here} here, {there} there) out of {compared} compared tasks. The listed families ({families}) contributed no paired information. Fewer than {required} disagreeing tasks cannot reach p < 0.05 however many tasks are added.",
        "note.eval_uncertainty": "95% Wilson interval for the overall pass rate: {lo:.1%}–{hi:.1%}; this {tasks}-task suite cannot resolve pass-rate differences below {minimum:.1%} (exact test, α=0.05).",
        "note.eval_suite_upgrade": "The {tasks}-task extended suite lowers the minimum resolvable difference to {minimum:.1%} at α=0.05; run `nmesh eval --suite extended` to get it.",
        "warn.eval_config_mismatch": "{model}: an eval pass rate exists but not for the planned configuration ({quant}, {backend}); measured pass rates are not portable across quantizations or backends. Re-run nmesh eval against the running service.",
        "warn.context_unmeasured": (
            "{model} at {quant} on {backend}: the KV cache fits, so the plan "
            "advertises context {context}, but quality evidence only reaches "
            "{depth} real prompt tokens. Single-needle retrieval held to "
            "approximately 15k prompt tokens on the measured host and artifact; "
            "this is an evidence gap, not a known failure."
        ),
        "warn.context_depth_lost": (
            "{model} at {quant} on {backend}: {family} probes passed the shallow "
            "control but only {passed}/{of} at {depth} real prompt tokens — this "
            "is a measured depth failure, not task difficulty."
        ),
        "warn.embed_context_capped": (
            "{model} at {quant} on {backend}: embedding served context is capped "
            "at {cap} tokens, so planned context {context} is reduced."
        ),
        "warn.embed_context_unverified": (
            "{model} at {quant} on {backend}: embedding served context is "
            "unverified; run {command}."
        ),
        "note.embed_context_untruncated": (
            "{model} at {quant} on {backend}: the measurement found no silent "
            "truncation, so planned context {context} stands; oversize inputs "
            "are refused rather than shortened."
        ),
        "warn.context_depth_broken": (
            "{model} at {quant} on {backend}: context probes that passed the "
            "shallow control failed at {depth} real prompt tokens, so the planned "
            "context {context} is measured as broken, not merely unmeasured."
        ),
        "note.context_probe_uncontrolled": (
            "The {families} context probe families failed the shallow control, "
            "so their deep result says nothing about depth."
        ),
        "note.eval_config": "This pass rate applies to {model} at {quant} on {backend} only. Measured here: the same model at the same Q4_K_M label scored 12/16 on llama.cpp and 13/16 on Ollama.",
        "note.eval_divergence": "{config}: pass rate {other_rate} there vs {rate} here; {count} of the {compared} compared tasks disagree ({ids}). Measured here: two fp16 configurations of Qwen2.5 0.5B both scored 9/16 while disagreeing on 2 tasks in opposite directions, so an equal pass rate does not mean equivalent behaviour.",
        "warn.eval_artifact_changed": "{model} at {quant} on {backend}: the cached pass rate was measured on artifact {previous}, but this service loads {current}. The same model/quant label can map to different weight files, so the cached rate does not describe this artifact.",
        "warn.eval_stale_grader": "{model} at {quant} on {backend}, suite {suite}: this record was graded by a different rule version and is not used for planning.",
        "warn.eval_unscorable": "{count} of {tasks} tasks returned no answer text before the output budget ran out (finish_reason=length with empty content), so this run measures the budget, not the model, and is not used as planning evidence. Measured here: gemma-4-26B-A4B returned empty content on all 104 tasks at the suite budget and answered correctly at 512 tokens. Current reasoning allowance: {allowance} tokens; raise it with `nmesh eval --reasoning-allowance N`.",
        "warn.eval_transport": "{count} tasks failed to answer at the transport level (timeout or HTTP error); those are host failures, not wrong answers, so this run is recorded but not used as quality evidence.",
        "evidence.bench_title": "Saved benchmark evidence",
        "evidence.eval_title": "Saved evaluation evidence",
        "evidence.depth_title": "Saved depth evidence",
        "evidence.embed_title": "Saved embedding evidence",
        "evidence.depth_json_hint": "Depth family details are available in nmesh evidence --json.",
        "evidence.empty": "No saved evidence.",
        "evidence.column.kind": "Kind",
        "evidence.column.model": "Model",
        "evidence.column.quant": "Quant",
        "evidence.column.backend": "Backend",
        "evidence.column.scope": "Scope",
        "evidence.column.value": "Value",
        "evidence.column.usable": "Usable",
        "evidence.column.reasons": "Reasons",
        "evidence.column.remeasure": "Remeasure",
        "evidence.reason.harness_mismatch": "stored under an older benchmark harness",
        "evidence.reason.cap_unproven": "two different probe sizes did not prove a served cap",
        "evidence.reason.input_refused": "the backend refused the oversize probe instead of truncating it",
        "evidence.reason.unstable": "the controlled benchmark was unstable",
        "evidence.reason.epoch_degraded": "the benchmark was measured on a degraded host epoch",
        "evidence.reason.unconfirmed": "fewer than two confirming benchmark sessions exist",
        "evidence.reason.suite_unknown": "the stored evaluation suite is not known",
        "evidence.reason.grader_digest_mismatch": "the stored evaluation uses stale grading rules",
        "evidence.reason.partial_suite": "a filtered subset ran; run the full suite for planner-usable evidence",
        "evidence.reason.unscorable": "some evaluation tasks had no scorable answer",
        "evidence.reason.transport_errors": "some evaluation tasks failed at transport level",
        "evidence.reason.depth_scoped": "deep evaluation is not used for shallow planner comparison",
        "evidence.reason.probe_digest_mismatch": "the stored depth probe rules are stale",
        "evidence.reason.control_failed": "the shallow control did not establish attributable depth evidence",
        "evidence.reason.depth_lost": "the deep probe lost an attributable task family",
        "evidence.reason.superseded": "a newer valid record for the same configuration is used",
        "evidence.retrieval_title": "Saved retrieval evidence",
        "evidence.spec_title": "Saved speculation evidence",
        "evidence.reason.spec_unstable": "the speculation control did not reproduce itself on this host",
        "evidence.reason.spec_stale": "the speculation speed claim was invalidated by a faster host epoch",
        "evidence.reason.spec_not_identical": "speculation changed the output tokens, so it is not free speed",
        "evidence.reason.spec_not_faster": "no workload class gained enough speed to pay for speculation",
        "evidence.reason.spec_mixed": "some classes gained and others regressed; the traffic mix is unknown",
        "evidence.reason.spec_no_evidence": "no workload classes were measured for this speculation configuration",
        "label.eval_category": "Category",
        "label.eval_passed": "Passed",
        "label.eval_total": "Total",
        "label.eval_pass_rate": "Pass rate",
        "label.eval_overall": "Overall pass rate: {passed}/{total} ({rate:.1%})",
        "label.eval_failed": "Failed task IDs: {ids}",
        "label.eval_title": "nmesh eval",
        "label.eval_depth": "Requested prompt depth: {requested}; served depth: {served}",
        "label.eval_context_probe": "Context probes: {passed}/{total} (literal {literal_passed}/{literal_total}, latent {latent_passed}/{latent_total}, multi {multi_passed}/{multi_total})",
        "label.eval_context_control": "Context probe controls: {passed}/{total}",
        "err.orchestrate_plan": "No saved plan with runnable services found.",
        "err.orchestrate_service": "Could not resolve lead {lead} and worker {worker} from the plan.",
        "err.orchestrate_no_worker": "Delegation needs a second generative service besides lead {lead}; add one to the plan or pass --worker-url for an external worker.",
        "err.orchestrate_nongenerative": "The selected {role} service {service} is not generative; choose a chat, code, or worker service.",
        "err.orchestrate_up": "A selected service is not running; provide its URL or run nmesh up first.",
        "err.orchestrate_measure": "Delegation measurement failed: {error}",
        "label.orchestrate_summary": "Delegation measurement: n={n}",
        "label.orchestrate_passed": "passed: worker={worker}, lead={lead}, delegated={delegated}, ceiling={ceiling}",
        "label.orchestrate_comparison": "{name} vs lead: gained={gained}, lost={lost}, p={p:.4f}",
        "label.orchestrate_verifier": "verifier: accuracy={accuracy:.4f}, accepted={accepted}, accepted-but-wrong={wrong}, rejected-but-right={right}, unparsed={unparsed}",
        "label.orchestrate_cost": "lead cost: solo={solo}, delegated={delegated}, overhead={overhead:.3f}; seconds: solo={solo_seconds:.2f}, delegated={delegated_seconds:.2f}",
        "label.orchestrate_gate": "gate: {decision} ({reason})",
        "label.orchestrate_repeats": "repeats: {repeats}, unstable tasks: {unstable}",
        "label.orchestrate_empty": "No delegation measurements recorded.",
        "label.orchestrate_title": "nmesh orchestrate show",
        "err.delegate_stream": "nmesh-delegate does not support streaming.",
        "err.delegate_worker": "nmesh-delegate requires two distinct planned services.",
        "err.delegate_not_coresident": "nmesh-delegate is unavailable: lead {lead} and worker {worker} are mutually exclusive by memory.",
        "err.delegate_gate": "nmesh-delegate is disabled by the evidence gate: {reason}.",
        "err.delegate_gate_stats": "nmesh-delegate is disabled by the evidence gate: {reason} (delegated={delegated}, lead={lead}, p={p:.4f}).",
        "note.watch_external_claim": "External posts are pointers, not evidence; findings are confirmed against queryable ground truth.",
        "note.watch_route": "Route findings are local to this gateway; third-party applications may expose routes nmesh does not implement.",
        "note.watch_zenn_body": "Zenn RSS summaries carry no technical content; this run fetched article bodies through the article API and HTML pages.",
        "note.watch_caps_unavailable": "Recorded llama.cpp capabilities were unavailable; flag comparison was not performed.",
        "note.watch_unknown_flag": "An unknown flag is not proof that the feature is missing; it may belong to another backend or version.",
        "warn.watch_unreachable": "{source} was not reachable: {detail}",
        "warn.watch_state": "Watch state was not saved: {error}",
        "warn.watch_draft": "Watch draft was not written: {error}",
        "err.watch_run": "Watch failed: {error}",
        "label.watch_title": "nmesh watch",
        "label.watch_source": "Source",
        "label.watch_reachable": "Reachable",
        "label.watch_items": "Items",
        "label.watch_detail": "Detail",
        "label.watch_finding": "{kind}: {value} ({mentions} mention(s))",
        "label.watch_catalog_title": "Catalog coverage",
        "label.watch_metric": "Metric",
        "label.watch_value": "Value",
        "label.watch_catalog_entries": "Catalog entries",
        "label.watch_catalog_repo_ids": "Catalog repo IDs",
        "label.watch_catalog_mentioned": "Mentioned repo IDs",
        "label.watch_catalog_in_catalog": "Mentioned IDs already in catalog",
        "label.watch_catalog_resolved": "Resolved repo IDs",
        "label.watch_catalog_absent": "Resolved repo IDs absent from catalog",
        "label.watch_candidates_title": "Candidate feasibility",
        "label.watch_fit_class": "Fit class",
        "label.watch_fit_no_weights": "No primary weights",
        "label.watch_fit_gated": "Gated",
        "label.watch_fit_role_unknown": "Role unknown",
        "label.watch_fit_not_text": "Not text generation",
        "label.watch_fit_too_large": "Too large",
        "label.watch_fit_fits": "Fits",
        "label.watch_candidate_budget": "Planner budget",
        "label.watch_filename": "Filename",
        "label.watch_install": "Install with",
    },
    "ja": {
        "label.telemetry_off_reference": "{count}件のライブサンプルは{tokens}プロンプトトークンより深く測定されたため、計画の証拠には使用しません",
        "label.telemetry_unknown_depth": "{count}件のライブサンプルはプロンプト深度の記録前のもので、計画の証拠には使用しません",
        "warn.backend_binary_missing": "{backend}: \u6307\u5b9a\u3055\u308c\u305f\u30d0\u30a4\u30ca\u30ea {path} \u306f\u5b9f\u884c\u53ef\u80fd\u3067\u306f\u306a\u3044\u305f\u3081\u3001PATH \u306b\u30d5\u30a9\u30fc\u30eb\u30d0\u30c3\u30af\u305b\u305a\u30d0\u30c3\u30af\u30a8\u30f3\u30c9\u3092\u5229\u7528\u4e0d\u53ef\u3068\u3057\u307e\u3059\u3002",
        "warn.embeddings_unsupported": "{model}: llama.cpp は埋め込みフラグに対応していないため、/v1/embeddings は利用できない可能性があります。",
        "warn.embeddings_pooling_unknown": "{model}: pooling のメタデータが nmesh に不明、またはこの llama.cpp ビルドで未対応です。publisher/vendor が pooling type 'none' を使うと、OpenAI 互換の埋め込みエンドポイントは拒否する可能性があります。",
        "warn.rerank_unsupported": "{model}: この llama.cpp ビルドは --reranking に対応していないため、rerank サービスは /v1/rerank に答えられません。",
        "warn.embeddings_batch_limit": "{model}: 物理バッチサイズのフラグに対応していないため、512 トークンを超える入力は拒否される可能性があります（計画コンテキスト {context}）。",
        "warn.embeddings_backend_unsupported": "{model}: mlx_lm.server には埋め込みエンドポイントがありません。",
        "warn.embeddings_backend_unverified": "{model}: vLLM の埋め込みと pooling の挙動は nmesh で未検証です。nmesh は設定を推測して注入しません。",
        "install.ollama": "Ollamaをインストールしてください: https://ollama.com/download",
        "install.llamacpp": "nmesh engine install で llama.cpp をインストールできます。winget / brew / ソースからのビルドも利用できます",
        "engine.installed": "インストール済みエンジン",
        "engine.active": "アクティブなエンジン",
        "engine.available": "利用可能なビルド",
        "engine.tag": "タグ",
        "engine.variant": "バリアント",
        "engine.active_column": "使用中",
        "engine.path": "パス",
        "engine.install": "{tag}（{variant}）をインストールしました",
        "info.engine_autoinstall": "llama.cpp バックエンドが見つからないため、管理対象エンジン {tag}（{variant}）をインストールします。",
        "engine.use": "{tag}（{variant}）を使用します",
        "engine.remove": "{tag}を削除しました",
        "engine.active_cleared": "アクティブなエンジンを解除しました",
        "models.removed": "{path}を削除しました",
        "models.planned_refusal": "プランが参照するモデルは削除できません: {path}。--force を使用してください",
        "models.local": "nmesh ローカルモデル",
        "models.scan": "nmesh モデル一覧",
        "models.scan_store": "{store}: {files} ファイル、{gib:.2f} GiB",
        "models.scan_total": "合計: {files} ファイル、{gib:.2f} GiB",
        "models.scan_reclaimable": "重複から回収可能: {gib:.2f} GiB",
        "models.scan_variants": "バリアントグループ: {count}",
        "err.unload_not_running": "{service} をアンロードしませんでした: 起動していません。",
        "err.unload_idle": "{service} をアンロードしませんでした: すでにアイドル状態です。",
        "err.unload_shared": "{service} をアンロードしませんでした: 共有デーモンのため nmesh は停止しません。",
        "err.unload_external": "{service} をアンロードしませんでした: nmesh が待ち受けを見つけただけの外部サーバーのため、nmesh は停止しません。",
        "err.unload_not_owned": "{service} をアンロードしませんでした: nmesh は自身が所有していない実行中プロセスを停止しません。nmesh down を使用してください。",
        "err.unload_unknown": "{service} をアンロードしませんでした。",
        "install.vllm": "vLLMをインストールしてください: pip install vllm",
        "install.mlx": "MLX-LMをインストールしてください: pip install mlx-lm",
        "warn.parallel_unsupported": "llama.cpp: --parallelは未対応です。1スロットと通常のコンテキストを使用します",
        "warn.gpu_layers_unsupported": "llama.cpp: GPUレイヤー指定は未対応です。CPU配置を使用します",
        "warn.tensor_split_unsupported": "llama.cpp: --tensor-splitは未対応です。テンソル分割を省略します",
        "warn.tensor_split_proportional": "{service}: 各GPUの利用可能VRAMに合わせ、{split} の比例配分(--tensor-split)でモデルを分散します。",
        "warn.gpu_over_budget": "{service}: GPU配置に{committed:.0f}バイト必要ですが、予算{budget:.0f}バイトを超えています",
        "warn.kv_quant_unsupported": "{service}: {backend} は要求された KV 精度 {requested} に対応していないため、KV キャッシュは f16 として計上します。",
        "warn.kv_quant_speed_unmodeled": "{service}: 計画のスループットは KV キャッシュ種別を考慮していません。参照ビルドでは {kv_quant} KV のプリフィルとデコードが遅くなりました。",
        "warn.gguf_mixed_precision": "{service}: {planned} 用に解決された GGUF は {label} ({filename}) です。{planned} の単一量子化ではない混合精度アーティファクトであり、この量子化ラベルで記録された測定値はこのアーティファクトを示します。",
        "warn.gguf_size_mismatch": "{service}: 解決された GGUF は {actual} バイトですが、推定値は {estimated} バイトです。計画の適合判定にはファイルではなく、パラメータ数 × 公称 bpw の推定値を使用しました。",
        "warn.gguf_corrupt": "{service}: キャッシュ済み GGUF {filename} は {actual} バイトですが、ダウンロード時の記録は {expected} バイトです。アーティファクトが破損しているため再取得します。",
        "warn.quant_fallback_skipped": "{service}: 実行中アーティファクトのラベル {quant} は計画の量子化ラダー外のため、量子化フォールバックを省略しました。",
        "warn.backend_placement_estimate": "{service} は {backend} を使用し、配置を独自に管理します。この RAM フォールバックは nmesh の推定であり、制御可能な設定ではありません。",
        "warn.selection_capacity_tradeoff": "{role}: 空き容量だけの場合は {empty_model} {empty_quant} ですが、GPU {reserved_vram_gib:.1f} GiB / RAM {reserved_ram_gib:.1f} GiB が既に使用中のため、{chosen_model} {chosen_quant} を選択しました。これは容量によるトレードオフです。",
        "warn.layers_reduced": "{service}: GPUレイヤーを{layers}に減らしました。{previous}レイヤーはRAMに収まりません",
        "warn.slots_clamped": "{service}: parallel_slots {requested}は空きメモリを超えるため、{slots}にします",
        "warn.engine_substituted": "{service}: 計画時のエンジン {old} は削除済みのため、インストール済みの {tag} で起動します。",
        "warn.slots_unsupported": "{service}: {backend}はここでは{requested}並列スロットを処理できないため、プランでは1を使用します。",
        "warn.slots_tradeoff": "{service}: {slots}並列スロットでは合計スループットは向上しますが、リクエスト単位のレートは低下します。表示された{tps} tok/sは単一ストリームの値です（1台のCPUマシンで測定: 単独46.2 tok/s、4件同時36.84、8件同時23.30、合計125および148）。",
        "warn.speed_saturated": "{role}: --prefer speedでは判別できませんでした。{chosen} {chosen_quant}（{chosen_tps} tok/s）が{other} {other_quant}（{other_tps} tok/s）より上位になったのは、スコアの速度項が{reference} tok/sで飽和するためで、未検証の品質事前分布が決定したためです。",
        "warn.no_source": "{model}: Hugging FaceまたはOllamaのソースが設定されていません",
        "warn.no_candidate": "ロール{role}に実行可能なモデルがありません",
        "warn.nvidia_unavailable": "NVIDIA の検出を利用できません。",
        "warn.parallel_clamped": "llama.cpp: --parallel は未対応です。{requested} スロットではなく1スロットを使用します。",
        "warn.rocm_parse": "rocm-smi の出力を解析できません。",
        "warn.mlx_check": "mlx_lm を確認できません。",
        "warn.system_probe": "システム検出に失敗しました: {error}",
        "warn.download_budget": "予定ダウンロードは合計 {total_gb:.1f} GiB で、ダウンロード予算 {limit_gb:.1f} GiB を超えています（policy.allow_download_gb）。", 
        "warn.language_coverage": "{model}は{languages}をカバーすると主張していません。言語対応は公開元/ベンダーの主張であり、測定ベンチマークではありません。",
        "hint.language": "ロケールは{language}です。{language}対応を主張するモデルを優先するには --lang {language} を使用してください。",
        "warn.free_admission": "空きメモリ判定: 使用予定はVRAM {need_gpu:.2f} GiB / RAM {need_cpu:.2f} GiBで、空きはVRAM {vram:.2f} GiB / RAM {ram:.2f} GiBです。",
        "warn.real_artifact_replanned": "{service}: 実アーティファクトのバイト数が推定値を超えたため、起動前に空きメモリを基準として再計画しました。",
        "warn.free_admission_fallback": "空きメモリ判定に失敗したため元のプランを使用します。理由: {error}",
        "warn.runtime_fallback": "ランタイムのフォールバック試行 {attempt}",
        "warn.admission_skipped": "空きメモリ判定をスキップしました: {error}",
        "warn.health_failed": "サービスのヘルスチェックに失敗しました: {service}",
        "warn.worker_not_coresident": "ワーカーロールが要求されましたが、リードと共存できる厳密に小さい生成モデルがありません。委譲は利用できません。",
        "warn.spec_unsupported": "{service}: 検出された llama.cpp のフラグでは投機的デコードを利用できないため、フラグを出力しません。",
        "warn.sleep_idle_unsupported": "{model}: この llama.cpp ビルドは --sleep-idle-seconds に対応していないため、サービスは常駐したままになります。",
        "warn.moe_cpu_offload": "{service}: llama.cpp は {model} の全レイヤーをGPUに置きつつ、{layers} MoEレイヤーのエキスパートテンソルをCPUに置きます(--n-cpu-moe)。表示のデコード速度は実測まで推定値です。",
        "warn.moe_cpu_unsupported": "{model}: この llama.cpp ビルドは --n-cpu-moe 非対応のため、計画していたMoEエキスパートオフロードを取りやめました。",
        "warn.vllm_sleep_unsupported": "{service}: vLLM {version} はスリープモードに未対応です（0.9以上が必要）。スワップ切替のたびにエンジンを再起動します。",
        "warn.cache_reuse_unsupported": "{model}: この llama.cpp ビルドは --cache-reuse に対応していないため、リクエスト間でプロンプトキャッシュは再利用されません。",
        "warn.context_shift_enabled": "{model}: コンテキストシフトが有効です。生成がコンテキスト窓を超えると最古のトークンが静かに捨てられます（窓を超える入力自体は従来どおり拒否されます）。",
        "warn.context_shift_unsupported": "{model}: この llama.cpp ビルドは --context-shift に対応していないため、窓を超える入力は従来どおり拒否されます。",
        "warn.up_flags_saved_plan": "{flags}: 保存済みのプランが使われているため、これらのオプションは無視されます。適用するには nmesh plan で再計画してください。",
    "warn.plan_stale": "保存済みのプランは古い nmesh で生成されました（起動フラグの仕様が変更されています）。最新の改善を取り込むには `nmesh plan` を再実行してください。",
        "warn.spec_refused": "{service}: 投機的デコードの証拠を拒否しました（{reason}）。速度向上: {speeds}。",
        "warn.spec_override": "{service}: ポリシーの上書きにより投機的デコードの証拠確認を省略しました。",
        "warn.spec_draft_no_fit": "{service}: ドラフトの実体ファイル（{bytes} バイト）がメモリ予算に収まらないため、投機的デコードを無効にします。",
        "warn.spec_draft_gpu_unmodeled": "{service}: GPU上のドラフト配置はモデル化されていないため、投機的デコードを無効にします。",
        "warn.spec_draft_kv_unmodeled": "{service}: ドラフトGGUFにアテンション構成のメタデータがないため、そのKVキャッシュは予算未計上です。",
        "warn.spec_draft_missing": "{service}: ドラフト実体 {draft} がローカルGGUFファイルとして見つからないため、投機的デコードを無効にします。",
        "info.resolved_gguf": "解決したGGUF: {name}",
        "err.no_plan": "プランが見つかりません",
        "err.no_active_plan": "アクティブなプランがありません",
        "err.unknown_service": "不明なサービス: {service}",
        "err.restart_budget": "再起動上限を使い切りました: {service}",
        "err.artifact_missing": "{service}: {error} — 取得するには `nmesh up` を実行してください。",
        "err.service_unhealthy": "サービスが正常になりませんでした: {service}",
        "err.service_port_in_use": "ポート {port} が他のプロセスで使用中です。そのプロセスを停止するか nmesh down で解放してください。",
        "err.service_vcredist": "Windows では Visual C++ 再頒布パッケージの不足・旧版が原因の可能性があります — 最新の vc_redist.x64.exe をインストールして再試行してください。",
        "err.service_log_tail": "バックエンドの最近の出力 ({path}):\n{tail}",
        "err.no_log": "サービスのログが見つかりません: {service}",
        "err.runtime_start": "ランタイムの起動に失敗しました",
        "err.gateway_reload": "ゲートウェイの再読み込みに失敗しました: {error}",
        "err.gateway_unavailable": "ゲートウェイを利用できません: {error}",
        "err.gateway_http": "ゲートウェイ要求が失敗しました（HTTP {code}）: {detail}",
        "err.gateway_unavailable.hint": "先に `nmesh up` でサービスを起動するか、`nmesh status` でポートを確認してください。",
        "label.eval_progress": "評価 {index}/{total}: {task}",
        "err.bench_up": "サービスが起動していません。先にnmesh upを実行してください。",
        "err.bench_measure": "ベンチマークに失敗しました: {error}",
        "err.bench_embedding": "ベンチマークはデコード速度を測定しますが、{service}は埋め込みサービスでデコード経路がありません。--service chatなどデコード可能なサービスを指定してください。",
        "err.bench_http": "{service}のベンチマーク要求が{url}でHTTPステータス{status}に失敗しました。",
        "label.embed_measurement": "埋め込みの実測上限: {cap} トークン、エンコード速度: {tps:.1f} tok/s。",
        "warn.embed_truncated": "{cap} トークンを超える入力はバックエンドで静かに切り詰められます。",
        "label.embed_refused": "バックエンドは {tokens} トークンの入力を切り詰めずに拒否したため、このプローブから静かな上限は得られません。",
        "err.spec_draft_required": "--kind draft では --draft が必要です。",
        "err.spec_repeats": "--repeats は 3 以上で指定してください。",
        "err.spec_no_generative_service": "投機的デコードの測定に使える生成サービス（chat、code、worker）がありません。",
        "label.spec_control": "A/A対照: 最悪比率 {ratio}、同一出力 {identical}",
        "label.spec_ref_spread": "参照ばらつき",
        "label.spec_cand_spread": "候補ばらつき",
        "err.autotune_measure": "自動調整に失敗しました: {error}",
        "label.item": "項目",
        "label.value": "値",
        "label.free": "空き",
        "label.none": "なし",
        "label.not_found": "見つかりません",
        "label.unknown": "不明",
        "label.backends": "バックエンド",
        "label.backend": "バックエンド",
        "label.binary": "バイナリ",
        "label.version": "バージョン",
        "label.flags": "フラグ",
        "label.models": "モデル",
        "label.selected_models": "選択されたモデル",
        "label.service": "サービス",
        "label.services": "サービス",
        "label.state": "状態",
        "label.port": "ポート",
        "status.running": "稼働中",
        "status.failed": "失敗",
        "status.stopped": "停止",
        "status.none": "稼働中のサービスはありません",
        "jobs.title": "ジョブ",
        "jobs.empty": "このゲートウェイに待機中・実行中のジョブはありません",
        "jobs.counts": "{service}: ジョブ running={running} queued={queued}",
        "jobs.cancelled": "{job} をキャンセルしました",
        "err.jobs_missing": "ジョブが見つかりません: {job}（完了ジョブは削除されます — nmesh jobs で直近を確認）",
        "err.jobs_cancel": "{job} はキャンセルできません — 既に開始または終了しています（待機中のジョブのみ取消可能）",
        "label.job": "ジョブ",
        "label.endpoint": "エンドポイント",
        "label.age_s": "経過（秒）",
        "err.jobs_gateway": "ポート {port} のゲートウェイに接続できません — スタックは起動していますか？（nmesh up）",
        "err.jobs_old_gateway": "ポート {port} のゲートウェイはジョブ追跡のない古いビルドです — 再起動してください（nmesh down && nmesh up）",
        "label.model": "モデル",
        "label.languages": "言語",
        "label.roles": "ロール",
        "label.context": "コンテキスト",
        "label.slots": "スロット",
        "label.gpu_layers": "GPUレイヤー",
        "label.tps": "トークン/秒",
        "label.saved_to": "保存先: {path}",
        "label.free_budgets": "予算には現在空いているメモリを使用します。",
        "label.telemetry_overlay": "ライブテレメトリの上書き: ベンチマークキー{count}件",
        "label.telemetry_under_load": "他のリクエストと重なったテレメトリサンプル（または同時実行数の記録前のサンプル）{count}件を単一ストリームのデコードオーバーレイから除外しました。負荷中に測定したリクエスト単位のレートは異なる値です。",
        "label.install": "インストール: {hint}",
        "label.warning": "警告: {warning}",
        "label.memory": "メモリ",
        "label.weights": "重み",
        "label.kv": "KV",
        "label.gpu_cpu": "GPU / CPU",
        "warn.gpu_layers_cpu_fallback": "{service}: GPU レイヤーがどのカードにも収まらないため、CPU で実行します。",
        "label.acquisition_note": "{service}: 取得メモ: {note}",
        "label.telemetry": "テレメトリ",
        "label.samples": "サンプル数",
        "label.decode_median": "デコード中央値",
        "label.ttft_median": "TTFT中央値",
        "label.ttft_p95": "TTFT p95",
        "label.total_median": "合計中央値",
        "label.reloaded": "再読み込み完了: {services} (created_at={created_at})",
        "label.median_decode": "デコード中央値: {marker}{value} トークン/秒",
        "label.prefill": "プレフィル:     {marker}{value} トークン/秒",
        "label.ttft": "TTFT:          {marker}{value} 秒",
        "label.backend_detail": "{service}: backend={backend} model_ref={model_ref} port={port} context={context} slots={slots} n_gpu_layers={layers} argv={argv}",
        "warn.backend_no_gpu": "llama.cpp バイナリは GPU バックエンドを報告しません。CPU 配置を使用します。検出された GPU を使うには Vulkan、CUDA、HIP、または SYCL ビルドをインストールしてください。",
        "warn.ollama_quant_estimate": "{service} は Ollama タグ固有の量子化を使用します。nmesh は制御できないため、このサービスのメモリ見積もりは推定値です。",
        "warn.ollama_context_default": "{service} の Ollama コンテキストを設定できないため、計画した {context} ではなく Ollama のデフォルトコンテキストで実行します。",
        "warn.low_vram": "{gpu} の専用 VRAM は 2 GiB 未満です。表示は維持しますが、GPU 配置には使用しません。",
        "err.doctor": "doctor に失敗しました: {error}",
        "err.plan": "plan に失敗しました: {error}",
        "err.profile_load": "プロファイルの読み込みに失敗しました: {error}",
        "warn.simulated_profile": "シミュレーションプロファイル: 計画のみです。実行はせず、この機の速度・埋め込み測定値はプロファイルではなくこの機の性質なので使いません（表示速度は推定値です）。",
        "err.plan_empty": "実行可能なサービスがないプランです",
        "err.plan_save": "plan の保存に失敗しました: {error}",
        "err.gateway_start": "gateway の起動に失敗しました: {error}",
        "err.up": "up に失敗しました: {error}",
        "err.gateway_not_ready": "gateway の準備が完了しませんでした。{path} を確認してください",
        "err.bench_save": "ベンチマーク結果の保存に失敗しました: {error}",
        "err.autotune_restore": "自動調整前の設定の復元に失敗しました: {error}",
        "err.autotune_winning": "最適設定の復元に失敗しました: {error}",
        "label.vram_source": "VRAM の情報源",
        "label.gateway_log": "ゲートウェイログ: {path}",
        "warn.embed_context_capped": "{model} の {quant} / {backend} で、埋め込みの実測文脈は {cap} トークンが上限のため、計画文脈 {context} を短縮します。",
        "warn.embed_context_unverified": "{model} の {quant} / {backend} で、埋め込みの実測文脈は未確認です。{command}を実行してください。",
        "note.embed_context_untruncated": "{model} の {quant} / {backend} で、静かな切り詰めは測定されませんでした。計画文脈 {context} はそのまま有効で、上限超過の入力は短縮されず拒否されます。",
        "label.retrieval_estimate": (
            "検索ラダー: 埋め込みリクエスト {requests} 件、この機の実測レートでは"
            "およそ {minutes} 分。所要時間は実測のエンコード速度に比例します。"
        ),
        "label.retrieval_measurement": (
            "検索の有用性: {usable} トークンまで有用、{degraded} トークンを超えると劣化。\n{rungs}"
            "\nチャンク回復: {chunk}"
            "\nプール済み単一ベクトル: {pool}"
        ),
        "warn.retrieval_control": (
            "検索の制御ランが失敗したため、この実行から長さについては何も証明できません。"
        ),
        "warn.retrieval_degraded": (
            "単一ベクトル検索は {degraded} トークンを超えると劣化しましたが、サービスは"
            "コンテキスト {context} を計画しています。これは単一ホスト・単一アーティファクトの測定です。"
            "コンテキストを増やさず、長い入力を分割してください。"
        ),
        "warn.embed_retrieval_recovered": (
            "{model} の {quant} / {backend}: {degraded} トークンを超えると検索は劣化しますが、"
            "このホストでは約 {chunk} トークンへの分割で {hits}/{trials} の回復を実測しました。"
            "空白で語を区切らない入力は {chunk_chars} 文字を保守的な上限として分割してください。"
        ),
        "warn.embed_retrieval_pooled": (
            "{model} の {quant} / {backend}: {degraded} トークンを超えると検索は劣化しますが、"
            "約 {chunk} トークンへの分割で {hits}/{trials} を回復しました。"
            "ゲートウェイはこのホストで"
            "プール済み回復 {pool_hits}/{pool_trials} の単一ベクトルにまとめられます。"
            "NMESH_EMBED_AUTOCHUNK=1 で有効化できます。"
            "空白で語を区切らない入力は {chunk_chars} 文字を保守的な上限として分割します。"
        ),
        "warn.embed_retrieval_client": (
            "{model} の {quant} / {backend}: クライアント側の分割では約 {chunk} トークンで"
            "{hits}/{trials} を回復しましたが、単一ベクトルへの統合では回復しませんでした。"
            "ゲートウェイは自動分割せず、クライアント側で入力を分割してください。"
            "空白で語を区切らない入力は {chunk_chars} 文字を保守的な上限として分割してください。"
        ),
        "warn.embed_retrieval_unrecovered": (
            "{model} の {quant} / {backend}: {degraded} トークンを超えると検索は劣化し、"
            "測定した有用長への分割でも回復しませんでした。このホストでは分割を対策として検証できません。"
        ),
        "warn.embed_retrieval_degraded": (
            "{model} の {quant} / {backend}: バックエンドは計画コンテキスト {context} トークンを提供しますが、"
            "このホストで単一ベクトルの rank-1 検索は {degraded} トークンを超えると 50% 以下になりました。"
            "コンテキストを増やさず、長い入力を分割してください。"
        ),
        "evidence.embed_title": "埋め込み証拠",
        "evidence.reason.cap_unproven": "異なる2つのプローブサイズで実際の上限を確認できませんでした",
        "evidence.reason.input_refused": "バックエンドが上限超過のプローブを切り詰めずに拒否しました",
    },
}

MESSAGES["ja"].update({
    "err.gateway_extras": "\u30b2\u30fc\u30c8\u30a6\u30a7\u30a4\u306b\u306f 'gateway' \u30a8\u30af\u30b9\u30c8\u30e9\u304c\u5fc5\u8981\u3067\u3059: pip install \"nmesh[gateway] @ git+https://github.com/shizukutanaka/n-.git\"",
    "err.gateway_unload": "\u30b2\u30fc\u30c8\u30a6\u30a7\u30a4\u306e\u30a2\u30f3\u30ed\u30fc\u30c9\u306b\u5931\u6557\u3057\u307e\u3057\u305f: {error}",
    "label.unloaded": "\u30a2\u30f3\u30ed\u30fc\u30c9\u5b8c\u4e86: {services}",
    "label.service_idle": "{service}: \u30a2\u30a4\u30c9\u30eb",
    "label.service_sleeping": "{service}: \u30b9\u30ea\u30fc\u30d7\u4e2d\uff08\u6b21\u306e\u30ea\u30af\u30a8\u30b9\u30c8\u3067\u5fa9\u5e30\uff09",
    "label.service_failed": "{service}: \u5931\u6557 \u2014 {reason}",
})

MESSAGES["ja"].update({
    "note.eval_underpowered_paired": (
        "\u4e21\u65b9\u306e\u69cb\u6210\u3092{tasks}\u554f\u306e\u5bfe\u5fdc\u4ed8\u3051\u3055\u308c\u305f\u30bf\u30b9\u30af\u3067\u6bd4\u8f03\u3057\u3001"
        "\u4e0d\u4e00\u81f4\u306f{discordant}\u554f\u3060\u3051\u3067\u3057\u305f\uff08\u4e00\u65b9\u5411\u304d{better_only}\u554f\u3001"
        "\u9006\u65b9\u5411\u304d{worse_only}\u554f\uff09\uff08\u6b63\u78ba\u306a p={p_value:.4f}\uff09\u3002\u6b63\u78ba\u306a\u5bfe\u5fdc\u4ed8\u3051\u691c\u5b9a\u306f\u3001\u30b9\u30a4\u30fc\u30c8\u30b5\u30a4\u30ba\u306b\u95a2\u4fc2\u306a\u304f"
        "{required}\u554f\u672a\u6e80\u306e\u4e0d\u4e00\u81f4\u3067\u306fp < 0.05\u306b\u306a\u308a\u307e\u305b\u3093\u3002\u4e21\u65b9\u304c\u5408\u683c\u307e\u305f\u306f\u4e21\u65b9\u304c\u5931\u6557\u3059\u308b\u30bf\u30b9\u30af\u3092"
        "\u8ffd\u52a0\u3057\u3066\u3082\u5bfe\u5fdc\u4ed8\u3051\u306e\u691c\u51fa\u529b\u306f\u5897\u3048\u307e\u305b\u3093\uff08\u3053\u306e\u4e0d\u4e00\u81f4\u6570\u3067\u306e\u6700\u5c0f\u504f\u5dee: {imbalance}\uff09\u3002"
    ),
    "note.eval_paired_power": (
        "\u5bfe\u5fdc\u4ed8\u3051\u306e\u60c5\u5831\u3092\u6301\u3064\u306e\u306f\u3001\u6bd4\u8f03\u3057\u305f{compared}\u554f\u306e\u3046\u3061"
        "\u4e0d\u4e00\u81f4\u3057\u305f{discordant}\u554f\u3060\u3051\u3067\u3059\uff08\u3053\u306e\u5b9f\u884c{here}\u3001\u6bd4\u8f03\u5bfe\u8c61{there}\uff09\u3002"
        "\u6307\u5b9a\u3055\u308c\u305f\u30d5\u30a1\u30df\u30ea\u30fc\uff08{families}\uff09\u306f\u5bfe\u5fdc\u4ed8\u3051\u306e\u60c5\u5831\u3092\u63d0\u4f9b\u3057\u307e\u305b\u3093\u3002"
        "{required}\u554f\u672a\u6e80\u306e\u4e0d\u4e00\u81f4\u3067\u306f\u3001\u30bf\u30b9\u30af\u3092\u4f55\u554f\u8ffd\u52a0\u3057\u3066\u3082p < 0.05\u306b\u306a\u308a\u307e\u305b\u3093\u3002"
    ),
    "label.watch_candidates_title": "\u5019\u88dc\u306e\u5b9f\u884c\u53ef\u80fd\u6027",
    "label.watch_fit_class": "\u5206\u985e",
    "label.watch_fit_no_weights": "\u4e3b\u8981\u91cd\u307f\u306a\u3057",
    "label.watch_fit_gated": "\u30b2\u30fc\u30c8\u4ed8\u304d",
    "label.watch_fit_role_unknown": "\u5f79\u5272\u4e0d\u660e",
    "label.watch_fit_not_text": "\u30c6\u30ad\u30b9\u30c8\u751f\u6210\u4ee5\u5916",
    "label.watch_fit_too_large": "\u5927\u304d\u3059\u304e\u308b",
    "label.watch_fit_fits": "\u9069\u5408",
    "label.watch_candidate_budget": "\u30d7\u30e9\u30f3\u30ca\u30fc\u4e88\u7b97",
    "warn.model_unknown": "\u6307\u5b9a\u3055\u308c\u305f\u30e2\u30c7\u30eb {model} \u306f\u30ab\u30bf\u30ed\u30b0\u306b\u3042\u308a\u307e\u305b\u3093\u3002",
    "warn.quality_unmeasured": "\u54c1\u8cea\u672a\u6e2c\u5b9a\u306e\u305f\u3081\u3001\u4ee5\u4e0b\u306e\u30e2\u30c7\u30eb\u3092\u81ea\u52d5\u30e9\u30f3\u30ad\u30f3\u30b0\u304b\u3089\u9664\u5916\u3057\u307e\u3057\u305f: {models}{remaining}\u3002nmesh eval \u3067\u6e2c\u5b9a\u3067\u304d\u307e\u3059\u3002",
    "warn.quality_unmeasured_selected": "{model} \u306f\u54c1\u8cea\u672a\u6e2c\u5b9a\u306e\u307e\u307e\u660e\u793a\u7684\u306b\u9078\u629e\u3055\u308c\u305f\u305f\u3081\u3001\u30e9\u30f3\u30ad\u30f3\u30b0\u3067\u306f\u901f\u5ea6\u9805\u306e\u307f\u3092\u4f7f\u7528\u3057\u307e\u3057\u305f\u3002",
    "label.launcher_written": "\u30e9\u30f3\u30c1\u30e3\u30fc\u3092\u66f8\u304d\u8fbc\u307f\u307e\u3057\u305f: {path}",
    "label.gateway_env": "\u30b2\u30fc\u30c8\u30a6\u30a7\u30a4\u74b0\u5883\u30d5\u30a1\u30a4\u30eb: {path}",
    "label.unit_written": "\u30e6\u30cb\u30c3\u30c8\u3092\u66f8\u304d\u8fbc\u307f\u307e\u3057\u305f: {path}",
    "label.autostart_limitation": "\u5236\u7d04: {text}",
    "autostart.windows_limitations": "Windows: /sc onlogon \u306f\u30e6\u30fc\u30b6\u30fc\u304c\u30ed\u30b0\u30aa\u30f3\u3059\u308b\u307e\u3067\u8d77\u52d5\u3057\u307e\u305b\u3093\u3002\u30d6\u30fc\u30c8\u6642\u306e\u8d77\u52d5\u306b\u306f /sc onstart \u3068 SYSTEM \u307e\u305f\u306f\u4fdd\u5b58\u6e08\u307f\u8cc7\u683c\u60c5\u5831\u304c\u5fc5\u8981\u3067\u3059\u3002\u73fe\u5728\u306e\u5358\u7d14\u306a\u30bf\u30b9\u30af \u30b9\u30b1\u30b8\u30e5\u30fc\u30e9\u8a2d\u5b9a\u306b\u306f\u30b2\u30fc\u30c8\u30a6\u30a7\u30a4\u81ea\u8eab\u306e\u30af\u30e9\u30c3\u30b7\u30e5\u518d\u8d77\u52d5\u6a5f\u80fd\u304c\u3042\u308a\u307e\u305b\u3093\u3002",
    "warn.quality_prior": "\u30e2\u30c7\u30eb\u306e\u9806\u4f4d\u4ed8\u3051\u306f\u672a\u691c\u8a3c\u306e\u30ab\u30bf\u30ed\u30b0\u54c1\u8cea\u4e3b\u5f35\u3092\u4f7f\u7528\u3057\u307e\u3059\u3002nmesh eval \u3067\u6e2c\u5b9a\u3067\u304d\u307e\u3059\u3002",
    "note.quality_measured_selected": "{model} \u306e {quant} / {backend} \u306f\u3053\u306e\u30db\u30b9\u30c8\u3067 {suite} \u30b9\u30a4\u30fc\u30c8 {rate} \u3068\u6e2c\u5b9a\u6e08\u307f\u3067\u3001\u3053\u306e\u9078\u629e\u306f\u6e2c\u5b9a\u306b\u57fa\u3065\u3044\u3066\u3044\u307e\u3059\u3002\u672a\u6e2c\u5b9a\u306e\u5019\u88dc\u306f\u30ab\u30bf\u30ed\u30b0\u306e\u4e3b\u5f35\u5024\u3067\u4e26\u3079\u3089\u308c\u3066\u3044\u308b\u3060\u3051\u3067\u3001nmesh eval \u3067\u6e2c\u5b9a\u3057\u306a\u3044\u9650\u308a\u8ffd\u3044\u8d8a\u3057\u307e\u305b\u3093\u3002",
    "warn.quality_contradiction": "{role}: \u5b9f\u6e2c\u5408\u683c\u7387\u3067\u306f {other} ({other_rate:.0%}) \u304c {selected} ({selected_rate:.0%}) \u3088\u308a\u4e0a\u4f4d\u3067\u3059\u304c\u3001\u30ab\u30bf\u30ed\u30b0\u4e3b\u5f35\u306f\u9006\u3067\u3059 ({other_prior} < {selected_prior})\u3002\u6b63\u78ba\u306ap\u5024\u306f {p_value:.4f}\u3001\u6bd4\u8f03\u306f {compared}\u554f\u3067\u3059\u3002",
    "warn.eval_incomparable_conditions": "{model}: \u4fdd\u5b58\u6e08\u307f\u5408\u683c\u7387\u306f\u7570\u306a\u308b\u6761\u4ef6\u3067\u6e2c\u5b9a\u3055\u308c\u305f\u305f\u3081\uff08{left} vs {right}\uff09\u3001\u6bd4\u8f03\u3057\u307e\u305b\u3093\u3002\u6e2c\u5b9a\u6761\u4ef6\u306b\u3088\u3063\u3066\u3053\u306e\u5224\u5b9a\u306f\u5909\u308f\u308a\u307e\u3059\uff08\u5b9f\u969b\u306e\u7d44\u307f\u5408\u308f\u305b\u3067\u306f\u3001\u30d7\u30ed\u30f3\u30d7\u30c8\u30ad\u30e3\u30c3\u30b7\u30e5\u6761\u4ef6\u306b\u3088\u308a\u6b63\u78ba\u306ap\u5024\u304c 0.0352 \u304b\u3089 0.1796 \u306b\u5909\u308f\u308a\u307e\u3057\u305f\uff09\u3002\u540c\u3058 suite\u3001\u63a1\u70b9\u5668\u30c0\u30a4\u30b8\u30a7\u30b9\u30c8\u3001\u63a8\u8ad6\u4e88\u7b97\u3001\u30d7\u30ed\u30f3\u30d7\u30c8\u30ad\u30e3\u30c3\u30b7\u30e5\u6761\u4ef6\u3067 nmesh eval \u3092\u4e21\u65b9\u306b\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "warn.eval_evidence_override": "{role}: \u5b9f\u6e2c\u304c\u30ab\u30bf\u30ed\u30b0\u4e3b\u5f35\u5024\u3092\u4e0a\u56de\u308a\u307e\u3057\u305f \u2014 {other} ({other_rate:.0%}) \u304c {selected} ({selected_rate:.0%}) \u3092\u4e0a\u56de\u308a\u3001\u6b63\u78ba\u306ap\u5024\u306f {p_value:.4f}\uff08\u6bd4\u8f03 {compared}\u554f\uff09\u3067\u3059\u3002\u4e3b\u5f35\u5024\u306f\u9006\u9806\uff08{other_prior} < {selected_prior}\uff09\u3067\u3059\u304c\u3001\u8a08\u753b\u306b\u306f {other} \u3092\u4f7f\u3044\u307e\u3059\u3002\u3069\u3061\u3089\u306e\u5408\u683c\u7387\u3082\u8a08\u753b\u3055\u308c\u305f\u91cf\u5b50\u5316\u30fb\u30d0\u30c3\u30af\u30a8\u30f3\u30c9\u3067\u540c\u4e00\u306e\u63a1\u70b9\u898f\u5247\u306b\u3088\u308a\u6e2c\u5b9a\u3055\u308c\u305f\u3082\u306e\u3067\u3059\u3002\u4e3b\u5f35\u5024\u3067\u9806\u4f4d\u4ed8\u3051\u3059\u308b\u306b\u306f --ignore-eval-evidence \u3092\u6307\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "warn.eval_evidence_override_quant": "{role}: \u6e2c\u5b9a\u304c\u91cf\u5b50\u5316\u30da\u30ca\u30eb\u30c6\u30a3\u3092\u4e0a\u56de\u308a\u307e\u3057\u305f \u2014 {model} \u306e {other_quant} ({other_rate:.0%}) \u304c {selected_quant} ({selected_rate:.0%}) \u3092\u4e0a\u56de\u308a\u3001\u6b63\u78ba\u306ap\u5024\u306f {p_value:.4f}\uff08\u6bd4\u8f03 {compared}\u554f\uff09\u3067\u3059\u3002QUANT_PENALTY \u3067\u306f {other_quant} \u306e\u65b9\u304c\u4f4e\u304f\u8a55\u4fa1\u3055\u308c\u307e\u3059\u304c\uff08{other_penalty} > {selected_penalty}\uff09\u3001{other_quant} \u3092\u8a08\u753b\u3057\u307e\u3059\u3002\u4e21\u65b9\u306e\u5408\u683c\u7387\u306f\u540c\u3058\u30d0\u30c3\u30af\u30a8\u30f3\u30c9\u3067\u540c\u4e00\u306e\u63a1\u70b9\u898f\u5247\u306b\u3088\u308a\u6e2c\u5b9a\u3055\u308c\u307e\u3057\u305f\u3002\u30da\u30ca\u30eb\u30c6\u30a3\u3067\u9806\u4f4d\u4ed8\u3051\u3059\u308b\u306b\u306f --ignore-eval-evidence \u3092\u6307\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "warn.quant_penalty_contradiction": "{role}: \u6e2c\u5b9a\u5408\u683c\u7387\u3067\u306f {model} \u306e {other_quant} ({other_rate:.0%}) \u304c {selected_quant} ({selected_rate:.0%}) \u3088\u308a\u4e0a\u4f4d\u3067\u3059\u304c\u3001QUANT_PENALTY \u306f\u9006\u306e\u9806\u4f4d\u3067\u3059\uff08{other_penalty} > {selected_penalty}\uff09\u3002\u6b63\u78ba\u306ap\u5024\u306f {p_value:.4f}\u3001\u6bd4\u8f03\u306f {compared}\u554f\u3067\u3059\u3002\u3053\u306e\u30da\u30ca\u30eb\u30c6\u30a3\u8868\u306f\u672a\u6e2c\u5b9a\u306e\u63a8\u5b9a\u3067\u3059\u3002",
    "note.quant_indistinguishable": "{role}: {model} \u306e {other_quant} \u306f {selected_quant} \u306b\u5bfe\u3057\u3066 {other_rate:.0%}\u3001{selected_quant} \u306f {selected_rate:.0%}\u3067\u3057\u305f\uff08{compared}\u554f\u3001\u6b63\u78ba\u306ap={p_value:.4f}\uff09\u3002\u3053\u306e\u30b9\u30a4\u30fc\u30c8\u3067\u306f\u4e21\u8005\u3092\u533a\u5225\u3067\u304d\u307e\u305b\u3093\u304c\u3001QUANT_PENALTY\u306f {penalty_gap}\u30dd\u30a4\u30f3\u30c8\u306e\u5dee\u3092\u8a2d\u3051\u3066\u3044\u307e\u3059\uff08{selected_penalty}\u5bfe{other_penalty}\uff09\u3002\u3053\u306e\u6bd4\u8f03\u3067\u89e3\u50cf\u3067\u304d\u308b\u6700\u5c0f\u5dee\u306f {minimum:.1%}\u3067\u3042\u308a\u3001\u3053\u306e2\u3064\u306e\u91cf\u5b50\u5316\u306e\u9806\u4f4d\u306f\u78ba\u8a8d\u3082\u53cd\u8a3c\u3082\u3067\u304d\u307e\u305b\u3093\u3002",
    "note.eval_underpowered": "{tasks}\u554f\u306e\u30b9\u30a4\u30fc\u30c8\u3067\u306f\u3001\u3053\u306e\u89b3\u6e2c\u30e9\u30f3\u30ad\u30f3\u30b0\u3092\u89e3\u6c7a\u3067\u304d\u307e\u305b\u3093: {other_rate:.1%} \u5bfe {selected_rate:.1%}\uff08\u6b63\u78ba\u306ap\u5024={p_value:.4f}\uff09\u3002\u03b1=0.05\u3067\u306e\u6700\u5c0f\u89e3\u6c7a\u53ef\u80fd\u5dee\u306f {minimum:.1%}\u3067\u3042\u308a\u3001\u3053\u306e\u30b5\u30a4\u30ba\u3067\u30e9\u30f3\u30ad\u30f3\u30b0\u306f\u78ba\u8a8d\u3082\u53cd\u8a3c\u3082\u3067\u304d\u307e\u305b\u3093\u3002",
    "warn.bench_excluded": "{model} {quant}: \u5b9f\u6e2c\u30b9\u30eb\u30fc\u30d7\u30c3\u30c8 ({tps} tok/s) \u304c {threshold} tok/s \u306e\u95be\u5024\u3092\u4e0b\u56de\u3063\u305f\u305f\u3081\u3001\u3053\u306e\u30e2\u30c7\u30eb\u306f\u9664\u5916\u3055\u308c\u307e\u3057\u305f\u3002\u63a8\u5b9a\u3067\u306f\u8a31\u53ef\u3055\u308c\u307e\u3059\u3002\u5b9f\u884c\u56de\u6570\u3092\u5897\u3084\u3057\u3066\u518d\u30d9\u30f3\u30c1\u30de\u30fc\u30af\u3059\u308b\u3068\u30d7\u30e9\u30f3\u304c\u5909\u308f\u308b\u53ef\u80fd\u6027\u304c\u3042\u308a\u307e\u3059\u3002",
    "warn.bench_reproducibility": "\u3053\u306e\u30de\u30b7\u30f3\u3067\u306f\u5b9f\u6e2c\u30b9\u30eb\u30fc\u30d7\u30c3\u30c8\u3092\u518d\u73fe\u3067\u304d\u307e\u305b\u3093\uff08\u6700\u5c0f {minimum:.2f}\u3001\u6700\u5927 {maximum:.2f}\u3001\u30b9\u30d7\u30ec\u30c3\u30c9 {spread:.1%}\uff09\u3002\u30d7\u30e9\u30f3\u30ca\u30fc\u306f\u30e1\u30c7\u30a3\u30a2\u30f3\u3092\u4e8b\u5b9f\u3068\u3057\u3066\u6271\u3044\u307e\u3059\u3002--runs \u3092\u5897\u3084\u3057\u3066\u518d\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "warn.bench_control": "ベンチマークのパスが一致しませんでした（制御比率 {ratio}）。このホストは安定していないため、{kept}。",
    "warn.bench_no_control": "制御パスが実行されていないため、この測定値は表示しますが証拠として保存しません。",
    "warn.bench_unconfirmed": "{model} {quant}: 測定スループット（{tps} tok/s）は {threshold} tok/s 未満ですが、除外には一致する測定がもう1回必要です。",
    "warn.bench_epoch": "ホストが過去に観測した基準速度の {ratio:.0%} の状態で測定されたため、{kept}。",
    "warn.bench_no_reference": "基準ワークロードを利用できないため、この測定値をエポック間で比較できません。",
    "warn.bench_demoted": "このホストが測定時より明らかに高速になったため、保存済みのベンチマーク測定 {count} 件を無効化しました。証拠を再測定してください。",
    "warn.bench_harness_mismatch": "保存済みベンチマーク測定 {count} 件は旧 harness のため証拠として使いません。nmesh bench で再測定してください。",
    "warn.bench_decode_unmeasurable": "bench は {requested} デコードトークンを要求しましたが、モデルが実際に返したのは {served} トークンでした。デコード速度には少なくとも {minimum} 個の提供トークンが必要です（llama.cpp の predicted_ms はプレフィル後のステップを表すためです）。そのためデコード測定値は記録しませんでした。",
    "warn.bench_decode_short": "bench は {requested} デコードトークンを要求しましたが、モデルは {served} トークンで停止しました。max_tokens は上限値なので、記録したデコード速度は {requested} トークン時ではなく {served} トークン時の速度です。",
    "warn.orchestrate_degraded": "\u59d4\u8b72\u306e\u6642\u9593\u6e2c\u5b9a\u306f\u52a3\u5316\u3057\u305f\u30db\u30b9\u30c8\u30a8\u30dd\u30c3\u30af\u3067\u884c\u308f\u308c\u307e\u3057\u305f\u3002\u54c1\u8cea\u30b2\u30fc\u30c8\u306f\u6709\u52b9\u3067\u3059\u304c\u3001\u30b3\u30b9\u30c8\u8a3c\u62e0\u306f\u53e4\u304f\u306a\u3063\u3066\u3044\u307e\u3059\u3002",
    "warn.orchestrate_demoted": "\u3053\u306e\u30db\u30b9\u30c8\u304c\u6e2c\u5b9a\u6642\u3088\u308a\u9ad8\u901f\u306b\u306a\u3063\u305f\u305f\u3081\u3001\u59d4\u8b72\u306e\u4fdd\u5b58\u6e08\u307f\u6e2c\u5b9a {count} \u4ef6\u3092\u7121\u52b9\u5316\u3057\u307e\u3057\u305f\u3002\u30b3\u30b9\u30c8\u8a3c\u62e0\u3092\u518d\u6e2c\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "warn.bench_orchestrate_demoted": "\u5171\u6709\u53c2\u7167\u30a8\u30dd\u30c3\u30af\u304c\u9ad8\u901f\u306b\u306a\u3063\u305f\u305f\u3081\u3001\u59d4\u8b72\u306e\u4fdd\u5b58\u6e08\u307f\u6e2c\u5b9a {count} \u4ef6\u3092\u7121\u52b9\u5316\u3057\u307e\u3057\u305f\u3002\u30b3\u30b9\u30c8\u8a3c\u62e0\u3092\u518d\u6e2c\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "warn.spec_degraded": "\u30db\u30b9\u30c8\u304c\u52a3\u5316\u3057\u305f\u30a8\u30dd\u30c3\u30af\u3067\u6295\u6a5f\u3092\u6e2c\u5b9a\u3057\u307e\u3057\u305f\u3002\u8a18\u9332\u306f\u4fdd\u5b58\u3057\u307e\u3059\u304c\u3001\u901f\u5ea6\u5411\u4e0a\u306e\u8a3c\u62e0\u3068\u3057\u3066\u306f\u4f7f\u3048\u307e\u305b\u3093\u3002",
    "warn.spec_demoted": "\u3053\u306e\u30db\u30b9\u30c8\u304c\u6e2c\u5b9a\u6642\u3088\u308a\u660e\u3089\u304b\u306b\u9ad8\u901f\u306b\u306a\u3063\u305f\u305f\u3081\u3001\u6295\u6a5f\u306e\u4fdd\u5b58\u6e08\u307f\u6e2c\u5b9a {count} \u4ef6\u3092\u7121\u52b9\u5316\u3057\u307e\u3057\u305f\u3002\u518d\u6e2c\u5b9a\u304c\u5fc5\u8981\u3067\u3059\u3002",
    "warn.bench_spec_demoted": "\u5171\u6709\u53c2\u7167\u30a8\u30dd\u30c3\u30af\u304c\u9ad8\u901f\u306b\u306a\u3063\u305f\u305f\u3081\u3001\u6295\u6a5f\u306e\u4fdd\u5b58\u6e08\u307f\u6e2c\u5b9a {count} \u4ef6\u3092\u7121\u52b9\u5316\u3057\u307e\u3057\u305f\u3002\u518d\u6e2c\u5b9a\u304c\u5fc5\u8981\u3067\u3059\u3002",
    "err.spec_transport": "\u6295\u6a5f\u306e\u6e2c\u5b9a\u306f\u8ee2\u9001\u30ec\u30d9\u30eb\u3067\u5b8c\u4e86\u3057\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u8a18\u9332\u306f\u884c\u308f\u308c\u3066\u3044\u307e\u305b\u3093\u3002",
    "label.decode_range": "\u30c7\u30b3\u30fc\u30c9\u7bc4\u56f2: {minimum:.2f}\u2013{maximum:.2f} tok/s\uff08\u30b9\u30d7\u30ec\u30c3\u30c9 {spread:.1%}\uff09",
    "label.bench_passes": "ベンチマークパス（{passes}）: {values} tok/s",
    "label.bench_control": "制御比率: {ratio}",
    "label.bench_kept": "以前の証拠を保持しました",
    "label.bench_nothing_stored": "証拠は保存しませんでした",
    "err.eval_up": "\u30b5\u30fc\u30d3\u30b9\u304c\u8d77\u52d5\u3057\u3066\u3044\u307e\u305b\u3093\u3002\u5148\u306b nmesh up \u3092\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "err.eval_run": "\u8a55\u4fa1\u306b\u5931\u6557\u3057\u307e\u3057\u305f: {error}",
    "err.eval_save": "\u8a55\u4fa1\u306e\u4fdd\u5b58\u306b\u5931\u6557\u3057\u307e\u3057\u305f: {error}",
    "err.eval_categories": "\u6307\u5b9a\u3057\u305f\u30ab\u30c6\u30b4\u30ea\u306b\u4e00\u81f4\u3059\u308b\u8a55\u4fa1\u30bf\u30b9\u30af\u304c\u3042\u308a\u307e\u305b\u3093\u3002",
    "err.eval_unknown_categories": "\u3053\u306e\u30b9\u30a4\u30fc\u30c8\u306b\u5b58\u5728\u3057\u306a\u3044\u30ab\u30c6\u30b4\u30ea: {categories}",
    "note.gpu_pinned": "{service}: GPU {indices} \u306b\u30d4\u30f3\u7559\u3081\u3057\u307e\u3057\u305f\uff08\u63a8\u5b9a\u914d\u7f6e\u30fb\u30c7\u30d0\u30a4\u30b9\u53ef\u8996\u6027\u3067\u5206\u96e2\uff09",
    "note.tps_estimate": "~ = \u30ab\u30bf\u30ed\u30b0\u4e0a\u306e\u63a8\u5b9a\u5024\u3067\u3001\u3053\u306e\u30de\u30b7\u30f3\u3067\u306e\u5b9f\u6e2c\u5024\u3067\u306f\u3042\u308a\u307e\u305b\u3093\u3002`nmesh bench` \u3067\u5b9f\u6e2c\u5024\u306b\u7f6e\u304d\u63db\u308f\u308a\u307e\u3059\u3002",
    "note.eval_scope": "\u5b9f\u65bd\u3059\u308b{tasks}\u554f\u306e\u6c7a\u5b9a\u7684\u30de\u30a4\u30af\u30ed\u8a55\u4fa1\u3067\u3059\uff08\u6307\u793a\u8ffd\u5f93\u30fb\u51fa\u529b\u5f62\u5f0f\u30fb\u62bd\u51fa\u30fb\u7ffb\u8a33\u65b9\u5411\uff09\u3002\u77e5\u8b58\u30d9\u30f3\u30c1\u30de\u30fc\u30af\u3067\u306f\u306a\u304f\u3001MMLU\u578b\u30b9\u30b3\u30a2\u3068\u6bd4\u8f03\u3057\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002",
    "note.eval_value_vs_discipline": "\u4fa1\u5024\u30c1\u30a7\u30c3\u30af\u306e\u3042\u308b{checked}\u4ef6\u306e\u5931\u6557\u306e\u3046\u3061\u3001{value_only}\u4ef6\u306f\u6b63\u3057\u3044\u5024\u3092\u51fa\u3057\u306a\u304c\u3089\u5fc5\u8981\u306a\u51fa\u529b\u5f62\u5f0f\u3060\u3051\u306b\u9055\u53cd\u3057\u307e\u3057\u305f\u3002\u3053\u308c\u3089\u306e\u5931\u6557\u306f\u80fd\u529b\u3067\u306f\u306a\u304f\u3001\u51fa\u529b\u898f\u5f8b\u3092\u6e2c\u3063\u3066\u3044\u307e\u3059\u3002",
    "note.eval_failure_kinds": "{failures}\u4ef6\u306e\u5931\u6557\uff1a\u6b63\u3057\u3044\u7b54\u3048\u3067\u306f\u306a\u3044\u3082\u306e\u304c{value}\u4ef6\u3001\u6b63\u3057\u3044\u7b54\u3048\u3067\u51fa\u529b\u5f62\u5f0f\u3060\u3051\u304c\u7570\u306a\u308b\u3082\u306e\u304c{form}\u4ef6\u3067\u3059\u3002\u51fa\u529b\u5f62\u5f0f\u306e\u5931\u6557\u306f\u51fa\u529b\u3092\u5236\u7d04\u3059\u308b\u3053\u3068\u3067\u6539\u5584\u3067\u304d\u307e\u3059\u304c\u3001\u8aa4\u7b54\u306f\u6539\u5584\u3067\u304d\u307e\u305b\u3093\u3002",
    "note.eval_uncertainty": "全体合格率の95% Wilson区間: {lo:.1%}–{hi:.1%}。この{tasks}問のスイートは、正確検定（α=0.05）で{minimum:.1%}未満の合格率差を解決できません。",
    "warn.eval_config_mismatch": "{model}: \u8a55\u4fa1\u5408\u683c\u7387\u306f\u5b58\u5728\u3057\u307e\u3059\u304c\u3001\u8a08\u753b\u3055\u308c\u305f\u69cb\u6210\uff08{quant}\u3001{backend}\uff09\u306e\u3082\u306e\u3067\u306f\u3042\u308a\u307e\u305b\u3093\u3002\u6e2c\u5b9a\u5408\u683c\u7387\u306f\u91cf\u5b50\u5316\u65b9\u6cd5\u3084\u30d0\u30c3\u30af\u30a8\u30f3\u30c9\u9593\u3067\u79fb\u690d\u3067\u304d\u307e\u305b\u3093\u3002\u5b9f\u884c\u4e2d\u306e\u30b5\u30fc\u30d3\u30b9\u306b\u5bfe\u3057\u3066 nmesh eval \u3092\u518d\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "warn.context_unmeasured": "{model} ({quant} / {backend}): KV\u30ad\u30e3\u30c3\u30b7\u30e5\u304c\u53ce\u307e\u308b\u305f\u3081\u3001\u8a08\u753b\u306f\u30b3\u30f3\u30c6\u30ad\u30b9\u30c8 {context} \u3092\u63b2\u8f09\u3057\u307e\u3059\u304c\u3001\u54c1\u8cea\u306e\u8a55\u4fa1\u8a3c\u62e0\u306f\u5b9f\u30d7\u30ed\u30f3\u30d7\u30c8\u30c8\u30fc\u30af\u30f3 {depth} \u307e\u3067\u3067\u3059\u3002\u6e2c\u5b9a\u3057\u305f\u30db\u30b9\u30c8\u3068\u30a2\u30fc\u30c6\u30a3\u30d5\u30a1\u30af\u30c8\u3067\u306f\u30b7\u30f3\u30b0\u30eb\u30cb\u30fc\u30c9\u53d6\u5f97\u304c\u7d04 15k \u30d7\u30ed\u30f3\u30d7\u30c8\u30c8\u30fc\u30af\u30f3\u307e\u3067\u7dad\u6301\u3055\u308c\u307e\u3057\u305f\u3002\u3053\u308c\u306f\u65e2\u77e5\u306e\u5931\u6557\u3067\u306f\u306a\u304f\u3001\u8a3c\u62e0\u306e\u7a7a\u767d\u3067\u3059\u3002",
    "warn.context_depth_lost": "{model} \u306e {quant} / {backend} \u3067\u3001{family} \u30d7\u30ed\u30fc\u30d6\u306f\u6d45\u3044\u30b3\u30f3\u30c8\u30ed\u30fc\u30eb\u306b\u5408\u683c\u3057\u307e\u3057\u305f\u304c\u3001\u5b9f\u30d7\u30ed\u30f3\u30d7\u30c8\u30c8\u30fc\u30af\u30f3 {depth} \u3067\u306f {passed}/{of} \u306b\u3068\u3069\u307e\u308a\u307e\u3057\u305f\u3002\u3053\u308c\u306f\u8ab2\u984c\u96e3\u6613\u5ea6\u3067\u306f\u306a\u304f\u3001\u6e2c\u5b9a\u3055\u308c\u305f\u6df1\u5ea6\u5931\u6557\u3067\u3059\u3002",
    "warn.context_depth_broken": "{model} \u306e {quant} / {backend} \u3067\u3001\u6d45\u3044\u30b3\u30f3\u30c8\u30ed\u30fc\u30eb\u306b\u5408\u683c\u3057\u305f\u6587\u8108\u30d7\u30ed\u30fc\u30d6\u304c\u5b9f\u30d7\u30ed\u30f3\u30d7\u30c8\u30c8\u30fc\u30af\u30f3 {depth} \u3067\u5931\u6557\u3057\u305f\u305f\u3081\u3001\u8a08\u753b\u3057\u305f\u30b3\u30f3\u30c6\u30ad\u30b9\u30c8 {context} \u306f\u672a\u6e2c\u5b9a\u3067\u306f\u306a\u304f\u3001\u6e2c\u5b9a\u4e0a\u58ca\u308c\u3066\u3044\u307e\u3059\u3002",
    "note.context_probe_uncontrolled": "{families} \u306e\u6587\u8108\u30d7\u30ed\u30fc\u30d6\u30d5\u30a1\u30df\u30ea\u30fc\u306f\u6d45\u3044\u30b3\u30f3\u30c8\u30ed\u30fc\u30eb\u306b\u5931\u6557\u3057\u305f\u305f\u3081\u3001\u6df1\u5ea6\u306e\u7d50\u679c\u304b\u3089\u306f\u6df1\u5ea6\u306b\u3064\u3044\u3066\u4f55\u3082\u5224\u5b9a\u3067\u304d\u307e\u305b\u3093\u3002",
    "note.eval_config": "\u3053\u306e\u5408\u683c\u7387\u306f {model} \u306e {quant} \u3092 {backend} \u3067\u5b9f\u884c\u3057\u305f\u5834\u5408\u306b\u306e\u307f\u9069\u7528\u3055\u308c\u307e\u3059\u3002\u6e2c\u5b9a\u7d50\u679c: \u540c\u3058\u30e2\u30c7\u30eb\u3067\u540c\u3058 Q4_K_M \u30e9\u30d9\u30eb\u3067\u3082\u3001llama.cpp \u306f 12/16\u3001Ollama \u306f 13/16 \u3067\u3057\u305f\u3002",
    "note.eval_divergence": "{config}: \u305d\u306e\u69cb\u6210\u306e\u5408\u683c\u7387\u306f {other_rate}\u3001\u3053\u3061\u3089\u306f {rate} \u3067\u3059\u3002\u6bd4\u8f03\u3057\u305f {compared}\u554f\u306e\u3046\u3061 {count}\u554f\u3067\u5224\u5b9a\u304c\u98df\u3044\u9055\u3044\u307e\u3057\u305f\uff08{ids}\uff09\u3002\u5b9f\u6e2c\u3067\u306f Qwen2.5 0.5B \u306e\u540c\u3058 fp16 \u69cb\u6210\u304c\u4e21\u65b9\u3068\u3082 9/16 \u3067\u3042\u308a\u306a\u304c\u3089 2\u554f\u3067\u9006\u65b9\u5411\u306b\u98df\u3044\u9055\u3044\u3001\u5408\u683c\u7387\u304c\u540c\u3058\u3067\u3082\u6319\u52d5\u304c\u540c\u7b49\u3068\u306f\u9650\u308a\u307e\u305b\u3093\u3002",
    "warn.eval_unscorable": "{tasks}\u554f\u306e\u3046\u3061 {count}\u554f\u304c\u3001\u51fa\u529b\u4e88\u7b97\u3092\u4f7f\u3044\u5207\u308b\u307e\u3067\u306b\u56de\u7b54\u672c\u6587\u3092\u8fd4\u3057\u307e\u305b\u3093\u3067\u3057\u305f\uff08finish_reason=length \u3067 content \u304c\u7a7a\uff09\u3002\u3053\u306e\u5b9f\u884c\u306f\u30e2\u30c7\u30eb\u3067\u306a\u304f\u4e88\u7b97\u3092\u6e2c\u3063\u3066\u3044\u308b\u306e\u3067\u3001\u8a08\u753b\u306e\u6839\u62e0\u306b\u306f\u4f7f\u3044\u307e\u305b\u3093\u3002\u5b9f\u6e2c: gemma-4-26B-A4B \u306f\u30b9\u30a4\u30fc\u30c8\u4e88\u7b97\u3067 104\u554f\u5168\u3066\u304c\u7a7a\u3001512\u30c8\u30fc\u30af\u30f3\u3067\u306f\u6b63\u3057\u304f\u7b54\u3048\u307e\u3057\u305f\u3002\u73fe\u5728\u306e allowance: {allowance} \u30c8\u30fc\u30af\u30f3\u3002`nmesh eval --reasoning-allowance N` \u3067\u5897\u3084\u305b\u307e\u3059\u3002",
    "warn.eval_transport": "{count} \u4ef6\u306e\u30bf\u30b9\u30af\u304c\u8ee2\u9001\u30ec\u30d9\u30eb\uff08\u30bf\u30a4\u30e0\u30a2\u30a6\u30c8\u307e\u305f\u306fHTTP\u30a8\u30e9\u30fc\uff09\u3067\u5fdc\u7b54\u3057\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u3053\u308c\u306f\u30e2\u30c7\u30eb\u306e\u8aa4\u7b54\u3067\u306f\u306a\u304f\u30db\u30b9\u30c8\u5074\u306e\u5931\u6557\u306a\u306e\u3067\u3001\u3053\u306e\u5b9f\u884c\u306f\u8a18\u9332\u3057\u307e\u3059\u304c\u54c1\u8cea\u306e\u8a3c\u62e0\u306b\u306f\u4f7f\u3044\u307e\u305b\u3093\u3002",
    "evidence.bench_title": "保存済みベンチマーク証拠",
    "evidence.eval_title": "保存済み評価証拠",
    "evidence.depth_title": "保存済み深度証拠",
    "evidence.depth_json_hint": "深度の系統内訳は nmesh evidence --json で確認できます。",
    "evidence.empty": "保存済み証拠はありません。",
    "evidence.column.kind": "種別",
    "evidence.column.model": "モデル",
    "evidence.column.quant": "量子化",
    "evidence.column.backend": "バックエンド",
    "evidence.column.scope": "範囲",
    "evidence.column.value": "値",
    "evidence.column.usable": "使用可",
    "evidence.column.reasons": "理由",
    "evidence.column.remeasure": "再測定",
    "evidence.reason.harness_mismatch": "旧 benchmark harness で保存された記録です",
    "evidence.reason.unstable": "統制ベンチマークが安定しませんでした",
    "evidence.reason.epoch_degraded": "劣化したホストエポックで測定された記録です",
    "evidence.reason.unconfirmed": "確認用のベンチマークセッションが2回未満です",
    "evidence.reason.suite_unknown": "保存された評価 suite は不明です",
    "evidence.reason.grader_digest_mismatch": "古い採点規則で保存された評価です",
    "evidence.reason.partial_suite": "絞り込んだ一部のみ実行されています — planner が使える証拠にするには全スイートを実行してください",
    "evidence.reason.unscorable": "採点可能な回答がない評価タスクがあります",
    "evidence.reason.transport_errors": "転送レベルで失敗した評価タスクがあります",
    "evidence.reason.depth_scoped": "深い評価は浅いプランナー比較には使いません",
    "evidence.reason.probe_digest_mismatch": "古いプローブ規則で保存された深度記録です",
    "evidence.reason.control_failed": "浅い制御が深度証拠の帰属を確立しませんでした",
    "evidence.reason.depth_lost": "帰属可能なタスクファミリーが深度測定で失敗しました",
    "evidence.reason.superseded": "同じ構成のより新しい有効な記録を使用しています",
    "evidence.retrieval_title": "保存済み検索証拠",
    "evidence.spec_title": "保存済み投機実行証拠",
    "evidence.reason.spec_unstable": "この機で投機実行の制御が再現されませんでした",
    "evidence.reason.spec_stale": "より高速なホストエポックにより投機実行の速度主張が無効化されました",
    "evidence.reason.spec_not_identical": "投機実行が出力トークンを変えたため、無料の高速化ではありません",
    "evidence.reason.spec_not_faster": "どの負荷クラスも投機実行の代償を払うほど高速化しませんでした",
    "evidence.reason.spec_mixed": "高速化と低速化が混在し、要求の負荷配分が不明なため判断できません",
    "evidence.reason.spec_no_evidence": "この投機実行設定では負荷クラスが測定されていません",
    "warn.eval_artifact_changed": "{model} \u306e {quant} \u3092 {backend} \u3067\u5b9f\u884c\u3057\u305f\u7d50\u679c\u306e\u30ad\u30e3\u30c3\u30b7\u30e5\u6e08\u307f\u5408\u683c\u7387\u306f\u6210\u679c\u7269 {previous} \u3067\u6e2c\u5b9a\u3055\u308c\u307e\u3057\u305f\u304c\u3001\u73fe\u5728\u306e\u30b5\u30fc\u30d3\u30b9\u306f {current} \u3092\u8aad\u307f\u8fbc\u3093\u3067\u3044\u307e\u3059\u3002\u540c\u3058\u30e2\u30c7\u30eb\u3068\u91cf\u5b50\u5316\u30e9\u30d9\u30eb\u304c\u7570\u306a\u308b\u91cd\u307f\u30d5\u30a1\u30a4\u30eb\u3092\u6307\u3057\u5f97\u308b\u305f\u3081\u3001\u305d\u306e\u5408\u683c\u7387\u306f\u3053\u306e\u6210\u679c\u7269\u3092\u8aac\u660e\u3057\u307e\u305b\u3093\u3002",
    "warn.eval_stale_grader": "{model} \u306e {quant} \u3092 {backend} \u3067\u5b9f\u884c\u3057\u305f suite {suite} \u306e\u8a18\u9332\u306f\u3001\u7570\u306a\u308b\u8a55\u4fa1\u30eb\u30fc\u30eb\u30d0\u30fc\u30b8\u30e7\u30f3\u3067\u5224\u5b9a\u3055\u308c\u305f\u305f\u3081\u3001\u8a08\u753b\u306b\u306f\u4f7f\u7528\u3057\u307e\u305b\u3093\u3002",
    "note.watch_external_claim": "\u5916\u90e8\u6295\u7a3f\u306f\u8a3c\u62e0\u3067\u306f\u306a\u304f\u53c2\u7167\u5148\u3067\u3059\u3002\u767a\u898b\u306f\u691c\u7d22\u53ef\u80fd\u306a\u6839\u62e0\u3067\u78ba\u8a8d\u3057\u307e\u3059\u3002",
    "note.watch_route": "\u30eb\u30fc\u30c8\u306e\u767a\u898b\u306f\u3053\u306e\u30b2\u30fc\u30c8\u30a6\u30a7\u30a4\u306b\u9650\u5b9a\u3055\u308c\u307e\u3059\u3002\u7b2c\u4e09\u8005\u30a2\u30d7\u30ea\u306e\u30eb\u30fc\u30c8\u306f nmesh \u306b\u5b9f\u88c5\u3055\u308c\u3066\u3044\u306a\u3044\u53ef\u80fd\u6027\u304c\u3042\u308a\u307e\u3059\u3002",
    "note.watch_zenn_body": "Zenn RSS \u306e\u8981\u7d04\u306b\u306f\u6280\u8853\u5185\u5bb9\u304c\u306a\u3044\u305f\u3081\u3001\u3053\u306e\u5b9f\u884c\u3067\u306f\u8a18\u4e8b API \u3068 HTML \u304b\u3089\u672c\u6587\u3092\u53d6\u5f97\u3057\u307e\u3057\u305f\u3002",
    "note.watch_caps_unavailable": "\u8a18\u9332\u6e08\u307f\u306e llama.cpp \u80fd\u529b\u60c5\u5831\u304c\u306a\u3044\u305f\u3081\u3001\u30d5\u30e9\u30b0\u6bd4\u8f03\u306f\u5b9f\u884c\u3057\u3066\u3044\u307e\u305b\u3093\u3002",
    "note.watch_unknown_flag": "\u672a\u77e5\u306e\u30d5\u30e9\u30b0\u306f\u6a5f\u80fd\u304c\u6b20\u843d\u3057\u3066\u3044\u308b\u8a3c\u62e0\u3067\u306f\u3042\u308a\u307e\u305b\u3093\u3002\u5225\u306e\u30d0\u30c3\u30af\u30a8\u30f3\u30c9\u3084\u30d0\u30fc\u30b8\u30e7\u30f3\u306e\u53ef\u80fd\u6027\u304c\u3042\u308a\u307e\u3059\u3002",
    "warn.watch_unreachable": "{source} \u306b\u5230\u9054\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f: {detail}",
    "warn.watch_state": "\u76e3\u8996\u72b6\u614b\u3092\u4fdd\u5b58\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f: {error}",
    "warn.watch_draft": "\u76e3\u8996\u30c9\u30e9\u30d5\u30c8\u3092\u66f8\u304d\u8fbc\u3081\u307e\u305b\u3093\u3067\u3057\u305f: {error}",
    "err.watch_run": "\u76e3\u8996\u306b\u5931\u6557\u3057\u307e\u3057\u305f: {error}",
    "label.watch_title": "nmesh watch",
    "label.watch_source": "\u30bd\u30fc\u30b9",
    "label.watch_reachable": "\u5230\u9054\u53ef\u80fd",
    "label.watch_items": "\u30a2\u30a4\u30c6\u30e0",
    "label.watch_detail": "\u8a73\u7d30",
    "label.watch_finding": "{kind}: {value} ({mentions} \u4ef6\u306e\u8a00\u53ca)",
    "label.watch_catalog_title": "\u30ab\u30bf\u30ed\u30b0\u30ab\u30d0\u30ec\u30c3\u30b8",
    "label.watch_metric": "\u6307\u6a19",
    "label.watch_value": "\u5024",
    "label.watch_catalog_entries": "\u30ab\u30bf\u30ed\u30b0\u30a8\u30f3\u30c8\u30ea\u6570",
    "label.watch_catalog_repo_ids": "\u30ab\u30bf\u30ed\u30b0\u306e\u30ea\u30dd\u30b8\u30c8\u30ea ID \u6570",
    "label.watch_catalog_mentioned": "\u8a00\u53ca\u3055\u308c\u305f\u30ea\u30dd\u30b8\u30c8\u30ea ID \u6570",
    "label.watch_catalog_in_catalog": "\u30ab\u30bf\u30ed\u30b0\u306b\u65e2\u306b\u3042\u308b\u8a00\u53ca ID \u6570",
    "label.watch_catalog_resolved": "\u89e3\u6c7a\u3057\u305f\u30ea\u30dd\u30b8\u30c8\u30ea ID \u6570",
    "label.watch_catalog_absent": "\u30ab\u30bf\u30ed\u30b0\u306b\u306a\u3044\u89e3\u6c7a\u6e08\u307f ID \u6570",
    "label.watch_filename": "\u30d5\u30a1\u30a4\u30eb\u540d",
    "label.watch_install": "\u30a4\u30f3\u30b9\u30c8\u30fc\u30eb\u65b9\u6cd5",
    "label.eval_category": "\u30ab\u30c6\u30b4\u30ea",
    "label.eval_passed": "\u5408\u683c",
    "label.eval_total": "\u5168\u4f53",
    "label.eval_pass_rate": "\u5408\u683c\u7387",
    "label.eval_overall": "\u5168\u4f53\u5408\u683c\u7387: {passed}/{total} ({rate:.1%})",
    "label.eval_failed": "\u5931\u6557\u30bf\u30b9\u30afID: {ids}",
    "label.eval_title": "nmesh eval",
    "label.eval_depth": "\u8981\u6c42\u30d7\u30ed\u30f3\u30d7\u30c8\u6df1\u5ea6: {requested}\uff1b\u5b9f\u969b\u306e\u6df1\u5ea6: {served}",
    "label.eval_context_probe": "\u6587\u8108\u30d7\u30ed\u30fc\u30d6: {passed}/{total} (\u30ea\u30c6\u30e9\u30eb {literal_passed}/{literal_total}\u3001\u30e9\u30c6\u30f3\u30c8 {latent_passed}/{latent_total}\u3001\u8907\u6570\u30b3\u30fc\u30c9 {multi_passed}/{multi_total})",
    "label.eval_context_control": "\u6587\u8108\u30d7\u30ed\u30fc\u30d6\u30b3\u30f3\u30c8\u30ed\u30fc\u30eb: {passed}/{total}",
    "err.orchestrate_plan": "\u5b9f\u884c\u53ef\u80fd\u306a\u30b5\u30fc\u30d3\u30b9\u3092\u542b\u3080\u4fdd\u5b58\u6e08\u307f\u8a08\u753b\u304c\u3042\u308a\u307e\u305b\u3093\u3002",
    "err.orchestrate_service": "\u8a08\u753b\u304b\u3089 lead {lead} \u3068 worker {worker} \u3092\u89e3\u6c7a\u3067\u304d\u307e\u305b\u3093\u3002",
    "err.orchestrate_no_worker": "\u59d4\u8b72\u306b\u306f lead {lead} \u4ee5\u5916\u306b\u3082\u30461\u3064\u306e\u751f\u6210\u7528\u30b5\u30fc\u30d3\u30b9\u304c\u5fc5\u8981\u3067\u3059\u3002\u8a08\u753b\u306b\u8ffd\u52a0\u3059\u308b\u304b\u3001\u5916\u90e8 worker \u306e --worker-url \u3092\u6e21\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "err.orchestrate_nongenerative": "\u9078\u629e\u3055\u308c\u305f {role} \u30b5\u30fc\u30d3\u30b9 {service} \u306f\u751f\u6210\u7528\u3067\u306f\u3042\u308a\u307e\u305b\u3093\u3002chat\u3001code\u3001worker \u306e\u30b5\u30fc\u30d3\u30b9\u3092\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "err.orchestrate_up": "\u9078\u629e\u3057\u305f\u30b5\u30fc\u30d3\u30b9\u304c\u8d77\u52d5\u3057\u3066\u3044\u307e\u305b\u3093\u3002URL \u3092\u6307\u5b9a\u3059\u308b\u304b\u3001\u5148\u306b nmesh up \u3092\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "err.orchestrate_measure": "\u59d4\u8b72\u6e2c\u5b9a\u306b\u5931\u6557\u3057\u307e\u3057\u305f: {error}",
    "label.orchestrate_summary": "\u59d4\u8b72\u6e2c\u5b9a: n={n}",
    "label.orchestrate_passed": "\u5408\u683c: worker={worker}, lead={lead}, delegated={delegated}, ceiling={ceiling}",
    "label.orchestrate_comparison": "{name} \u3068 lead: \u6539\u5584={gained}, \u60aa\u5316={lost}, p={p:.4f}",
    "label.orchestrate_verifier": "\u691c\u8a3c\u5668: \u6b63\u78ba\u5ea6={accuracy:.4f}, \u53d7\u7406={accepted}, \u8aa4\u53d7\u7406={wrong}, \u6b63\u7b54\u5374\u4e0b={right}, \u672a\u89e3\u6790={unparsed}",
    "label.orchestrate_cost": "lead \u30b3\u30b9\u30c8: \u5358\u4f53={solo}, \u59d4\u8b72={delegated}, \u30aa\u30fc\u30d0\u30fc\u30d8\u30c3\u30c9={overhead:.3f}; \u79d2: \u5358\u4f53={solo_seconds:.2f}, \u59d4\u8b72={delegated_seconds:.2f}",
    "label.orchestrate_gate": "\u30b2\u30fc\u30c8: {decision} ({reason})",
    "label.orchestrate_repeats": "\u53cd\u5fa9: {repeats} \u56de\u3001\u4e0d\u5b89\u5b9a\u30bf\u30b9\u30af: {unstable}",
    "label.orchestrate_empty": "\u59d4\u8b72\u6e2c\u5b9a\u306e\u8a18\u9332\u306f\u3042\u308a\u307e\u305b\u3093\u3002",
    "label.orchestrate_title": "nmesh orchestrate show",
    "err.delegate_stream": "nmesh-delegate \u306f\u30b9\u30c8\u30ea\u30fc\u30df\u30f3\u30b0\u3092\u30b5\u30dd\u30fc\u30c8\u3057\u307e\u305b\u3093\u3002",
    "err.delegate_worker": "nmesh-delegate \u306b\u306f\u7570\u306a\u308b\u8a08\u753b\u6e08\u307f\u30b5\u30fc\u30d3\u30b9\u304c2\u3064\u5fc5\u8981\u3067\u3059\u3002",
    "err.delegate_not_coresident": "nmesh-delegate \u306f\u5229\u7528\u3067\u304d\u307e\u305b\u3093: lead {lead} \u3068 worker {worker} \u306f\u30e1\u30e2\u30ea\u4e0a\u306e\u76f8\u4e92\u6392\u4ed6\u3067\u3059\u3002",
    "err.delegate_gate": "nmesh-delegate \u306f\u8a3c\u62e0\u30b2\u30fc\u30c8\u3067\u7121\u52b9\u3067\u3059: {reason}\u3002",
    "err.delegate_gate_stats": "nmesh-delegate \u306f\u8a3c\u62e0\u30b2\u30fc\u30c8\u3067\u7121\u52b9\u3067\u3059: {reason}\uff08delegated={delegated}, lead={lead}, p={p:.4f}\uff09\u3002",
})

MESSAGES["ja"].update({
    "note.eval_underpowered": (
        "{tasks}\u554f\u306e\u30b9\u30a4\u30fc\u30c8\u3067\u306f\u3001\u3053\u306e\u89b3\u6e2c\u30e9\u30f3\u30ad\u30f3\u30b0\u3092"
        "\u89e3\u6c7a\u3067\u304d\u307e\u305b\u3093: {other_rate:.1%} \u5bfe {selected_rate:.1%}"
        "\uff08\u6b63\u78ba\u306ap\u5024={p_value:.4f}\uff09\u3002\u03b1=0.05\u3067\u306e\u6700\u5c0f"
        "\u89e3\u6c7a\u53ef\u80fd\u5dee\u306f {minimum:.1%}\u3067\u3042\u308a\u3001\u3053\u306e\u30b5\u30a4\u30ba\u3067"
        "\u30e9\u30f3\u30ad\u30f3\u30b0\u306f\u78ba\u8a8d\u3082\u53cd\u8a3c\u3082\u3067\u304d\u307e\u305b\u3093\u3002"
        "{upgrade_tasks}\u554f\u306e\u62e1\u5f35\u30b9\u30a4\u30fc\u30c8\u306a\u3089{upgrade_minimum:.1%}\u307e\u3067"
        "\u89e3\u6c7a\u3067\u304d\u307e\u3059\u3002"
    ),
    "note.eval_underpowered_full": (
        "{tasks}\u554f\u306e\u30b9\u30a4\u30fc\u30c8\u3067\u306f\u3001\u3053\u306e\u89b3\u6e2c\u30e9\u30f3\u30ad\u30f3\u30b0\u3092"
        "\u89e3\u6c7a\u3067\u304d\u307e\u305b\u3093: {other_rate:.1%} \u5bfe {selected_rate:.1%}"
        "\uff08\u6b63\u78ba\u306ap\u5024={p_value:.4f}\uff09\u3002\u03b1=0.05\u3067\u306e\u6700\u5c0f"
        "\u89e3\u6c7a\u53ef\u80fd\u5dee\u306f {minimum:.1%}\u3067\u3042\u308a\u3001\u3053\u306e\u30b5\u30a4\u30ba\u3067"
        "\u30e9\u30f3\u30ad\u30f3\u30b0\u306f\u78ba\u8a8d\u3082\u53cd\u8a3c\u3082\u3067\u304d\u307e\u305b\u3093\u3002"
    ),
    "note.eval_suite_upgrade": (
        "{tasks}\u554f\u306e\u62e1\u5f35\u30b9\u30a4\u30fc\u30c8\u3067\u306f\u3001\u03b1=0.05\u3067\u6700\u5c0f"
        "\u89e3\u6c7a\u53ef\u80fd\u5dee\u304c{minimum:.1%}\u307e\u3067\u5c0f\u3055\u304f\u306a\u308a\u307e\u3059\u3002"
        "`nmesh eval --suite extended` \u3092\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
    ),
})

_PRIMARY_SUBTAG = re.compile(r"^[A-Za-z]+")


def _primary(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = _PRIMARY_SUBTAG.match(value.strip())
    return match.group(0).lower() if match else None


def lang() -> str:
    """Resolve a supported language without changing process locale state."""
    values: list[object] = [
        os.environ.get("NMESH_LANG"),
        os.environ.get("LC_ALL"),
        os.environ.get("LC_MESSAGES"),
        os.environ.get("LANG"),
    ]
    try:
        values.append(locale.getlocale()[0])
    except (AttributeError, ValueError):
        values.append(None)
    for value in values:
        primary = _primary(value)
        if primary:
            return primary if primary in SUPPORTED else "en"
    return "en"


def t(key: str, lang_code: str = "en", **params: object) -> str:
    """Translate and format a message; missing translations never raise."""
    language = lang_code if lang_code in MESSAGES else "en"
    template = MESSAGES[language].get(key)
    if template is None:
        return key
    try:
        return template.format(**params)
    except (KeyError, IndexError, ValueError):
        return template
