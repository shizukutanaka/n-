---
name: testing-nmesh-cli
description: How to set up and E2E-test the nmesh local-LLM orchestrator CLI on this machine — NMESH_HOME homes, ports, saved-plan gotchas, gateway endpoints, auth, and recording guidance.
---

# Testing nmesh CLI

nmesh is a user-facing CLI — the terminal IS its UI. Run commands in a visible maximized konsole and record it (record shell-only runs too here; this is the documented exception). Use `/home/ubuntu/repos/n-/.venv/bin/nmesh` (or `.venv/bin/python -m nmesh.cli`).

## Environment
- Repo: `/home/ubuntu/repos/n-` (shared worktree with the lead — ALWAYS check `git branch --show-current` + `git status` before testing; the tree may be mid-rebase from the lead's work).
- Two NMESH_HOMEs exist; the task names which to use:
  - `/home/ubuntu/nmesh-home` — llama.cpp b10970, models qwen2.5-1.5b + bge-m3
  - `/home/ubuntu/.nmesh` — llama.cpp b11037, models qwen3-1.7b (as `Qwen_Qwen3-1.7B-Q4_K_M.gguf`), bge-m3-FP16, qwen2.5-1.5b, Qwen3-Reranker-0.6B.f16
- Always `export NMESH_HOME=...` per task; never assume.
- Ports: gateway 127.0.0.1:18000; services 18010+ in plan order (chat 18010, embed 18011, rerank 18012 when all three exist — with 2 roles the second service takes 18011).
- Check listeners: `ss -tln | grep -E '18000|18010|18011|18012'`; check procs: `pgrep -af llama-server`.

## Saved-plan gotcha (biggest trap)
`nmesh up`/`run` reuse a saved runnable `plan.json` VERBATIM — plan-shaping flags on `up` (`--roles`, `--model`, `--context-shift`, etc.) are silently ignored when a runnable plan exists. `nmesh plan` always re-plans and rewrites plan.json. `up --dry-run` prints the resolved plan+argv without launching — use it to detect shadowing. To force a re-plan through `up`, move plan.json aside or use `--model`/`--lang`/`--ignore-eval-evidence` (which force replan). At end of testing, restore a sane `plan.json` (e.g. `nmesh plan --roles ...` canonical) and `nmesh down`.

## Model/planner facts
- Catalog: `nmesh/catalog/models.yaml` (id, family, params, roles, quality, sources.hf_gguf). Downloads land in `$NMESH_HOME/models/` under planned names; `artifacts.json` caches sizes — acquisition may hardlink a differently-named on-disk file instead of re-downloading.
- Planner ranks candidates by a `score` built from quality etc.; for role=rerank both `roles=[rerank]` (dedicated) and `roles=[embed]` (dual-use) models are candidates — quality decides (bge-m3 q90 > qwen3-reranker-0.6b q85 > ... ; reranker-8b q91). To force the dedicated reranker: `nmesh plan --roles chat,rerank --model <chat>,<reranker>` so it's the only rerank-capable candidate.
- `--model` is comma-separated catalog ids restricting the candidate set.
- Non-generative roles (embed/rerank) get `kv_bytes_per_tok: 0.0` in plan memory and no decode bench (`_is_non_generative`). llama.cpp rerank service argv includes `--reranking`; embed gets `--embeddings --pooling cls|last`.
- `up` prints an HF warning for unauthenticated downloads; big f16 models (8b) are ~16GB — check `total_download_bytes` in `plan --json` before committing to `up`; fall back to on-disk models via `--model`.
- llama.cpp `--reranking` on a dedicated qwen3-reranker returns near-zero undifferentiated logits (no instruction template applied) — scores work but ranking may look wrong; bge-m3 as reranker ranks correctly.

## Runtime verification tricks
- Verify real service argv: `tr '\0' ' ' </proc/<pid>/cmdline` (find pids via `pgrep -f llama-server`).
- Exact prompt token counts: llama-server `/tokenize` on the service port (NOT the gateway).
- Gateway endpoints: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/messages` + `/v1/messages/count_tokens` (Anthropic — llama.cpp b11037 serves natively), `/v1/rerank`, `/v1/jobs` + `/v1/jobs/{id}` (jobs carry `progress:{decoded,remaining}` from llama.cpp `/slots`; CLI `nmesh jobs` shows "running N/M"). `/v1/jobs` needs an in-flight request — background a `curl` with large `max_tokens`.
- `--reranking` and `--embeddings` are mutually exclusive pooling modes — one llama-server instance per mode.

## Auth / CORS
- Gateway reads `NMESH_API_KEY` from its process env only. `nmesh up --detach` spawns the gateway inheriting parent env → `NMESH_API_KEY=x nmesh up --detach` enables auth. `gateway.env` (under NMESH_HOME) is only used by installed autostart service units, not ad-hoc `up`.
- CLI calls (`run`, `jobs`, `status`) send `Authorization: Bearer $NMESH_API_KEY` when the env is set in the CLI's own env.
- CORS middleware: OPTIONS on /v1/* WITH `Origin` header → 204 + reflected allow-origin/methods/headers/max-age 600, evaluated BEFORE auth (204 even without key). OPTIONS without Origin → falls through to routes (405). Real requests get `access-control-allow-origin` only when Origin header sent.
- Error surfacing: `run` prints upstream `error.message` via `err.gateway_http` (e.g. "gateway request failed (HTTP 500): <llama-server msg>", "(HTTP 401): Invalid or missing API key") — distinct from a connect failure.

## i18n / misc
- `NMESH_LANG=ja` or `LANG=ja_JP.UTF-8` switches messages; this box has NO CJK fonts → terminal shows boxes; capture `> file` and check UTF-8 content with python to prove real Japanese.
- `nmesh bench` without services exercises the `err.bench_up` i18n path.
- Keep artifacts in /tmp/nmesh-test/; screenshots auto-save under /home/ubuntu/screenshots/; recordings /home/ubuntu/screencasts/.

## macOS session notes (Apple Silicon VM, repo `/Users/devin/repos/n-`)
- Use `.venv/bin/nmesh` (py3.12); system `python3` is 3.9.6 — too old. `.venv310` is a 3.10 check env.
- Smoke home: `NMESH_HOME=/Users/devin/repos/n-/.nmesh-smoke` (llama.cpp b11056 + qwen3-0.6b Q8_0/bf16 already installed; NOT git-tracked).
- No `/proc`, no `ss`. Listeners: `lsof -nP -iTCP:18010 -sTCP:LISTEN`. Proc argv: `ps -p <pid> -o command=` or `ps aux | grep llama-server`.
- `psutil.net_connections()` raises `AccessDenied` when unprivileged — nmesh falls back to `lsof`; the product path works either way.
- **Reasoning models think by default**: qwen3-0.6b (the default chat on this profile) spends the whole output budget on `<think>` — `nmesh run` looks stuck ~50s; use `/no_think` in the prompt (Qwen template magic word) or `chat_template_kwargs: {"enable_thinking": false}` in API calls for fast checks. `nmesh eval` needs `--reasoning-allowance 1024` or all 16 tasks score "unscorable" (finish_reason=length, empty content) — by design those runs are excluded from planner evidence (`planner_eval_records` drops records with unscorable/transport_errors/depth>0).
- `nmesh autostart --install` writes `~/Library/LaunchAgents/com.nmesh.gateway.plist` + `$NMESH_HOME/nmesh-gateway-launcher.sh`/`gateway.env` (0600); it does NOT `launchctl load` — remove the plist after testing.
- Decode ~5-7 tok/s for 0.6B is this shared VM's limit (CPU-bound even with `-ngl 14` Metal offload), not a product bug.
