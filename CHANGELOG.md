# 変更履歴

## 未リリース

- `nmesh --version` が bench harness と深度プローブ規則の識別子を出力するようになりました。`bench-v2` への更新（デコード速度が n/(n-1) 倍に膨らんでいた件の修正）は別のブランチで入り、その時点で `bench-v1` の記録は比較に使わなくなります。
- 深度プローブの採点規則変更により既存の深度証拠は無効です。`nmesh eval --depth` の再実行が必要です。
- GPU 実機・vLLM・MLX は未検証です。
- CI は未稼働です（ワークフロー未設置）。
- Add pooled retrieval evidence and opt-in gateway auto-chunking.
- `nmesh bench --retrieval` records single-vector retrieval usability; the planner warns rather than clamps when recall degrades and recommends chunking long inputs.
- Retrieval evidence now confirms whether chunking at the measured usable length restores rank-1 retrieval before recommending it.
- `nmesh bench` now rejects embedding-only services and reports benchmark HTTP failures without a traceback.
- The gateway now rejects proven silently truncated single-input embeddings and marks saturated multi-input requests as unverified.
- Embedding-only plans no longer claim decode throughput or use decode speed for planner gating and ranking; `nmesh plan` shows `—`.
- 旧 benchmark harness の保存済み計測は planner の証拠に使わず、`nmesh bench` による再測定を必要とするようにしました。
- `nmesh evidence` で保存済み証拠の使用可否、理由コード、再測定コマンドを棚卸しできるようにしました。
- 埋め込みサービスの実測入力上限とエンコード速度を保存し、計画の文脈を実際の served cap に合わせて短縮するようにしました。Ollama 0.33.2 の bge-m3 がこのホストで `num_ctx` に関係なく 2048 トークンで静かに切り詰めること、llama.cpp が計画した窓全体を提供したことを記録します。
