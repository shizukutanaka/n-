# 変更履歴

## 未リリース

- `nmesh bench` now rejects embedding-only services and reports benchmark HTTP failures without a traceback.
- Embedding-only plans no longer claim decode throughput or use decode speed for planner gating and ranking; `nmesh plan` shows `—`.
- 旧 benchmark harness の保存済み計測は planner の証拠に使わず、`nmesh bench` による再測定を必要とするようにしました。
- `nmesh evidence` で保存済み証拠の使用可否、理由コード、再測定コマンドを棚卸しできるようにしました。
