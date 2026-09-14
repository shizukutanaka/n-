# 変更履歴

## 未リリース

- `nmesh evidence` の棚卸しに投機実行（spec）の測定記録を加え、テキスト表では検索（retrieval）の記録も表示するようにしました。どちらもこれまで保存されながら一覧に出ていませんでした。
- 計画したサービス全てに自身の構成の実測合格率がある場合は、「順位付けは未検証のカタログ品質主張を使用します」ではなく測定値そのものを報告するようにしました。
- 埋め込みベンチの上限プローブでバックエンドが明示的に拒否した入力を静かな切り詰めと区別して記録し、測定済みで上限が無い場合は「未検証」ではなく「切り詰めなし」として報告するようにしました。
- Linux の llama.cpp 配布物に含まれる相対シンボリックリンクと実行ビットを保持して展開し、`nmesh up --dry-run` ではエンジンを自動取得しないようにしました。
- Add character-bound gateway auto-chunking for non-whitespace text.
- 純粋な推定値にもデコード速度の下限を適用し、未確認の実測値だけを再測定待ちとして残すようにしました。
- Add pooled retrieval evidence and opt-in gateway auto-chunking.
- `nmesh bench --retrieval` records single-vector retrieval usability; the planner warns rather than clamps when recall degrades and recommends chunking long inputs.
- Retrieval evidence now confirms whether chunking at the measured usable length restores rank-1 retrieval before recommending it.
- `nmesh bench` now rejects embedding-only services and reports benchmark HTTP failures without a traceback.
- The gateway now rejects proven silently truncated single-input embeddings and marks saturated multi-input requests as unverified.
- Embedding-only plans no longer claim decode throughput or use decode speed for planner gating and ranking; `nmesh plan` shows `—`.
- 旧 benchmark harness の保存済み計測は planner の証拠に使わず、`nmesh bench` による再測定を必要とするようにしました。
- `nmesh evidence` で保存済み証拠の使用可否、理由コード、再測定コマンドを棚卸しできるようにしました。
- 埋め込みサービスの実測入力上限とエンコード速度を保存し、計画の文脈を実際の served cap に合わせて短縮するようにしました。Ollama 0.33.2 の bge-m3 がこのホストで `num_ctx` に関係なく 2048 トークンで静かに切り詰めること、llama.cpp が計画した窓全体を提供したことを記録します。