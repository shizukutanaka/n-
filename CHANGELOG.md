# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 未リリース

### 互換性に影響する変更

- bench harness を `bench-v2` に更新しました。デコード速度が n/(n-1) 倍に膨らんでいたため、`bench-v1` の記録は比較に使いません。再測定が必要です。
- 深度プローブの採点規則変更により既存の深度証拠は無効です。`nmesh eval --depth` の再実行が必要です。

### 既知の制約

- GPU 実機・vLLM・MLX は未検証です。
- CI は未稼働です（ワークフロー未設置）。
