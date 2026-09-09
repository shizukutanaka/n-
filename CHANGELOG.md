# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 未リリース

### 互換性に影響する変更

- `nmesh --version` が bench harness と深度プローブ規則の識別子を出力するようになりました。`bench-v2` への更新（デコード速度が n/(n-1) 倍に膨らんでいた件の修正）は別のブランチで入り、その時点で `bench-v1` の記録は比較に使わなくなります。
- 深度プローブの採点規則変更により既存の深度証拠は無効です。`nmesh eval --depth` の再実行が必要です。

### 既知の制約

- GPU 実機・vLLM・MLX は未検証です。
- CI は未稼働です（ワークフロー未設置）。
