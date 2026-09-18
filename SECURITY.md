# Security Policy

## Reporting a vulnerability

Please open a GitHub issue marked **security** or contact the maintainer directly
via the repository. Do not post exploit details publicly before a fix lands.

## Scope

nmesh runs local inference processes and a localhost-bound gateway. Relevant
concerns:

- command injection through model IDs, paths, or policy flags
- the gateway binding beyond 127.0.0.1 unintentionally
- secrets or local paths leaking into logs, evidence records, or telemetry
- downloaded artifacts (engines, GGUF weights) that fail integrity checks

## Principles

- The gateway binds `127.0.0.1` by default; exposing it on a LAN requires an
  explicit flag.
- Downloads verify sizes and record provenance; a mismatched artifact is
  refused, not run.
- nmesh never sends prompts, file contents, or telemetry off the machine.
