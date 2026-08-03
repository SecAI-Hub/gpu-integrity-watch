# gpu-integrity-watch

`gpu-integrity-watch` monitors model files, deterministic sentinel behavior, GPU ECC state, driver fingerprints, and device-node allowlists for SecAI_OS.

## Security model

- Missing baselines, skipped probes, and probe errors produce an `unknown` verdict, so unavailable evidence cannot become a healthy claim. `unknown` can alert and report an incident but never runs quarantine or shutdown actions.
- Model scans reject symlinks, non-regular files, oversized inputs, traversal, scan errors, and unexpected model files.
- Inference responses, subprocess output, configuration, and baselines are bounded. Probe commands have deadlines; configured action commands require an explicit opt-in and a canonical absolute executable and never use a shell.
- Remote service URLs require HTTPS; plaintext HTTP is accepted only for loopback endpoints.
- Quarantine refuses unsafe directories, symlinks, nested destinations, and overwrites, and syncs source/destination directories.
- The daemon reads owner-only credentials of at least 32 characters from files, authenticates and rate-limits every `/v1/*` route, serializes probe/reload/baseline mutations, and requires a strict, bounded, hash-chained audit log. An audit failure makes `/health` unhealthy and blocks sensitive operations.
- Warning, critical, and unknown results can be delivered to `ai-incident-recorder` through the authenticated integration, with duplicate-report cooldown.

## Development

```sh
go test -race ./...
go vet ./...
go run golang.org/x/vuln/cmd/govulncheck@v1.3.0 ./...
```

Capture a baseline only on a known-good, measured host:

```sh
go run . baseline -profile profiles/default-profile.yaml -out baseline.yaml
go run . check -profile profiles/default-profile.yaml
```

Never automatically enroll a baseline after an alert. Review the model, driver, device inventory, and release provenance first.

## Daemon deployment

Provide owner-only credential files through `SERVICE_TOKEN_PATH` and, when configured, `integrations.incident_recorder_token_path`. Mount the model directory read-only, the enrolled baseline read-only during ordinary operation, only the required GPU device nodes, and durable audit storage. Baseline capture is a privileged maintenance operation: the HTTP endpoint is disabled unless `GPU_WATCH_ALLOW_BASELINE_CAPTURE=true` is set for a controlled maintenance window. Remove the setting and restore the baseline mount to read-only immediately afterward. Quarantine and inference shutdown are separately disabled unless `GPU_WATCH_ALLOW_QUARANTINE=true` and `GPU_WATCH_ALLOW_SHUTDOWN=true`, respectively; enable either only after staging the configured cooldown and recovery procedure.

Use a read-only root filesystem, drop all capabilities, set `no-new-privileges`, and apply PID/CPU/memory limits. Bind to loopback or a private service network. The unauthenticated `/health` endpoint exposes only readiness and returns `503` if audit durability is lost; `/v1/check`, `/v1/status`, `/v1/history`, `/v1/attest-state`, `/v1/baseline`, `/v1/reload`, and `/v1/metrics` require a bearer token. Rotate the 64 MiB audit log only while the service is stopped, preserving and externally anchoring the prior chain head.

Quarantine preserves nested paths with no-clobber hard links, so its target must be on the same filesystem as the model directory. Treat any partial-containment result as an operational emergency.

See [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for the detailed findings, validation record, and residual deployment risks.
