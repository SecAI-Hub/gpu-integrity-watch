# Security Policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Email **security@secai-hub.dev** with reproduction steps, impact, and any proposed remediation. Do not include customer data or active credentials.

## Design guarantees

- All `/v1/*` endpoints require a bearer token loaded from an owner-only file; `/health` is liveness-only.
- Missing, skipped, and errored probe evidence produces `unknown`; it can alert but never triggers quarantine or shutdown.
- Profiles and baselines are strict, bounded YAML documents. Model and driver reads reject symlinks and unsafe files.
- External HTTP calls have deadlines, bounded responses, and same-origin redirect rules.
- Action commands are disabled by default, use no shell, require a canonical absolute executable, and have bounded output and execution time.
- Baselines are atomically written and must be enrolled only on a known-good host.
- Quarantine and shutdown require separate environment opt-ins and destructive-action cooldowns; audit failure blocks future protected operations.
- Audit records are strict, bounded, hash chained, synced, and verified in full before startup.

## Residual trust assumptions

The service cannot establish GPU integrity if the kernel, driver, device namespace, baseline, or service process is already controlled by an attacker. Hardware-backed boot/runtime attestation and protected baseline distribution belong in the surrounding SecAI_OS deployment. ECC support and device/driver evidence vary by vendor and require qualification on each supported GPU family.
