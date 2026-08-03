# Security and Production-Readiness Audit

Audit date: 2026-08-02

## Scope

The audit covered profile/baseline handling, tensor/sentinel/drift/ECC probes, scoring, containment actions, daemon authentication and HTTP behavior, recorder integration, filesystem/subprocess/network boundaries, audit persistence, container packaging, dependencies, tests, and GitHub automation. The review was source-based and exercised in Linux containers on a macOS development host; real Fedora GPU hardware was not available.

## Remediated findings

- **Critical — missing or failed evidence could classify the GPU as healthy or trigger unsafe containment.** Empty result sets, skipped probes, malformed/unsupported ECC fields, missing trusted tooling, and probe errors now produce `unknown` with a distinct CLI failure code. Unknown evidence can alert but never executes quarantine or shutdown. Those actions have separate explicit opt-ins and durable cooldown/idempotency guards.
- **Critical — action commands used a shell.** Command actions are disabled by default, require a maintenance opt-in and canonical absolute executable, run directly without a shell, and have strict time/output bounds.
- **Critical — quarantine could follow unsafe paths, overwrite evidence, or omit nested models.** Source/quarantine directories and entries are checked, symlinks/non-regular files are rejected, destinations are no-clobber hard links, nested model paths are preserved, directory changes are synced, and partial containment is reported as failure.
- **High — baseline/model hashing was incomplete and unbounded.** Recursive scans fail on symlinks, non-regular entries, unreadable paths, excessive files, oversized files, invalid patterns, hash races, missing files, extra files, and mismatches.
- **High — GPU hardware state was not included.** Optional fail-closed driver fingerprint and exact device-node allowlist probes were added; baseline capture enrolls their evidence when enabled.
- **High — ECC parsing and `nvidia-smi` trust were ambiguous.** ECC CSV is parsed as exactly three fields with bounded unsigned counters, every GPU is accounted for, and the worst state wins. `N/A`, unsupported modes, malformed rows, overflow, and untrusted tooling become `unknown`. A configured tool must be an absolute, canonical, non-symlink regular executable with trusted ownership and permissions.
- **High — HTTP probes/actions/integrations could hang, leak resources, or redirect credentials.** Requests use bounded clients and response reads, same-origin redirect limits, strict response parsing, status checks, and TLS for non-loopback destinations.
- **High — daemon routes and baseline enrollment were insufficiently separated.** An owner-only bearer token is required for every `/v1/*` route. Remote baseline capture additionally requires `GPU_WATCH_ALLOW_BASELINE_CAPTURE=true` during an explicit maintenance window.
- **High — malformed reloads could terminate the service and valid reloads left stale scoring.** Reload now parses without fatal exit, rejects bind changes that require restart, atomically updates probes/baseline/actions/integration credentials, updates scoring/history limits, and resets the background interval.
- **Feature integration.** `/v1/attest-state` exposes authenticated GPU trust state, and warning/critical/unknown results can be sent as bounded authenticated events to `ai-incident-recorder` with duplicate suppression.
- **High — audit loss did not fail closed.** Audit records are now strict, bounded SHA-256 chains verified in full at startup. Writes and syncs are checked, a failure poisons service health and blocks protected endpoints/actions, and a 64 MiB cap forces controlled external rotation.
- **Supply chain / operations.** The image uses digest-pinned Go 1.26.5 and Alpine 3.23 bases, a stripped static binary, non-root UID, health check, narrow build context, and durable state volume. The runtime copies CA roots from the immutable builder stage instead of installing unpinned packages. CI actions are SHA-pinned and enforce module verification, formatting, bounded race tests, vet, `govulncheck`, current-tree and full-history secret scanning, container builds, CodeQL, and Dependabot updates. Tagged releases produce Linux binaries, a CycloneDX SBOM, Sigstore-signed checksums, and GitHub provenance attestations. A read-only release gate requires a v-prefixed strict SemVer annotated tag that resolves exactly to the workflow commit and is contained in `origin/main`; write and OIDC permissions remain confined to the publishing job.

## Residual risks and deployment requirements

- These are software probes, not GPU firmware/VBIOS attestation or TPM/TEE-backed measured boot. A driver module is hashed only when `module_path` is configured, and firmware/microcode provenance remains outside scope.
- Baselines are owner-controlled YAML, not signed transparency-log entries. Capture them only on a measured known-good host, record release provenance separately, and mount them read-only during normal operation.
- Sentinel similarity is intentionally simple and model outputs can be nondeterministic. Tune deterministic reference cases and thresholds on the deployed model, and never let sentinel checks replace tensor/driver/device measurements.
- Quarantine uses no-clobber hard links and therefore requires source and quarantine to be on the same filesystem. It may partially succeed under concurrent change or storage failure; failed containment must page an operator and the inference service should also have an independent shutdown control.
- Local JSONL audit entries are hash-chained but not externally anchored. Stop the service for controlled rotation, preserve the prior chain and head, enable the authenticated incident-recorder integration, and export evidence to append-only remote storage.
- Authentication is a single bearer token without RBAC or mTLS. Rotate credentials, bind to loopback/private networks, and use a hardened authenticated proxy when crossing hosts.
- Real NVIDIA/AMD/Intel devices, ECC output variants, Fedora/SELinux permissions, container device namespaces, inference APIs, and destructive-action fault injection require hardware-in-the-loop staging before production enablement.

## Verification performed

- `go test -race -count=1 ./...`
- `go vet ./...`
- `govulncheck ./...` with `golang.org/x/vuln/cmd/govulncheck@v1.3.0` — no reachable vulnerabilities
- Gitleaks 8.30.1 current tree and Git history — no leaks
- Trivy 0.70.0 filesystem, Dockerfile, and image scan at HIGH/CRITICAL — no findings (database dated 2026-08-02)
- Digest-pinned Docker image build and inspection — non-root UID `65534:65534`
- Actionlint on all workflows — no findings
- Release dependency/permission graph checks, strict SemVer positive/negative fixtures, and synthetic annotated/lightweight/off-main Git ancestry cases — passed
