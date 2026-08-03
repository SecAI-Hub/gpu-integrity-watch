package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
)

// ---------- profile / config types ----------

// IntegrityProfile is the top-level configuration loaded from YAML.
type IntegrityProfile struct {
	Version      int               `yaml:"version"`
	ModelDir     string            `yaml:"model_dir"`
	InferenceURL string            `yaml:"inference_url"`
	Probes       []ProbeConfig     `yaml:"probes"`
	Scoring      ScoringConfig     `yaml:"scoring"`
	Actions      []ActionConfig    `yaml:"actions"`
	Daemon       DaemonConfig      `yaml:"daemon"`
	BaselineFile string            `yaml:"baseline_file"`
	Integrations IntegrationConfig `yaml:"integrations"`
}

const (
	maxConfigSize = 1 << 20
	// #nosec G101 -- these are filesystem locations for runtime-mounted tokens, not credential material.
	defaultTokenPath = "/run/secure-ai/service-token"
	// #nosec G101 -- this is likewise only a credential-file path.
	defaultIncidentTokenPath = "/run/secure-ai/incident-recorder-token"
	defaultAuditLogPath      = "/var/lib/secure-ai/logs/gpu-integrity-watch-audit.jsonl"
	maxAuditSize             = 64 << 20
	maxAuditLine             = 64 << 10
)

// ProbeConfig defines a single probe's configuration.
type ProbeConfig struct {
	Name     string            `yaml:"name" json:"name"`
	Type     ProbeType         `yaml:"type" json:"type"`
	Enabled  bool              `yaml:"enabled" json:"enabled"`
	Settings map[string]string `yaml:"settings,omitempty" json:"settings,omitempty"`
}

// ScoringConfig defines scoring engine parameters.
type ScoringConfig struct {
	Weights    map[string]float64 `yaml:"weights"`
	MaxHistory int                `yaml:"max_history"`
}

// DaemonConfig defines daemon mode settings.
type DaemonConfig struct {
	BindAddr string `yaml:"bind_addr"`
	Interval string `yaml:"interval"` // e.g. "5m"
}

// ---------- audit log ----------

var (
	auditMu       sync.Mutex
	auditFile     *os.File
	auditPath     string
	auditLastHash string
	auditRequired atomic.Bool
	auditHealthy  atomic.Bool
)

type GPUAuditEntry struct {
	Timestamp string  `json:"timestamp"`
	Event     string  `json:"event"`
	Verdict   Verdict `json:"verdict,omitempty"`
	Score     float64 `json:"score,omitempty"`
	Source    string  `json:"source,omitempty"`
	Files     int     `json:"files,omitempty"`
	Actions   int     `json:"actions,omitempty"`
	PrevHash  string  `json:"prev_hash,omitempty"`
	Hash      string  `json:"hash"`
}

func initAudit(path string) error {
	auditRequired.Store(true)
	auditHealthy.Store(false)
	auditLastHash = ""
	if path == "" {
		return fmt.Errorf("audit log path is required")
	}
	auditPath = path
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return fmt.Errorf("audit log path must be canonical and absolute")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create audit directory: %w", err)
	}
	dirInfo, err := os.Lstat(filepath.Dir(path))
	if err != nil || !dirInfo.IsDir() || dirInfo.Mode()&os.ModeSymlink != 0 ||
		dirInfo.Mode().Perm()&0o022 != 0 || !trustedFileOwner(dirInfo) {
		return fmt.Errorf("audit directory is unsafe")
	}
	if err := verifyGPUAudit(path); err != nil {
		return err
	}
	f, err := openSafeAppend(path)
	if err != nil {
		return fmt.Errorf("open audit log: %w", err)
	}
	auditFile = f
	auditHealthy.Store(true)
	return nil
}

func verifyGPUAudit(path string) error {
	// #nosec G304 -- initAudit requires a canonical absolute path and safe parent; identity, owner, mode, and bounds are rechecked here.
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read audit log: %w", err)
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !opened.Mode().IsRegular() || opened.Size() > maxAuditSize ||
		opened.Mode().Perm()&0o077 != 0 || !trustedFileOwner(opened) {
		return fmt.Errorf("audit log is unsafe, oversized, or has weak permissions")
	}
	pathInfo, err := os.Lstat(path)
	if err != nil || pathInfo.Mode()&os.ModeSymlink != 0 || !os.SameFile(opened, pathInfo) {
		return fmt.Errorf("audit path is unsafe or changed while opening")
	}
	expectedPrev := ""
	scanner := bufio.NewScanner(io.LimitReader(f, maxAuditSize+1))
	scanner.Buffer(make([]byte, 4096), maxAuditLine)
	line := 0
	for scanner.Scan() {
		line++
		decoder := json.NewDecoder(bytes.NewReader(scanner.Bytes()))
		decoder.DisallowUnknownFields()
		var entry GPUAuditEntry
		if err := decoder.Decode(&entry); err != nil {
			return fmt.Errorf("decode audit line %d: %w", line, err)
		}
		var extra any
		if err := decoder.Decode(&extra); err != io.EOF {
			return fmt.Errorf("decode audit line %d: trailing JSON data", line)
		}
		if entry.Event == "" || entry.PrevHash != expectedPrev || entry.Hash != computeGPUAuditHash(entry) {
			return fmt.Errorf("audit integrity failure at line %d", line)
		}
		if _, err := time.Parse(time.RFC3339Nano, entry.Timestamp); err != nil {
			return fmt.Errorf("invalid audit timestamp at line %d", line)
		}
		expectedPrev = entry.Hash
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scan audit log: %w", err)
	}
	after, err := f.Stat()
	if err != nil || !os.SameFile(opened, after) || after.Size() != opened.Size() {
		return fmt.Errorf("audit log changed while reading")
	}
	auditLastHash = expectedPrev
	return nil
}

func computeGPUAuditHash(entry GPUAuditEntry) string {
	entry.Hash = ""
	data, _ := json.Marshal(entry)
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func auditAvailable() bool {
	return !auditRequired.Load() || auditHealthy.Load()
}

func auditLog(entry GPUAuditEntry) error {
	if auditFile == nil {
		if auditRequired.Load() {
			return fmt.Errorf("audit log is unavailable")
		}
		return nil
	}
	auditMu.Lock()
	defer auditMu.Unlock()
	if !auditHealthy.Load() {
		return fmt.Errorf("audit log is unhealthy")
	}
	entry.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	entry.PrevHash = auditLastHash
	entry.Hash = computeGPUAuditHash(entry)
	data, err := json.Marshal(entry)
	if err != nil {
		auditHealthy.Store(false)
		return fmt.Errorf("audit marshal: %w", err)
	}
	line := append(data, '\n')
	info, err := auditFile.Stat()
	pathInfo, pathErr := os.Lstat(auditPath)
	if err != nil || pathErr != nil || !os.SameFile(info, pathInfo) || pathInfo.Mode()&os.ModeSymlink != 0 ||
		info.Mode().Perm()&0o077 != 0 || !trustedFileOwner(info) {
		auditHealthy.Store(false)
		return fmt.Errorf("audit path changed or became unsafe")
	}
	if info.Size() > maxAuditSize-int64(len(line)) {
		auditHealthy.Store(false)
		return fmt.Errorf("audit log reached its %d-byte rotation limit", maxAuditSize)
	}
	written, err := auditFile.Write(line)
	if err != nil || written != len(line) {
		if err == nil {
			err = io.ErrShortWrite
		}
		auditHealthy.Store(false)
		return fmt.Errorf("audit write: %w", err)
	}
	if err := auditFile.Sync(); err != nil {
		auditHealthy.Store(false)
		return fmt.Errorf("audit sync: %w", err)
	}
	auditLastHash = entry.Hash
	return nil
}

// ---------- metrics ----------

var (
	metricChecks   atomic.Int64
	metricPass     atomic.Int64
	metricDrift    atomic.Int64
	metricFail     atomic.Int64
	metricActions  atomic.Int64
	metricHTTPReqs atomic.Int64
)

// ---------- main ----------

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "check":
		cmdCheck()
	case "watch":
		cmdWatch()
	case "daemon":
		cmdDaemon()
	case "baseline":
		cmdBaseline()
	case "status":
		cmdStatus()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `gpu-integrity-watch — GPU model integrity monitoring

Usage:
  gpu-integrity-watch check     [-profile FILE]              Run probes once
  gpu-integrity-watch watch     [-profile FILE]              Watch continuously (foreground)
  gpu-integrity-watch daemon    [-profile FILE]              Run as HTTP daemon
  gpu-integrity-watch baseline  [-profile FILE] [-out FILE]  Capture baseline hashes
  gpu-integrity-watch status    [-addr ADDR]                 Query daemon status

Environment:
  INTEGRITY_PROFILE     Path to profile YAML (default: profiles/default-profile.yaml)
  SERVICE_TOKEN_PATH    Owner-only file containing the daemon bearer token
  AUDIT_LOG             Path to JSONL audit log
  GPU_WATCH_ALLOW_BASELINE_CAPTURE
                        Set to true only during an authorized maintenance window
`)
}

// ---------- profile loading ----------

func profilePath() string {
	path := envOr("INTEGRITY_PROFILE", "profiles/default-profile.yaml")
	args := os.Args[2:]
	for i, arg := range args {
		if arg == "-profile" && i+1 < len(args) {
			path = args[i+1]
		}
	}
	return path
}

func loadProfileFile(path string) (*IntegrityProfile, error) {
	data, err := readTrustedConfigFile(path, maxConfigSize)
	if err != nil {
		return nil, fmt.Errorf("load profile %s: %w", path, err)
	}

	var profile IntegrityProfile
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&profile); err != nil {
		return nil, fmt.Errorf("parse profile: %w", err)
	}
	if err := ensureYAMLEOF(decoder); err != nil {
		return nil, fmt.Errorf("parse profile: %w", err)
	}
	if err := validateProfile(&profile); err != nil {
		return nil, fmt.Errorf("invalid profile: %w", err)
	}
	return &profile, nil
}

func loadProfile() *IntegrityProfile {
	profile, err := loadProfileFile(profilePath())
	if err != nil {
		log.Fatal(err)
	}
	return profile
}

func loadBaselineFile(profile *IntegrityProfile) (*Baseline, error) {
	path := profile.BaselineFile
	if path == "" {
		path = "baseline.yaml"
	}

	data, err := readTrustedConfigFile(path, maxConfigSize)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("load baseline: %w", err)
	}

	var b Baseline
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&b); err != nil {
		return nil, fmt.Errorf("parse baseline: %w", err)
	}
	if err := ensureYAMLEOF(decoder); err != nil {
		return nil, fmt.Errorf("parse baseline: %w", err)
	}
	if err := validateBaseline(&b); err != nil {
		return nil, fmt.Errorf("invalid baseline: %w", err)
	}
	return &b, nil

}

func loadBaseline(profile *IntegrityProfile) *Baseline {
	baseline, err := loadBaselineFile(profile)
	if err != nil {
		log.Fatal(err)
	}
	return baseline
}

func ensureYAMLEOF(decoder *yaml.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple YAML documents are not allowed")
		}
		return err
	}
	return nil
}

func validateProfile(profile *IntegrityProfile) error {
	if profile.Version != 1 {
		return fmt.Errorf("unsupported version %d", profile.Version)
	}
	if len(profile.Probes) == 0 {
		return fmt.Errorf("at least one probe is required")
	}
	seen := make(map[string]bool)
	enabled := 0
	for _, probe := range profile.Probes {
		if probe.Name == "" || seen[probe.Name] {
			return fmt.Errorf("probe names must be non-empty and unique")
		}
		seen[probe.Name] = true
		switch probe.Type {
		case ProbeTensorHash, ProbeSentinelInfer, ProbeReferenceDrift, ProbeECCStatus,
			ProbeDriverFingerprint, ProbeDeviceAllowlist:
		default:
			return fmt.Errorf("unknown probe type %q", probe.Type)
		}
		if probe.Enabled {
			enabled++
		}
		if configured := strings.TrimSpace(probe.Settings["nvidia_smi_path"]); configured != "" {
			if _, err := validateTrustedExecutable(configured); err != nil {
				return fmt.Errorf("probe %q has invalid nvidia_smi_path: %w", probe.Name, err)
			}
		}
	}
	if enabled == 0 {
		return fmt.Errorf("at least one probe must be enabled")
	}
	if profile.Scoring.MaxHistory < 0 || profile.Scoring.MaxHistory > 10000 {
		return fmt.Errorf("max_history is out of range")
	}
	for name, weight := range profile.Scoring.Weights {
		switch ProbeType(name) {
		case ProbeTensorHash, ProbeSentinelInfer, ProbeReferenceDrift, ProbeECCStatus,
			ProbeDriverFingerprint, ProbeDeviceAllowlist:
		default:
			return fmt.Errorf("unknown scoring probe type %q", name)
		}
		if math.IsNaN(weight) || math.IsInf(weight, 0) || weight <= 0 || weight > 100 {
			return fmt.Errorf("invalid scoring weight for %s", name)
		}
	}
	if profile.Daemon.Interval != "" {
		interval, err := time.ParseDuration(profile.Daemon.Interval)
		if err != nil || interval < time.Second {
			return fmt.Errorf("daemon interval must be at least one second")
		}
	}
	for _, rawURL := range []string{profile.InferenceURL, profile.Integrations.IncidentRecorderURL} {
		if rawURL == "" {
			continue
		}
		if _, err := parseServiceURL(rawURL); err != nil {
			return fmt.Errorf("invalid inference URL")
		}
	}
	for _, action := range profile.Actions {
		switch action.Type {
		case ActionReload, ActionQuarantine, ActionAlert, ActionFailClosed:
		default:
			return fmt.Errorf("unknown action type %q", action.Type)
		}
		if action.Trigger != VerdictWarning && action.Trigger != VerdictCritical {
			return fmt.Errorf("action %q has invalid trigger", action.Name)
		}
		if action.Cooldown != "" {
			cooldown, err := time.ParseDuration(action.Cooldown)
			if err != nil || cooldown < time.Minute || cooldown > 24*time.Hour {
				return fmt.Errorf("action %q cooldown must be between one minute and 24 hours", action.Name)
			}
		}
	}
	return nil
}

func readBoundedRegularFile(path string, limit int64) ([]byte, error) {
	before, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || before.Mode()&os.ModeSymlink != 0 || before.Size() > limit {
		return nil, fmt.Errorf("not a bounded regular file")
	}
	// #nosec G304 -- path was Lstat-validated as a bounded regular non-symlink and is identity-checked after open.
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(before, opened) {
		return nil, fmt.Errorf("file changed while opening")
	}
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil || int64(len(data)) > limit {
		return nil, fmt.Errorf("file exceeds size limit or could not be read")
	}
	after, err := f.Stat()
	if err != nil || !os.SameFile(opened, after) || after.Size() != int64(len(data)) {
		return nil, fmt.Errorf("file changed while reading")
	}
	return data, nil
}

func readTrustedConfigFile(path string, limit int64) ([]byte, error) {
	before, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || before.Mode()&os.ModeSymlink != 0 || before.Mode().Perm()&0o022 != 0 || !trustedFileOwner(before) {
		return nil, fmt.Errorf("configuration must be a non-writable regular file owned by root or the service user")
	}
	data, err := readBoundedRegularFile(path, limit)
	if err != nil {
		return nil, err
	}
	after, err := os.Lstat(path)
	if err != nil || !os.SameFile(before, after) || after.Mode().Perm()&0o022 != 0 || !trustedFileOwner(after) {
		return nil, fmt.Errorf("configuration changed while validating trust")
	}
	return data, nil
}

func trustedFileOwner(info os.FileInfo) bool {
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return true
	}
	euid := int64(os.Geteuid())
	return stat.Uid == 0 || (euid >= 0 && int64(stat.Uid) == euid)
}

func readOwnerOnlyFile(path string, limit int64) ([]byte, error) {
	before, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || before.Mode()&os.ModeSymlink != 0 || before.Mode().Perm()&0o077 != 0 || !trustedFileOwner(before) {
		return nil, fmt.Errorf("file must be regular and owner-only")
	}
	data, err := readBoundedRegularFile(path, limit)
	if err != nil {
		return nil, err
	}
	after, err := os.Lstat(path)
	if err != nil || !os.SameFile(before, after) || after.Mode().Perm()&0o077 != 0 || !trustedFileOwner(after) {
		return nil, fmt.Errorf("file changed while validating permissions")
	}
	return data, nil
}

func openSafeAppend(path string) (*os.File, error) {
	// #nosec G304 -- the canonical audit path and parent are validated by initAudit and the opened inode is revalidated below.
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	opened, statErr := f.Stat()
	pathInfo, lstatErr := os.Lstat(path)
	if statErr != nil || lstatErr != nil || !opened.Mode().IsRegular() ||
		pathInfo.Mode()&os.ModeSymlink != 0 || !os.SameFile(opened, pathInfo) ||
		opened.Mode().Perm()&0o077 != 0 || !trustedFileOwner(opened) || opened.Size() > maxAuditSize {
		f.Close()
		return nil, fmt.Errorf("unsafe path")
	}
	return f, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func configuredInterval(profile *IntegrityProfile) time.Duration {
	interval := 5 * time.Minute
	if profile.Daemon.Interval != "" {
		interval, _ = time.ParseDuration(profile.Daemon.Interval)
	}
	return interval
}

// ---------- CLI commands ----------

func cmdCheck() {
	profile := loadProfile()
	baseline := loadBaseline(profile)
	if path := os.Getenv("AUDIT_LOG"); path != "" {
		if err := initAudit(path); err != nil {
			log.Fatalf("audit unavailable: %v", err)
		}
		defer auditFile.Close()
	}

	runner := NewProbeRunner(profile, baseline)
	results := runner.RunAll()

	weights := convertWeights(profile.Scoring.Weights)
	scorer := NewScoringEngine(weights, profile.Scoring.MaxHistory)
	entry := scorer.Score(results)
	if err := auditLog(GPUAuditEntry{Event: "check", Verdict: entry.Verdict, Score: entry.CompositeScore}); err != nil {
		log.Fatalf("audit persistence failed before action evaluation: %v", err)
	}

	executor := NewActionExecutor(profile.Actions, profile.ModelDir, profile.InferenceURL)
	actions := executor.Evaluate(entry)
	if err := auditLog(GPUAuditEntry{Event: "check_actions", Verdict: entry.Verdict, Score: entry.CompositeScore, Actions: len(actions)}); err != nil {
		log.Fatalf("audit persistence failed after action evaluation: %v", err)
	}

	metricChecks.Add(1)
	countStatuses(results)
	countTriggeredActions(actions)

	output := map[string]interface{}{
		"probes":  results,
		"score":   entry,
		"actions": actions,
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(output)

	switch entry.Verdict {
	case VerdictCritical:
		os.Exit(2)
	case VerdictWarning:
		os.Exit(1)
	case VerdictUnknown:
		os.Exit(3)
	}
}

func cmdWatch() {
	profile := loadProfile()
	baseline := loadBaseline(profile)
	if path := os.Getenv("AUDIT_LOG"); path != "" {
		if err := initAudit(path); err != nil {
			log.Fatalf("audit unavailable: %v", err)
		}
		defer auditFile.Close()
	}

	interval := configuredInterval(profile)

	runner := NewProbeRunner(profile, baseline)
	weights := convertWeights(profile.Scoring.Weights)
	scorer := NewScoringEngine(weights, profile.Scoring.MaxHistory)
	executor := NewActionExecutor(profile.Actions, profile.ModelDir, profile.InferenceURL)

	log.Printf("watching at interval %s", interval)

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Run once immediately
	runCycle(runner, scorer, executor)

	for range ticker.C {
		runCycle(runner, scorer, executor)
	}
}

func runCycle(runner *ProbeRunner, scorer *ScoringEngine, executor *ActionExecutor) {
	results := runner.RunAll()
	entry := scorer.Score(results)
	if err := auditLog(GPUAuditEntry{Event: "cycle", Verdict: entry.Verdict, Score: entry.CompositeScore}); err != nil {
		log.Printf("audit persistence failed; action evaluation skipped: %v", err)
		return
	}
	actions := executor.Evaluate(entry)
	if err := auditLog(GPUAuditEntry{Event: "cycle_actions", Verdict: entry.Verdict, Score: entry.CompositeScore, Actions: len(actions)}); err != nil {
		log.Printf("audit persistence failed after action evaluation: %v", err)
		return
	}

	metricChecks.Add(1)
	countStatuses(results)
	countTriggeredActions(actions)

	log.Printf("verdict=%s score=%.2f probes=%d actions=%d",
		entry.Verdict, entry.CompositeScore, len(results), len(actions))

	for _, a := range actions {
		if a.Triggered {
			log.Printf("  action=%s success=%v msg=%s", a.Action, a.Success, a.Message)
		}
	}
}

func cmdBaseline() {
	profile := loadProfile()

	outFile := "baseline.yaml"
	args := os.Args[2:]
	for i, arg := range args {
		if arg == "-out" && i+1 < len(args) {
			outFile = args[i+1]
		}
	}

	if profile.ModelDir == "" {
		log.Fatal("model_dir must be configured in profile to capture baseline")
	}

	patterns := []string{"*.gguf", "*.bin", "*.safetensors"}
	hashes, err := hashModelFiles(profile.ModelDir, patterns)
	if err != nil {
		log.Fatalf("failed to hash model directory: %v", err)
	}

	if len(hashes) == 0 {
		log.Fatalf("no model files found in %s", profile.ModelDir)
	}

	baseline := Baseline{
		CapturedAt:   time.Now().UTC(),
		TensorHashes: hashes,
	}
	if err := enrichBaseline(profile, &baseline); err != nil {
		log.Fatalf("failed to capture extended baseline: %v", err)
	}

	data, err := yaml.Marshal(&baseline)
	if err != nil {
		log.Fatalf("failed to marshal baseline: %v", err)
	}

	if err := writeFileAtomic(outFile, data, 0o600); err != nil {
		log.Fatalf("failed to write baseline: %v", err)
	}

	fmt.Printf("baseline captured: %d files → %s\n", len(hashes), outFile)
}

func cmdStatus() {
	addr := envOr("DAEMON_ADDR", "http://127.0.0.1:8505")
	args := os.Args[2:]
	for i, arg := range args {
		if arg == "-addr" && i+1 < len(args) {
			addr = args[i+1]
		}
	}

	token := readRequiredCredential("SERVICE_TOKEN_PATH", defaultTokenPath)
	statusURL, err := validatedActionURL(addr, "/v1/status")
	if err != nil {
		log.Fatalf("invalid daemon address: %v", err)
	}
	req, err := http.NewRequest(http.MethodGet, statusURL, nil)
	if err != nil {
		log.Fatalf("cannot create daemon request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := outboundHTTPClient.Do(req)
	if err != nil {
		log.Fatalf("cannot reach daemon: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Fatalf("daemon returned HTTP %d", resp.StatusCode)
	}

	var status map[string]interface{}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxProbeResponse)).Decode(&status); err != nil {
		log.Fatalf("invalid daemon response: %v", err)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(status)
}

// ---------- daemon ----------

func cmdDaemon() {
	profile := loadProfile()
	baseline := loadBaseline(profile)
	auditPath := envOr("AUDIT_LOG", defaultAuditLogPath)
	if err := initAudit(auditPath); err != nil {
		log.Fatalf("audit unavailable: %v", err)
	}
	defer auditFile.Close()

	runner := NewProbeRunner(profile, baseline)
	weights := convertWeights(profile.Scoring.Weights)
	scorer := NewScoringEngine(weights, profile.Scoring.MaxHistory)
	executor := NewActionExecutor(profile.Actions, profile.ModelDir, profile.InferenceURL)
	token := readRequiredCredential("SERVICE_TOKEN_PATH", defaultTokenPath)
	incidentToken := ""
	integrationURL := profile.Integrations.IncidentRecorderURL
	if integrationURL != "" {
		path := profile.Integrations.IncidentRecorderTokenPath
		if path == "" {
			path = defaultIncidentTokenPath
		}
		var err error
		incidentToken, err = readCredentialFile(path)
		if err != nil {
			log.Fatalf("incident recorder credential unavailable: %v", err)
		}
	}
	reporter := newIncidentReporter()

	// Background probe cycle
	interval := configuredInterval(profile)
	intervalUpdates := make(chan time.Duration, 1)

	var latestMu sync.Mutex
	var cycleMu sync.Mutex
	checkQueue := make(chan struct{}, 1)
	var latestResults []ProbeResult
	var latestActions []ActionResult

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			cycleMu.Lock()
			results := runner.RunAll()
			entry := scorer.Score(results)
			actions := []ActionResult(nil)
			auditErr := auditLog(GPUAuditEntry{Event: "daemon_cycle", Verdict: entry.Verdict, Score: entry.CompositeScore})
			if auditErr == nil {
				actions = executor.Evaluate(entry)
				auditErr = auditLog(GPUAuditEntry{Event: "daemon_actions", Verdict: entry.Verdict, Score: entry.CompositeScore, Actions: len(actions)})
			}
			reportURL := ""
			reportToken := ""
			if auditErr == nil {
				reportURL = integrationURL
				reportToken = incidentToken
			}
			cycleMu.Unlock()
			if auditErr != nil {
				log.Printf("audit persistence failed; sensitive cycle operations disabled: %v", auditErr)
			}

			metricChecks.Add(1)
			countStatuses(results)
			countTriggeredActions(actions)

			latestMu.Lock()
			latestResults = results
			latestActions = actions
			latestMu.Unlock()

			log.Printf("cycle: verdict=%s score=%.2f", entry.Verdict, entry.CompositeScore)
			if auditErr == nil {
				if err := reporter.Report(reportURL, reportToken, entry, results); err != nil {
					log.Printf("incident reporting failed: %v", err)
				}
			}
			waiting := true
			for waiting {
				select {
				case updated := <-intervalUpdates:
					ticker.Reset(updated)
				case <-ticker.C:
					waiting = false
				}
			}
		}
	}()

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		status := "ok"
		statusCode := http.StatusOK
		if !auditAvailable() {
			status = "unhealthy"
			statusCode = http.StatusServiceUnavailable
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		json.NewEncoder(w).Encode(map[string]string{"status": status})
	})

	mux.HandleFunc("/v1/check", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		select {
		case checkQueue <- struct{}{}:
			defer func() { <-checkQueue }()
		default:
			http.Error(w, "integrity check already running or queued", http.StatusTooManyRequests)
			return
		}

		cycleMu.Lock()
		results := runner.RunAll()
		entry := scorer.Score(results)
		if err := auditLog(GPUAuditEntry{Event: "api_check", Verdict: entry.Verdict, Score: entry.CompositeScore, Source: r.RemoteAddr}); err != nil {
			cycleMu.Unlock()
			http.Error(w, "audit persistence failed", http.StatusInternalServerError)
			return
		}
		actions := executor.Evaluate(entry)
		auditErr := auditLog(GPUAuditEntry{Event: "api_actions", Verdict: entry.Verdict, Score: entry.CompositeScore, Source: r.RemoteAddr, Actions: len(actions)})
		reportURL := integrationURL
		reportToken := incidentToken
		cycleMu.Unlock()
		if auditErr != nil {
			http.Error(w, "audit persistence failed", http.StatusInternalServerError)
			return
		}

		metricChecks.Add(1)
		countStatuses(results)
		countTriggeredActions(actions)

		latestMu.Lock()
		latestResults = results
		latestActions = actions
		latestMu.Unlock()

		if err := reporter.Report(reportURL, reportToken, entry, results); err != nil {
			log.Printf("incident reporting failed: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"probes":  results,
			"score":   entry,
			"actions": actions,
		})
	})

	mux.HandleFunc("/v1/status", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		latest := scorer.Latest()
		trend := scorer.Trend(10)

		latestMu.Lock()
		probes := latestResults
		actions := latestActions
		latestMu.Unlock()

		status := map[string]interface{}{
			"latest":  latest,
			"trend":   trend,
			"probes":  probes,
			"actions": actions,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(status)
	})

	mux.HandleFunc("/v1/history", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(scorer.History())
	})

	mux.HandleFunc("/v1/attest-state", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(buildAttestState(scorer))
	})

	mux.HandleFunc("/v1/baseline", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !checkToken(r, token) {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if os.Getenv("GPU_WATCH_ALLOW_BASELINE_CAPTURE") != "true" {
			http.Error(w, "baseline capture is disabled", http.StatusForbidden)
			return
		}
		if err := auditLog(GPUAuditEntry{Event: "baseline_capture_authorized", Source: r.RemoteAddr}); err != nil {
			http.Error(w, "audit persistence failed", http.StatusInternalServerError)
			return
		}
		cycleMu.Lock()
		defer cycleMu.Unlock()

		if profile.ModelDir == "" {
			http.Error(w, `{"error":"model_dir not configured"}`, http.StatusBadRequest)
			return
		}

		patterns := []string{"*.gguf", "*.bin", "*.safetensors"}
		hashes, err := hashModelFiles(profile.ModelDir, patterns)
		if err != nil {
			http.Error(w, "baseline capture failed", http.StatusInternalServerError)
			return
		}
		if len(hashes) == 0 {
			http.Error(w, "no model files found", http.StatusBadRequest)
			return
		}

		newBaseline := &Baseline{
			CapturedAt:   time.Now().UTC(),
			TensorHashes: hashes,
		}
		if err := enrichBaseline(profile, newBaseline); err != nil {
			http.Error(w, "extended baseline capture failed", http.StatusInternalServerError)
			return
		}

		// Persist baseline
		bPath := profile.BaselineFile
		if bPath == "" {
			bPath = "baseline.yaml"
		}
		data, err := yaml.Marshal(newBaseline)
		if err != nil || writeFileAtomic(bPath, data, 0o600) != nil {
			http.Error(w, "baseline persistence failed", http.StatusInternalServerError)
			return
		}
		runner.baseline = newBaseline

		if err := auditLog(GPUAuditEntry{Event: "baseline_capture", Files: len(hashes), Source: r.RemoteAddr}); err != nil {
			http.Error(w, "audit persistence failed", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "baseline captured",
			"files":  len(hashes),
		})
	})

	mux.HandleFunc("/v1/reload", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !checkToken(r, token) {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		newProfile, err := loadProfileFile(profilePath())
		if err != nil {
			http.Error(w, "profile reload rejected", http.StatusBadRequest)
			return
		}
		newBaseline, err := loadBaselineFile(newProfile)
		if err != nil {
			http.Error(w, "baseline reload rejected", http.StatusBadRequest)
			return
		}
		newIncidentToken := ""
		newIntegrationURL := newProfile.Integrations.IncidentRecorderURL
		if newIntegrationURL != "" {
			path := newProfile.Integrations.IncidentRecorderTokenPath
			if path == "" {
				path = defaultIncidentTokenPath
			}
			var err error
			newIncidentToken, err = readCredentialFile(path)
			if err != nil {
				http.Error(w, "integration credential unavailable", http.StatusInternalServerError)
				return
			}
		}
		if newProfile.Daemon.BindAddr != profile.Daemon.BindAddr {
			http.Error(w, "bind address changes require a restart", http.StatusConflict)
			return
		}
		newInterval := configuredInterval(newProfile)
		if err := auditLog(GPUAuditEntry{Event: "profile_reload_authorized", Source: r.RemoteAddr}); err != nil {
			http.Error(w, "audit persistence failed", http.StatusInternalServerError)
			return
		}

		cycleMu.Lock()
		*profile = *newProfile
		runner.profile = profile
		runner.baseline = newBaseline
		executor.Update(profile.Actions, profile.ModelDir, profile.InferenceURL)
		integrationURL = newIntegrationURL
		incidentToken = newIncidentToken
		scorer.UpdateConfig(convertWeights(profile.Scoring.Weights), profile.Scoring.MaxHistory)
		cycleMu.Unlock()
		select {
		case intervalUpdates <- newInterval:
		default:
			<-intervalUpdates
			intervalUpdates <- newInterval
		}

		if err := auditLog(GPUAuditEntry{Event: "profile_reload", Source: r.RemoteAddr}); err != nil {
			http.Error(w, "audit persistence failed", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "profile reloaded"})
	})

	mux.HandleFunc("/v1/metrics", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]int64{
			"checks_total":    metricChecks.Load(),
			"pass_total":      metricPass.Load(),
			"drift_total":     metricDrift.Load(),
			"fail_total":      metricFail.Load(),
			"actions_total":   metricActions.Load(),
			"http_reqs_total": metricHTTPReqs.Load(),
		})
	})

	addr := profile.Daemon.BindAddr
	if addr == "" {
		addr = "127.0.0.1:8505"
	}

	log.Printf("gpu-integrity-watch daemon listening on %s", addr)
	server := &http.Server{
		Addr:              addr,
		Handler:           authenticatedGPUHandler(mux, token),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      probeCycleTimeout + 30*time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    64 << 10,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("shutting down gpu-integrity-watch...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("server shutdown error: %v", err)
	}
	log.Println("gpu-integrity-watch stopped")
}

// ---------- helpers ----------

func checkToken(r *http.Request, expected string) bool {
	if expected == "" {
		return false
	}
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") {
		return false
	}
	provided := strings.TrimPrefix(auth, "Bearer ")
	return subtle.ConstantTimeCompare([]byte(provided), []byte(expected)) == 1
}

func authenticatedGPUHandler(next http.Handler, token string) http.Handler {
	limiter := newRequestLimiter(120)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" && !checkToken(r, token) {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if r.URL.Path != "/health" && !auditAvailable() {
			http.Error(w, "audit subsystem unavailable", http.StatusServiceUnavailable)
			return
		}
		if r.URL.Path != "/health" && !limiter.Allow() {
			w.Header().Set("Retry-After", "60")
			http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
			return
		}
		next.ServeHTTP(w, r)
	})
}

type requestLimiter struct {
	mu     sync.Mutex
	window time.Time
	count  int
	limit  int
}

func newRequestLimiter(limit int) *requestLimiter {
	return &requestLimiter{window: time.Now(), limit: limit}
}

func (limiter *requestLimiter) Allow() bool {
	limiter.mu.Lock()
	defer limiter.mu.Unlock()
	now := time.Now()
	if now.Sub(limiter.window) >= time.Minute {
		limiter.window = now
		limiter.count = 0
	}
	limiter.count++
	return limiter.count <= limiter.limit
}

func readRequiredCredential(envName, fallback string) string {
	path := envOr(envName, fallback)
	token, err := readCredentialFile(path)
	if err != nil {
		log.Fatalf("cannot read %s: %v", envName, err)
	}
	return token
}

func readCredentialFile(path string) (string, error) {
	data, err := readOwnerOnlyFile(path, 4096)
	if err != nil {
		return "", err
	}
	token := strings.TrimSpace(string(data))
	if len(token) < 32 {
		return "", fmt.Errorf("credential must contain at least 32 characters")
	}
	return token, nil
}

func writeFileAtomic(path string, data []byte, mode os.FileMode) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".gpu-integrity-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	cleanup := func() { tmp.Close(); os.Remove(tmpPath) }
	if err := tmp.Chmod(mode); err != nil {
		cleanup()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		cleanup()
		return err
	}
	if err := tmp.Sync(); err != nil {
		cleanup()
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		return err
	}
	// #nosec G304 -- dir is the parent of the caller-authorized atomic output and this handle is used only for fsync.
	dirHandle, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer dirHandle.Close()
	return dirHandle.Sync()
}

func convertWeights(raw map[string]float64) map[ProbeType]float64 {
	if raw == nil {
		return nil
	}
	out := make(map[ProbeType]float64)
	for k, v := range raw {
		out[ProbeType(k)] = v
	}
	return out
}

func countStatuses(results []ProbeResult) {
	for _, r := range results {
		switch r.Status {
		case StatusPass:
			metricPass.Add(1)
		case StatusDrift:
			metricDrift.Add(1)
		case StatusFail:
			metricFail.Add(1)
		}
	}
}

func countTriggeredActions(actions []ActionResult) {
	for _, action := range actions {
		if action.Triggered {
			metricActions.Add(1)
		}
	}
}
