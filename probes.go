package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	maxProbeResponse   = 1 << 20
	maxProbeOutput     = 1 << 20
	maxModelFileSize   = 10 << 30
	maxDriverFileSize  = 512 << 20
	maxModelFileCount  = 10000
	maxWalkEntryCount  = 50000
	maxTotalModelBytes = 50 << 30
	probeTimeout       = 15 * time.Second
	modelScanTimeout   = 2 * time.Minute
	probeCycleTimeout  = 3 * time.Minute
)

type boundedBuffer struct {
	bytes.Buffer
	limit int
}

func (buffer *boundedBuffer) Write(data []byte) (int, error) {
	if len(data) > buffer.limit-buffer.Len() {
		return 0, fmt.Errorf("output exceeds %d bytes", buffer.limit)
	}
	return buffer.Buffer.Write(data)
}

func runBoundedCommand(executable string, args ...string) ([]byte, error) {
	return runBoundedCommandContext(context.Background(), executable, args...)
}

func runBoundedCommandContext(parent context.Context, executable string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(parent, probeTimeout)
	defer cancel()
	stdout := &boundedBuffer{limit: maxProbeOutput}
	stderr := &boundedBuffer{limit: maxProbeOutput}
	cmd := exec.CommandContext(ctx, executable, args...)
	cmd.Stdin = nil
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	err := cmd.Run()
	if ctx.Err() != nil {
		return nil, fmt.Errorf("command timed out")
	}
	if err != nil {
		return nil, fmt.Errorf("command failed: %w", err)
	}
	return stdout.Bytes(), nil
}

// ProbeType identifies the kind of integrity probe.
type ProbeType string

const (
	ProbeTensorHash        ProbeType = "tensor_hash"
	ProbeSentinelInfer     ProbeType = "sentinel_inference"
	ProbeReferenceDrift    ProbeType = "reference_drift"
	ProbeECCStatus         ProbeType = "ecc_status"
	ProbeDriverFingerprint ProbeType = "driver_fingerprint"
	ProbeDeviceAllowlist   ProbeType = "device_allowlist"
)

// ProbeStatus is the outcome of a single probe run.
type ProbeStatus string

const (
	StatusPass    ProbeStatus = "pass"
	StatusDrift   ProbeStatus = "drift"
	StatusFail    ProbeStatus = "fail"
	StatusSkip    ProbeStatus = "skip"
	StatusError   ProbeStatus = "error"
	StatusUnknown ProbeStatus = "unknown"
)

// Finding records a single observation from a probe.
type Finding struct {
	Description string `json:"description"`
	Severity    string `json:"severity"` // info, warning, critical
	Detail      string `json:"detail,omitempty"`
}

// ProbeResult is the output of running one probe.
type ProbeResult struct {
	Probe     string      `json:"probe"`
	Type      ProbeType   `json:"type"`
	Status    ProbeStatus `json:"status"`
	Score     float64     `json:"score"` // 0.0 = normal, 1.0 = severe
	Findings  []Finding   `json:"findings,omitempty"`
	Timestamp time.Time   `json:"timestamp"`
	Duration  string      `json:"duration"`
}

// Baseline stores known-good state for comparison.
type Baseline struct {
	CapturedAt        time.Time         `json:"captured_at" yaml:"captured_at"`
	TensorHashes      map[string]string `json:"tensor_hashes" yaml:"tensor_hashes"` // file -> sha256
	SentinelRefs      []SentinelRef     `json:"sentinel_refs" yaml:"sentinel_refs"`
	DriverFingerprint *DriverBaseline   `json:"driver_fingerprint,omitempty" yaml:"driver_fingerprint,omitempty"`
	DeviceAllowlist   []string          `json:"device_allowlist,omitempty" yaml:"device_allowlist,omitempty"`
}

type DriverBaseline struct {
	DriverVersion string `json:"driver_version" yaml:"driver_version"`
	KernelModule  string `json:"kernel_module" yaml:"kernel_module"`
	ModuleHash    string `json:"module_hash,omitempty" yaml:"module_hash,omitempty"`
}

// SentinelRef is a known input/output pair for sentinel inference.
type SentinelRef struct {
	Name     string `json:"name" yaml:"name"`
	Input    string `json:"input" yaml:"input"`
	Expected string `json:"expected" yaml:"expected"`
}

// ProbeRunner executes configured probes against the current system state.
type ProbeRunner struct {
	profile  *IntegrityProfile
	baseline *Baseline
}

// NewProbeRunner creates a runner with the given profile and baseline.
func NewProbeRunner(profile *IntegrityProfile, baseline *Baseline) *ProbeRunner {
	return &ProbeRunner{profile: profile, baseline: baseline}
}

// RunAll executes all enabled probes and returns results.
func (r *ProbeRunner) RunAll() []ProbeResult {
	ctx, cancel := context.WithTimeout(context.Background(), probeCycleTimeout)
	defer cancel()
	var results []ProbeResult

	for _, pc := range r.profile.Probes {
		if !pc.Enabled {
			continue
		}
		start := time.Now()
		var result ProbeResult

		switch pc.Type {
		case ProbeTensorHash:
			result = r.runTensorHashContext(ctx, pc)
		case ProbeSentinelInfer:
			result = r.runSentinelInferenceContext(ctx, pc)
		case ProbeReferenceDrift:
			result = r.runReferenceDriftContext(ctx, pc)
		case ProbeECCStatus:
			result = r.runECCStatusContext(ctx, pc)
		case ProbeDriverFingerprint:
			result = r.runDriverFingerprintContext(ctx, pc)
		case ProbeDeviceAllowlist:
			result = r.runDeviceAllowlist(pc)
		default:
			result = ProbeResult{
				Probe:  pc.Name,
				Type:   pc.Type,
				Status: StatusError,
				Score:  0.5,
				Findings: []Finding{{
					Description: fmt.Sprintf("unknown probe type: %s", pc.Type),
					Severity:    "warning",
				}},
			}
		}

		result.Timestamp = time.Now()
		result.Duration = time.Since(start).String()
		results = append(results, result)
	}

	return results
}

func (r *ProbeRunner) runDriverFingerprint(pc ProbeConfig) ProbeResult {
	return r.runDriverFingerprintContext(context.Background(), pc)
}

func (r *ProbeRunner) runDriverFingerprintContext(ctx context.Context, pc ProbeConfig) ProbeResult {
	result := ProbeResult{Probe: pc.Name, Type: ProbeDriverFingerprint}
	if err := ctx.Err(); err != nil {
		result.Status, result.Score = StatusError, 1
		result.Findings = append(result.Findings, Finding{Description: "probe cycle deadline exceeded", Severity: "critical"})
		return result
	}
	version, module, moduleHash, err := detectDriverFingerprintContext(ctx, pc.Settings)
	if err != nil {
		result.Status = StatusError
		result.Score = 1
		result.Findings = append(result.Findings, Finding{Description: "driver fingerprint collection failed", Severity: "critical", Detail: err.Error()})
		return result
	}
	if version == "" && module == "" {
		result.Status = StatusSkip
		result.Findings = append(result.Findings, Finding{Description: "no GPU driver detected", Severity: "warning"})
		return result
	}
	if r.baseline == nil || r.baseline.DriverFingerprint == nil {
		result.Status = StatusError
		result.Score = 1
		result.Findings = append(result.Findings, Finding{Description: "driver baseline is missing", Severity: "critical"})
		return result
	}
	expected := r.baseline.DriverFingerprint
	if expected.DriverVersion != version || expected.KernelModule != module ||
		(expected.ModuleHash != "" && expected.ModuleHash != moduleHash) {
		result.Status = StatusFail
		result.Score = 1
		result.Findings = append(result.Findings, Finding{
			Description: "GPU driver fingerprint differs from baseline", Severity: "critical",
			Detail: fmt.Sprintf("expected version=%q module=%q; got version=%q module=%q", expected.DriverVersion, expected.KernelModule, version, module),
		})
		return result
	}
	result.Status = StatusPass
	result.Findings = append(result.Findings, Finding{Description: "GPU driver fingerprint matches baseline", Severity: "info"})
	return result
}

func detectDriverFingerprint(settings map[string]string) (version, module, moduleHash string, err error) {
	return detectDriverFingerprintContext(context.Background(), settings)
}

func detectDriverFingerprintContext(ctx context.Context, settings map[string]string) (version, module, moduleHash string, err error) {
	module = strings.TrimSpace(settings["kernel_module"])
	if custom := strings.TrimSpace(settings["driver_version_path"]); custom != "" {
		if !filepath.IsAbs(custom) || filepath.Clean(custom) != custom {
			return "", "", "", fmt.Errorf("driver_version_path must be canonical and absolute")
		}
		data, readErr := readSmallFile(custom, 4096)
		if readErr != nil {
			return "", "", "", readErr
		}
		version = strings.TrimSpace(string(data))
	} else {
		for _, candidate := range []struct{ module, path string }{
			{"nvidia", "/sys/module/nvidia/version"},
			{"amdgpu", "/sys/module/amdgpu/version"},
			{"i915", "/sys/module/i915/version"},
			{"xe", "/sys/module/xe/version"},
		} {
			if data, readErr := readSmallFile(candidate.path, 4096); readErr == nil {
				version = strings.TrimSpace(string(data))
				if module == "" {
					module = candidate.module
				}
				break
			}
		}
	}
	if version == "" {
		custom := strings.TrimSpace(settings["nvidia_smi_path"])
		resolved, lookupErr := resolveNvidiaSMI(custom)
		if lookupErr == nil {
			output, commandErr := runBoundedCommandContext(ctx, resolved, "--query-gpu=driver_version", "--format=csv,noheader,nounits")
			if commandErr != nil {
				return "", "", "", commandErr
			}
			version = strings.TrimSpace(strings.Split(string(output), "\n")[0])
			if module == "" {
				module = "nvidia"
			}
		} else if custom != "" || !errors.Is(lookupErr, exec.ErrNotFound) {
			return "", "", "", lookupErr
		}
	}
	if modulePath := strings.TrimSpace(settings["module_path"]); modulePath != "" {
		if !filepath.IsAbs(modulePath) || filepath.Clean(modulePath) != modulePath {
			return "", "", "", fmt.Errorf("module_path must be canonical and absolute")
		}
		moduleHash, err = hashFileBounded(modulePath, maxDriverFileSize)
		if err != nil {
			return "", "", "", err
		}
	}
	return version, module, moduleHash, nil
}

func resolveNvidiaSMI(configured string) (string, error) {
	path := configured
	if path == "" {
		var err error
		path, err = exec.LookPath("nvidia-smi")
		if err != nil {
			return "", err
		}
		if !filepath.IsAbs(path) {
			path, err = filepath.Abs(path)
			if err != nil {
				return "", err
			}
		}
	}
	return validateTrustedExecutable(path)
}

func validateTrustedExecutable(path string) (string, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return "", fmt.Errorf("nvidia_smi_path must be canonical and absolute")
	}
	info, err := os.Lstat(path)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm()&0o111 == 0 || info.Mode().Perm()&0o022 != 0 {
		return "", fmt.Errorf("nvidia_smi_path must be a non-writable regular executable")
	}
	if !trustedFileOwner(info) {
		return "", fmt.Errorf("nvidia_smi_path has an untrusted owner")
	}
	return path, nil
}

func readSmallFile(path string, limit int64) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return nil, fmt.Errorf("unsafe driver evidence path")
	}
	// #nosec G304 -- path was Lstat-validated as regular non-symlink driver evidence and is identity-checked after open.
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(info, opened) {
		return nil, fmt.Errorf("driver evidence changed while opening")
	}
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil || int64(len(data)) > limit {
		return nil, fmt.Errorf("driver evidence exceeds size limit")
	}
	after, err := f.Stat()
	if err != nil || !os.SameFile(opened, after) || after.Size() != int64(len(data)) {
		return nil, fmt.Errorf("driver evidence changed while reading")
	}
	return data, nil
}

func (r *ProbeRunner) runDeviceAllowlist(pc ProbeConfig) ProbeResult {
	result := ProbeResult{Probe: pc.Name, Type: ProbeDeviceAllowlist}
	directories := []string{"/dev/dri", "/dev"}
	if custom := strings.TrimSpace(pc.Settings["device_dir"]); custom != "" {
		if !filepath.IsAbs(custom) || filepath.Clean(custom) != custom {
			result.Status = StatusError
			result.Score = 1
			return result
		}
		directories = []string{custom}
	}
	current := make([]string, 0)
	for _, directory := range directories {
		devices, err := discoverGPUDevices(directory)
		if err != nil && !os.IsNotExist(err) {
			result.Status = StatusError
			result.Score = 1
			result.Findings = append(result.Findings, Finding{Description: "GPU device discovery failed", Severity: "critical"})
			return result
		}
		current = append(current, devices...)
	}
	sort.Strings(current)
	current = uniqueStrings(current)
	if r.baseline == nil || len(r.baseline.DeviceAllowlist) == 0 {
		result.Status = StatusError
		result.Score = 1
		result.Findings = append(result.Findings, Finding{Description: "GPU device allowlist baseline is missing", Severity: "critical"})
		return result
	}
	expected := make(map[string]bool, len(r.baseline.DeviceAllowlist))
	for _, device := range r.baseline.DeviceAllowlist {
		expected[device] = true
	}
	actual := make(map[string]bool, len(current))
	for _, device := range current {
		actual[device] = true
		if !expected[device] {
			result.Findings = append(result.Findings, Finding{Description: "unexpected GPU device: " + device, Severity: "critical"})
		}
	}
	failed := false
	for device := range expected {
		if !actual[device] {
			failed = true
			result.Findings = append(result.Findings, Finding{Description: "expected GPU device missing: " + device, Severity: "critical"})
		}
	}
	for device := range actual {
		if !expected[device] {
			failed = true
		}
	}
	if failed {
		result.Status = StatusFail
		result.Score = 1
	} else {
		result.Status = StatusPass
		result.Findings = append(result.Findings, Finding{Description: "GPU device nodes match baseline", Severity: "info"})
	}
	return result
}

func discoverGPUDevices(directory string) ([]string, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	devices := make([]string, 0)
	for _, entry := range entries {
		name := entry.Name()
		if !(strings.HasPrefix(name, "card") || strings.HasPrefix(name, "renderD") || strings.HasPrefix(name, "nvidia")) {
			continue
		}
		path := filepath.Join(directory, name)
		info, err := os.Lstat(path)
		if err != nil {
			return nil, err
		}
		if info.Mode()&os.ModeSymlink != 0 || info.IsDir() {
			continue
		}
		if info.Mode()&os.ModeDevice != 0 {
			devices = append(devices, path)
		}
	}
	return devices, nil
}

func uniqueStrings(values []string) []string {
	if len(values) == 0 {
		return values
	}
	result := values[:1]
	for _, value := range values[1:] {
		if value != result[len(result)-1] {
			result = append(result, value)
		}
	}
	return result
}

func enrichBaseline(profile *IntegrityProfile, baseline *Baseline) error {
	for _, probe := range profile.Probes {
		if !probe.Enabled {
			continue
		}
		switch probe.Type {
		case ProbeDriverFingerprint:
			version, module, moduleHash, err := detectDriverFingerprint(probe.Settings)
			if err != nil {
				return err
			}
			if version == "" || module == "" {
				return fmt.Errorf("cannot capture driver baseline without driver evidence")
			}
			baseline.DriverFingerprint = &DriverBaseline{DriverVersion: version, KernelModule: module, ModuleHash: moduleHash}
		case ProbeDeviceAllowlist:
			directories := []string{"/dev/dri", "/dev"}
			if custom := strings.TrimSpace(probe.Settings["device_dir"]); custom != "" {
				directories = []string{custom}
			}
			devices := make([]string, 0)
			for _, directory := range directories {
				found, err := discoverGPUDevices(directory)
				if err != nil && !os.IsNotExist(err) {
					return err
				}
				devices = append(devices, found...)
			}
			sort.Strings(devices)
			devices = uniqueStrings(devices)
			if len(devices) == 0 {
				return fmt.Errorf("cannot capture device allowlist without GPU devices")
			}
			baseline.DeviceAllowlist = devices
		}
	}
	return nil
}

func validateBaseline(baseline *Baseline) error {
	if baseline == nil || baseline.CapturedAt.IsZero() {
		return fmt.Errorf("baseline captured_at is required")
	}
	for path, digest := range baseline.TensorHashes {
		if path == "" || filepath.IsAbs(path) || filepath.Clean(path) != path || strings.HasPrefix(path, "..") ||
			len(digest) != sha256.Size*2 || strings.ToLower(digest) != digest {
			return fmt.Errorf("invalid tensor baseline entry")
		}
		if _, err := hex.DecodeString(digest); err != nil {
			return fmt.Errorf("invalid tensor baseline digest")
		}
	}
	if len(baseline.SentinelRefs) > 1000 {
		return fmt.Errorf("too many sentinel references")
	}
	if driver := baseline.DriverFingerprint; driver != nil {
		if driver.DriverVersion == "" || driver.KernelModule == "" {
			return fmt.Errorf("driver baseline is incomplete")
		}
		if driver.ModuleHash != "" {
			if len(driver.ModuleHash) != sha256.Size*2 {
				return fmt.Errorf("invalid driver module digest")
			}
			if _, err := hex.DecodeString(driver.ModuleHash); err != nil {
				return fmt.Errorf("invalid driver module digest")
			}
		}
	}
	seenDevices := make(map[string]bool)
	for _, device := range baseline.DeviceAllowlist {
		if !filepath.IsAbs(device) || filepath.Clean(device) != device || seenDevices[device] {
			return fmt.Errorf("invalid or duplicate GPU device baseline")
		}
		seenDevices[device] = true
	}
	return nil
}

// runTensorHash computes SHA-256 of model files and compares to baseline.
func (r *ProbeRunner) runTensorHash(pc ProbeConfig) ProbeResult {
	return r.runTensorHashContext(context.Background(), pc)
}

func (r *ProbeRunner) runTensorHashContext(ctx context.Context, pc ProbeConfig) ProbeResult {
	result := ProbeResult{Probe: pc.Name, Type: ProbeTensorHash}

	modelDir := pc.Settings["model_dir"]
	if modelDir == "" {
		modelDir = r.profile.ModelDir
	}
	if modelDir == "" {
		result.Status = StatusSkip
		result.Score = 0.0
		result.Findings = append(result.Findings, Finding{
			Description: "no model directory configured",
			Severity:    "info",
		})
		return result
	}

	if r.baseline == nil || len(r.baseline.TensorHashes) == 0 {
		result.Status = StatusSkip
		result.Score = 0.0
		result.Findings = append(result.Findings, Finding{
			Description: "no baseline hashes available; run baseline capture first",
			Severity:    "info",
		})
		return result
	}

	patterns := []string{"*.gguf", "*.bin", "*.safetensors"}
	if p, ok := pc.Settings["patterns"]; ok {
		patterns = strings.Split(p, ",")
	}

	currentHashes, err := hashModelFilesContext(ctx, modelDir, patterns)
	if err != nil {
		result.Status = StatusError
		result.Score = 1.0
		result.Findings = append(result.Findings, Finding{
			Description: "model scan failed", Severity: "critical", Detail: err.Error(),
		})
		return result
	}
	if len(currentHashes) == 0 && len(r.baseline.TensorHashes) == 0 {
		result.Status = StatusError
		result.Score = 0.5
		result.Findings = append(result.Findings, Finding{
			Description: "no model files found in " + modelDir,
			Severity:    "warning",
		})
		return result
	}

	mismatches := 0
	missing := 0
	extras := 0
	for file, baselineHash := range r.baseline.TensorHashes {
		currentHash, exists := currentHashes[file]
		if !exists {
			missing++
			result.Findings = append(result.Findings, Finding{
				Description: fmt.Sprintf("baseline file missing: %s", file),
				Severity:    "critical",
				Detail:      "expected hash: " + baselineHash,
			})
			continue
		}
		if currentHash != baselineHash {
			mismatches++
			result.Findings = append(result.Findings, Finding{
				Description: fmt.Sprintf("hash mismatch: %s", file),
				Severity:    "critical",
				Detail:      fmt.Sprintf("expected=%s got=%s", baselineHash, currentHash),
			})
		}
	}

	// Check for new files not in baseline
	for file := range currentHashes {
		if _, inBaseline := r.baseline.TensorHashes[file]; !inBaseline {
			extras++
			result.Findings = append(result.Findings, Finding{
				Description: fmt.Sprintf("new file not in baseline: %s", file),
				Severity:    "warning",
			})
		}
	}

	total := len(r.baseline.TensorHashes)
	if total == 0 {
		total = 1
	}
	failCount := mismatches + missing + extras

	if failCount == 0 {
		result.Status = StatusPass
		result.Score = 0.0
	} else {
		result.Status = StatusFail
		result.Score = float64(failCount) / float64(total)
		if result.Score > 1.0 {
			result.Score = 1.0
		}
	}

	return result
}

// hashModelFiles walks the directory and hashes files matching patterns.
func hashModelFiles(dir string, patterns []string) (map[string]string, error) {
	return hashModelFilesContext(context.Background(), dir, patterns)
}

func hashModelFilesContext(parent context.Context, dir string, patterns []string) (map[string]string, error) {
	hashes := make(map[string]string)
	ctx, cancel := context.WithTimeout(parent, modelScanTimeout)
	defer cancel()
	root, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}
	entries := 0
	var totalBytes int64
	err = filepath.Walk(root, func(path string, info os.FileInfo, walkErr error) error {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("model scan deadline exceeded: %w", err)
		}
		if walkErr != nil {
			return walkErr
		}
		entries++
		if entries > maxWalkEntryCount {
			return fmt.Errorf("model tree exceeds %d total entries", maxWalkEntryCount)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("symlink encountered: %s", path)
		}
		if info.IsDir() {
			return nil
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("non-regular entry encountered: %s", path)
		}
		name := info.Name()
		for _, pattern := range patterns {
			matched, matchErr := filepath.Match(strings.TrimSpace(pattern), name)
			if matchErr != nil {
				return fmt.Errorf("invalid model pattern: %w", matchErr)
			}
			if matched {
				if info.Size() < 0 || info.Size() > maxModelFileSize {
					return fmt.Errorf("model exceeds size limit: %s", path)
				}
				if info.Size() > maxTotalModelBytes-totalBytes {
					return fmt.Errorf("model tree exceeds %d total bytes", maxTotalModelBytes)
				}
				totalBytes += info.Size()
				h, err := hashFileBoundedContext(ctx, path, maxModelFileSize)
				if err != nil {
					return err
				}
				rel, err := filepath.Rel(root, path)
				if err != nil || strings.HasPrefix(rel, "..") {
					return fmt.Errorf("model escaped configured root")
				}
				hashes[rel] = h
				if len(hashes) > maxModelFileCount {
					return fmt.Errorf("model count exceeds %d", maxModelFileCount)
				}
				break
			}
		}
		return nil
	})

	if err != nil {
		return nil, err
	}
	return hashes, nil
}

// hashFile computes the SHA-256 hex digest of a file.
func hashFile(path string) (string, error) {
	return hashFileBounded(path, maxModelFileSize)
}

func hashFileBounded(path string, limit int64) (string, error) {
	return hashFileBoundedContext(context.Background(), path, limit)
}

func hashFileBoundedContext(ctx context.Context, path string, limit int64) (string, error) {
	before, err := os.Lstat(path)
	if err != nil || !before.Mode().IsRegular() || before.Mode()&os.ModeSymlink != 0 ||
		before.Size() < 0 || before.Size() > limit {
		return "", fmt.Errorf("unsafe model file")
	}
	// #nosec G304 -- path was root-constrained and Lstat-validated as a bounded regular non-symlink, then identity-checked after open.
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(before, opened) {
		return "", fmt.Errorf("model changed while opening")
	}

	h := sha256.New()
	buffer := make([]byte, 128<<10)
	var written int64
	for {
		if err := ctx.Err(); err != nil {
			return "", fmt.Errorf("hash deadline exceeded: %w", err)
		}
		read, readErr := f.Read(buffer)
		if read > 0 {
			written += int64(read)
			if written > limit {
				return "", fmt.Errorf("file exceeds size limit")
			}
			if _, err := h.Write(buffer[:read]); err != nil {
				return "", err
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", readErr
		}
	}
	after, err := f.Stat()
	if err != nil || !os.SameFile(opened, after) || opened.Size() != after.Size() || written != opened.Size() {
		return "", fmt.Errorf("model changed while hashing")
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// runSentinelInference sends known inputs to the inference endpoint and checks outputs.
func (r *ProbeRunner) runSentinelInference(pc ProbeConfig) ProbeResult {
	return r.runSentinelInferenceContext(context.Background(), pc)
}

func (r *ProbeRunner) runSentinelInferenceContext(ctx context.Context, pc ProbeConfig) ProbeResult {
	result := ProbeResult{Probe: pc.Name, Type: ProbeSentinelInfer}

	endpoint := pc.Settings["inference_url"]
	if endpoint == "" {
		endpoint = r.profile.InferenceURL
	}
	if endpoint == "" {
		result.Status = StatusSkip
		result.Score = 0.0
		result.Findings = append(result.Findings, Finding{
			Description: "no inference endpoint configured",
			Severity:    "info",
		})
		return result
	}

	if r.baseline == nil || len(r.baseline.SentinelRefs) == 0 {
		result.Status = StatusSkip
		result.Score = 0.0
		result.Findings = append(result.Findings, Finding{
			Description: "no sentinel references in baseline",
			Severity:    "info",
		})
		return result
	}

	drifts := 0
	fails := 0
	total := len(r.baseline.SentinelRefs)

	for _, ref := range r.baseline.SentinelRefs {
		if err := ctx.Err(); err != nil {
			result.Status, result.Score = StatusError, 1
			result.Findings = append(result.Findings, Finding{Description: "probe cycle deadline exceeded", Severity: "critical"})
			return result
		}
		actual, err := querySentinelContext(ctx, endpoint, ref.Input)
		if err != nil {
			fails++
			result.Findings = append(result.Findings, Finding{
				Description: fmt.Sprintf("sentinel %q: request failed", ref.Name),
				Severity:    "critical",
				Detail:      err.Error(),
			})
			continue
		}

		similarity := computeSimilarity(ref.Expected, actual)
		threshold := 0.9
		if t, ok := pc.Settings["similarity_threshold"]; ok {
			parsed, err := strconv.ParseFloat(t, 64)
			if err != nil || parsed < 0 || parsed > 1 {
				result.Status = StatusError
				result.Score = 1
				result.Findings = append(result.Findings, Finding{Description: "invalid similarity threshold", Severity: "critical"})
				return result
			}
			threshold = parsed
		}

		if similarity >= threshold {
			result.Findings = append(result.Findings, Finding{
				Description: fmt.Sprintf("sentinel %q: pass (similarity=%.2f)", ref.Name, similarity),
				Severity:    "info",
			})
		} else if similarity >= 0.5 {
			drifts++
			result.Findings = append(result.Findings, Finding{
				Description: fmt.Sprintf("sentinel %q: drift (similarity=%.2f)", ref.Name, similarity),
				Severity:    "warning",
				Detail:      fmt.Sprintf("expected=%q got=%q", ref.Expected, actual),
			})
		} else {
			fails++
			result.Findings = append(result.Findings, Finding{
				Description: fmt.Sprintf("sentinel %q: fail (similarity=%.2f)", ref.Name, similarity),
				Severity:    "critical",
				Detail:      fmt.Sprintf("expected=%q got=%q", ref.Expected, actual),
			})
		}
	}

	switch {
	case fails > 0:
		result.Status = StatusFail
		result.Score = float64(fails+drifts) / float64(total)
	case drifts > 0:
		result.Status = StatusDrift
		result.Score = float64(drifts) / float64(total) * 0.5
	default:
		result.Status = StatusPass
		result.Score = 0.0
	}

	if result.Score > 1.0 {
		result.Score = 1.0
	}
	return result
}

// querySentinel sends a completion request to the inference endpoint.
func querySentinel(endpoint, input string) (string, error) {
	return querySentinelContext(context.Background(), endpoint, input)
}

func querySentinelContext(ctx context.Context, endpoint, input string) (string, error) {
	base, err := parseServiceURL(endpoint)
	if err != nil {
		return "", fmt.Errorf("invalid inference endpoint: %w", err)
	}
	base.Path = strings.TrimRight(base.Path, "/") + "/completion"
	payload, err := json.Marshal(map[string]any{"prompt": input, "n_predict": 64, "temperature": 0})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base.String(), bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := outboundHTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxProbeResponse+1))
	if err != nil {
		return "", err
	}
	if len(body) > maxProbeResponse {
		return "", fmt.Errorf("inference response exceeds size limit")
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("inference returned HTTP %d", resp.StatusCode)
	}
	var response struct {
		Content string `json:"content"`
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	if err := decoder.Decode(&response); err != nil {
		return "", fmt.Errorf("invalid inference response: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return "", fmt.Errorf("inference response has trailing data")
	}
	if response.Content == "" {
		return "", fmt.Errorf("inference response omitted content")
	}
	return response.Content, nil
}

// computeSimilarity returns a simple word-overlap similarity between two strings.
func computeSimilarity(expected, actual string) float64 {
	if expected == actual {
		return 1.0
	}

	expectedWords := strings.Fields(strings.ToLower(expected))
	actualWords := strings.Fields(strings.ToLower(actual))

	if len(expectedWords) == 0 || len(actualWords) == 0 {
		return 0.0
	}

	actualSet := make(map[string]bool)
	for _, w := range actualWords {
		actualSet[w] = true
	}

	matches := 0
	for _, w := range expectedWords {
		if actualSet[w] {
			matches++
		}
	}

	// Jaccard-like similarity
	union := len(expectedWords)
	for _, w := range actualWords {
		found := false
		for _, e := range expectedWords {
			if w == e {
				found = true
				break
			}
		}
		if !found {
			union++
		}
	}

	if union == 0 {
		return 0.0
	}
	return float64(matches) / float64(union)
}

// runReferenceDrift checks if outputs have drifted from baseline over time.
func (r *ProbeRunner) runReferenceDrift(pc ProbeConfig) ProbeResult {
	return r.runReferenceDriftContext(context.Background(), pc)
}

func (r *ProbeRunner) runReferenceDriftContext(ctx context.Context, pc ProbeConfig) ProbeResult {
	result := ProbeResult{Probe: pc.Name, Type: ProbeReferenceDrift}

	endpoint := pc.Settings["inference_url"]
	if endpoint == "" {
		endpoint = r.profile.InferenceURL
	}
	if endpoint == "" || r.baseline == nil || len(r.baseline.SentinelRefs) == 0 {
		result.Status = StatusSkip
		result.Score = 0.0
		result.Findings = append(result.Findings, Finding{
			Description: "drift check skipped: no endpoint or baseline",
			Severity:    "info",
		})
		return result
	}

	// Run each sentinel multiple times to detect variance
	iterations := 3
	if n, ok := pc.Settings["iterations"]; ok {
		parsed, err := strconv.Atoi(n)
		if err != nil || parsed < 1 || parsed > 10 {
			result.Status = StatusError
			result.Score = 1
			return result
		}
		iterations = parsed
	}

	var totalDrift float64
	probeCount := 0
	requestErrors := 0

	for _, ref := range r.baseline.SentinelRefs {
		var similarities []float64
		for i := 0; i < iterations; i++ {
			if err := ctx.Err(); err != nil {
				result.Status, result.Score = StatusError, 1
				result.Findings = append(result.Findings, Finding{Description: "probe cycle deadline exceeded", Severity: "critical"})
				return result
			}
			actual, err := querySentinelContext(ctx, endpoint, ref.Input)
			if err != nil {
				requestErrors++
				result.Findings = append(result.Findings, Finding{
					Description: fmt.Sprintf("drift check %q iteration %d: error", ref.Name, i+1),
					Severity:    "warning",
					Detail:      err.Error(),
				})
				continue
			}
			sim := computeSimilarity(ref.Expected, actual)
			similarities = append(similarities, sim)
		}

		if len(similarities) == 0 {
			continue
		}

		avg := 0.0
		for _, s := range similarities {
			avg += s
		}
		avg /= float64(len(similarities))

		// Check variance (inconsistent outputs suggest corruption)
		variance := 0.0
		for _, s := range similarities {
			d := s - avg
			variance += d * d
		}
		variance /= float64(len(similarities))

		drift := 1.0 - avg
		totalDrift += drift
		probeCount++

		severity := "info"
		if drift > 0.3 {
			severity = "warning"
		}
		if drift > 0.7 || variance > 0.1 {
			severity = "critical"
		}

		result.Findings = append(result.Findings, Finding{
			Description: fmt.Sprintf("drift %q: avg_similarity=%.2f variance=%.4f", ref.Name, avg, variance),
			Severity:    severity,
		})
	}
	if requestErrors > 0 {
		result.Status = StatusError
		result.Score = 1
		return result
	}

	if probeCount == 0 {
		result.Status = StatusSkip
		result.Score = 0.0
		return result
	}

	avgDrift := totalDrift / float64(probeCount)
	result.Score = avgDrift

	switch {
	case avgDrift < 0.1:
		result.Status = StatusPass
	case avgDrift < 0.5:
		result.Status = StatusDrift
	default:
		result.Status = StatusFail
	}

	return result
}

// runECCStatus checks GPU ECC memory error counters.
func (r *ProbeRunner) runECCStatus(pc ProbeConfig) ProbeResult {
	return r.runECCStatusContext(context.Background(), pc)
}

func (r *ProbeRunner) runECCStatusContext(ctx context.Context, pc ProbeConfig) ProbeResult {
	result := ProbeResult{Probe: pc.Name, Type: ProbeECCStatus}
	if err := ctx.Err(); err != nil {
		result.Status, result.Score = StatusError, 1
		result.Findings = append(result.Findings, Finding{Description: "probe cycle deadline exceeded", Severity: "critical"})
		return result
	}

	nvidiaSmi, err := resolveNvidiaSMI(strings.TrimSpace(pc.Settings["nvidia_smi_path"]))
	if err != nil {
		result.Status = StatusUnknown
		result.Score = 1.0
		result.Findings = append(result.Findings, Finding{
			Description: "trusted nvidia-smi unavailable; ECC state is unknown",
			Severity:    "critical",
		})
		return result
	}

	// Query ECC errors
	out, err := runBoundedCommandContext(ctx, nvidiaSmi,
		"--query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,ecc.mode.current",
		"--format=csv,noheader,nounits")
	if err != nil {
		result.Status = StatusError
		result.Score = 0.3
		result.Findings = append(result.Findings, Finding{
			Description: "failed to query ECC status",
			Severity:    "warning",
			Detail:      err.Error(),
		})
		return result
	}

	lines := strings.TrimSpace(string(out))
	result = parseECCOutput(result, lines)
	return result
}

// parseECCOutput parses nvidia-smi ECC output and updates the probe result.
func parseECCOutput(result ProbeResult, output string) ProbeResult {
	reader := csv.NewReader(strings.NewReader(output))
	reader.FieldsPerRecord = -1
	reader.TrimLeadingSpace = true
	worst := StatusPass
	worstRank := 0
	worstScore := 0.0
	rows := 0
	setWorst := func(status ProbeStatus, rank int, score float64) {
		if rank > worstRank {
			worst, worstRank, worstScore = status, rank, score
		} else if rank == worstRank && score > worstScore {
			worstScore = score
		}
	}
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			result.Findings = append(result.Findings, Finding{Description: "malformed ECC CSV output", Severity: "critical", Detail: err.Error()})
			setWorst(StatusUnknown, 2, 1)
			break
		}
		rows++
		gpu := fmt.Sprintf("GPU %d", rows-1)
		if len(record) != 3 {
			result.Findings = append(result.Findings, Finding{Description: gpu + ": ECC row must contain exactly three fields", Severity: "critical"})
			setWorst(StatusUnknown, 2, 1)
			continue
		}
		for i := range record {
			record[i] = strings.TrimSpace(record[i])
		}
		correctedRaw, uncorrectedRaw, mode := record[0], record[1], record[2]
		if correctedRaw == "N/A" || uncorrectedRaw == "N/A" || mode == "N/A" || strings.Contains(mode, "Not Supported") {
			result.Findings = append(result.Findings, Finding{Description: gpu + ": ECC data unavailable", Severity: "critical"})
			setWorst(StatusUnknown, 2, 1)
			continue
		}
		corrected, correctedErr := strconv.ParseUint(correctedRaw, 10, 64)
		uncorrected, uncorrectedErr := strconv.ParseUint(uncorrectedRaw, 10, 64)
		if correctedErr != nil || uncorrectedErr != nil || (mode != "Enabled" && mode != "Disabled") {
			result.Findings = append(result.Findings, Finding{Description: gpu + ": malformed ECC fields", Severity: "critical"})
			setWorst(StatusUnknown, 2, 1)
			continue
		}
		if uncorrected > 0 {
			result.Findings = append(result.Findings, Finding{Description: fmt.Sprintf("%s: uncorrected ECC errors detected: %d", gpu, uncorrected), Severity: "critical", Detail: "uncorrected memory errors may corrupt model weights in VRAM"})
			setWorst(StatusFail, 3, 1)
			continue
		}
		if mode == "Disabled" {
			result.Findings = append(result.Findings, Finding{Description: gpu + ": ECC is disabled", Severity: "warning", Detail: "enable ECC for production AI workloads"})
			setWorst(StatusDrift, 1, .3)
			continue
		}
		if corrected > 100 {
			result.Findings = append(result.Findings, Finding{Description: fmt.Sprintf("%s: high corrected ECC error count: %d", gpu, corrected), Severity: "warning", Detail: "elevated corrected errors may indicate degrading memory"})
			setWorst(StatusDrift, 1, .4)
			continue
		}
		result.Findings = append(result.Findings, Finding{Description: fmt.Sprintf("%s: ECC enabled and healthy (corrected=%d)", gpu, corrected), Severity: "info"})
	}
	if rows == 0 {
		result.Findings = append(result.Findings, Finding{Description: "ECC output contained no GPU records", Severity: "critical"})
		setWorst(StatusUnknown, 2, 1)
	}
	result.Status = worst
	result.Score = worstScore
	return result
}
