package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// ---------- probe tests ----------

func TestTensorHash_Pass(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "model.gguf", "model-data-here")
	hash := sha256Hex("model-data-here")

	profile := &IntegrityProfile{ModelDir: dir}
	baseline := &Baseline{TensorHashes: map[string]string{"model.gguf": hash}}
	runner := NewProbeRunner(profile, baseline)

	result := runner.runTensorHash(ProbeConfig{
		Name: "hash-test", Type: ProbeTensorHash, Enabled: true,
	})

	if result.Status != StatusPass {
		t.Errorf("expected pass, got %s", result.Status)
	}
	if result.Score != 0.0 {
		t.Errorf("expected score 0.0, got %f", result.Score)
	}
}

func TestTensorHash_Mismatch(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "model.gguf", "corrupted-data")

	profile := &IntegrityProfile{ModelDir: dir}
	baseline := &Baseline{TensorHashes: map[string]string{"model.gguf": "deadbeef"}}
	runner := NewProbeRunner(profile, baseline)

	result := runner.runTensorHash(ProbeConfig{
		Name: "hash-test", Type: ProbeTensorHash, Enabled: true,
	})

	if result.Status != StatusFail {
		t.Errorf("expected fail, got %s", result.Status)
	}
	if result.Score == 0.0 {
		t.Error("expected nonzero score for mismatch")
	}

	hasMismatch := false
	for _, f := range result.Findings {
		if strings.Contains(f.Description, "hash mismatch") {
			hasMismatch = true
		}
	}
	if !hasMismatch {
		t.Error("expected hash mismatch finding")
	}
}

func TestTensorHash_MissingBaseline(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "model.gguf", "data")

	profile := &IntegrityProfile{ModelDir: dir}
	runner := NewProbeRunner(profile, nil)

	result := runner.runTensorHash(ProbeConfig{
		Name: "hash-test", Type: ProbeTensorHash, Enabled: true,
	})

	if result.Status != StatusSkip {
		t.Errorf("expected skip without baseline, got %s", result.Status)
	}
}

func TestTensorHash_NewFileDetection(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "model.gguf", "original")
	writeFile(t, dir, "extra.gguf", "unexpected-file")

	hash := sha256Hex("original")
	profile := &IntegrityProfile{ModelDir: dir}
	baseline := &Baseline{TensorHashes: map[string]string{"model.gguf": hash}}
	runner := NewProbeRunner(profile, baseline)

	result := runner.runTensorHash(ProbeConfig{
		Name: "hash-test", Type: ProbeTensorHash, Enabled: true,
	})

	hasNewFile := false
	for _, f := range result.Findings {
		if strings.Contains(f.Description, "new file not in baseline") {
			hasNewFile = true
		}
	}
	if !hasNewFile {
		t.Error("expected new file finding for extra.gguf")
	}
}

func TestTensorHash_MissingFile(t *testing.T) {
	dir := t.TempDir()
	// baseline expects a file that doesn't exist

	profile := &IntegrityProfile{ModelDir: dir}
	baseline := &Baseline{TensorHashes: map[string]string{"missing.gguf": "abc123"}}
	runner := NewProbeRunner(profile, baseline)

	result := runner.runTensorHash(ProbeConfig{
		Name: "hash-test", Type: ProbeTensorHash, Enabled: true,
	})

	if result.Status == StatusPass {
		t.Error("should not pass when baseline file is missing")
	}

	hasMissing := false
	for _, f := range result.Findings {
		if strings.Contains(f.Description, "baseline file missing") {
			hasMissing = true
		}
	}
	if !hasMissing {
		t.Error("expected missing file finding")
	}
}

func TestSentinelInference_Skip(t *testing.T) {
	profile := &IntegrityProfile{}
	runner := NewProbeRunner(profile, nil)

	result := runner.runSentinelInference(ProbeConfig{
		Name: "sentinel-test", Type: ProbeSentinelInfer, Enabled: true,
	})

	if result.Status != StatusSkip {
		t.Errorf("expected skip without endpoint, got %s", result.Status)
	}
}

func TestReferenceDrift_Skip(t *testing.T) {
	profile := &IntegrityProfile{}
	runner := NewProbeRunner(profile, nil)

	result := runner.runReferenceDrift(ProbeConfig{
		Name: "drift-test", Type: ProbeReferenceDrift, Enabled: true,
	})

	if result.Status != StatusSkip {
		t.Errorf("expected skip without baseline, got %s", result.Status)
	}
}

func TestECCStatus_UnknownWithoutTrustedTool(t *testing.T) {
	profile := &IntegrityProfile{}
	runner := NewProbeRunner(profile, nil)

	result := runner.runECCStatus(ProbeConfig{
		Name:     "ecc-test",
		Type:     ProbeECCStatus,
		Enabled:  true,
		Settings: map[string]string{"nvidia_smi_path": "/nonexistent/nvidia-smi"},
	})

	if result.Status != StatusUnknown {
		t.Errorf("expected unknown without trusted nvidia-smi, got %s", result.Status)
	}
}

func TestParseECC_Healthy(t *testing.T) {
	result := parseECCOutput(ProbeResult{}, "0, 0, Enabled")
	if result.Status != StatusPass {
		t.Errorf("expected pass for healthy ECC, got %s", result.Status)
	}
}

func TestParseECC_UncorrectedErrors(t *testing.T) {
	result := parseECCOutput(ProbeResult{}, "5, 3, Enabled")
	if result.Status != StatusFail {
		t.Errorf("expected fail for uncorrected errors, got %s", result.Status)
	}
	if result.Score != 1.0 {
		t.Errorf("expected score 1.0, got %f", result.Score)
	}
}

func TestParseECC_HighCorrectedErrors(t *testing.T) {
	result := parseECCOutput(ProbeResult{}, "150, 0, Enabled")
	if result.Status != StatusDrift {
		t.Errorf("expected drift for high corrected errors, got %s", result.Status)
	}
}

func TestParseECC_Disabled(t *testing.T) {
	result := parseECCOutput(ProbeResult{}, "0, 0, Disabled")
	if result.Status != StatusDrift {
		t.Errorf("expected drift for disabled ECC, got %s", result.Status)
	}
}

func TestParseECC_NotSupported(t *testing.T) {
	result := parseECCOutput(ProbeResult{}, "[Not Supported]")
	if result.Status != StatusUnknown {
		t.Errorf("expected unknown for unsupported ECC, got %s", result.Status)
	}
}

func TestParseECC_MalformedAndNAAreUnknown(t *testing.T) {
	for _, output := range []string{"N/A, 0, Enabled", "0, nope, Enabled", "0, 0, Mystery", "0,0", "18446744073709551616,0,Enabled"} {
		if result := parseECCOutput(ProbeResult{}, output); result.Status != StatusUnknown {
			t.Errorf("expected unknown for %q, got %s", output, result.Status)
		}
	}
}

func TestParseECC_AggregatesWorstAcrossAllGPUs(t *testing.T) {
	result := parseECCOutput(ProbeResult{}, "0, 0, Enabled\n150, 0, Enabled\n0, 2, Enabled")
	if result.Status != StatusFail || len(result.Findings) != 3 {
		t.Fatalf("expected all GPUs to be accounted with fail as worst: %#v", result)
	}
}

func TestParseECC_AccountsForMalformedWidthBeforeLaterFailure(t *testing.T) {
	result := parseECCOutput(ProbeResult{}, "0,0\n0,2,Enabled")
	if result.Status != StatusFail || len(result.Findings) != 2 {
		t.Fatalf("expected malformed and failing GPUs to be accounted: %#v", result)
	}
}

func TestProbeCycleDeadlineFailsClosed(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	runner := NewProbeRunner(&IntegrityProfile{InferenceURL: "http://127.0.0.1:1"}, &Baseline{
		SentinelRefs: []SentinelRef{{Name: "deadline", Input: "x", Expected: "y"}},
	})
	result := runner.runSentinelInferenceContext(ctx, ProbeConfig{Name: "sentinel", Type: ProbeSentinelInfer, Enabled: true})
	if result.Status != StatusError || result.Score != 1 {
		t.Fatalf("expired cycle must fail closed: %#v", result)
	}
}

func TestValidateTrustedExecutable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nvidia-smi")
	if err := os.WriteFile(path, []byte("fixture"), 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := validateTrustedExecutable(path); err != nil {
		t.Fatalf("trusted executable rejected: %v", err)
	}
	if err := os.Chmod(path, 0775); err != nil {
		t.Fatal(err)
	}
	if _, err := validateTrustedExecutable(path); err == nil {
		t.Fatal("group-writable executable must be rejected")
	}
}

// ---------- similarity tests ----------

func TestComputeSimilarity_Identical(t *testing.T) {
	sim := computeSimilarity("hello world", "hello world")
	if sim != 1.0 {
		t.Errorf("expected 1.0 for identical strings, got %f", sim)
	}
}

func TestComputeSimilarity_Partial(t *testing.T) {
	sim := computeSimilarity("hello world", "hello there")
	if sim <= 0 || sim >= 1.0 {
		t.Errorf("expected partial similarity, got %f", sim)
	}
}

func TestComputeSimilarity_Disjoint(t *testing.T) {
	sim := computeSimilarity("hello world", "foo bar")
	if sim != 0.0 {
		t.Errorf("expected 0.0 for disjoint strings, got %f", sim)
	}
}

func TestComputeSimilarity_Empty(t *testing.T) {
	sim := computeSimilarity("", "hello")
	if sim != 0.0 {
		t.Errorf("expected 0.0 for empty string, got %f", sim)
	}
}

// ---------- scoring tests ----------

func TestScoring_AllPass(t *testing.T) {
	scorer := NewScoringEngine(nil, 10)
	results := []ProbeResult{
		{Probe: "a", Type: ProbeTensorHash, Status: StatusPass, Score: 0.0},
		{Probe: "b", Type: ProbeECCStatus, Status: StatusPass, Score: 0.0},
	}

	entry := scorer.Score(results)
	if entry.Verdict != VerdictHealthy {
		t.Errorf("expected healthy, got %s", entry.Verdict)
	}
	if entry.CompositeScore != 0.0 {
		t.Errorf("expected score 0.0, got %f", entry.CompositeScore)
	}
}

func TestScoring_AnyFail(t *testing.T) {
	scorer := NewScoringEngine(nil, 10)
	results := []ProbeResult{
		{Probe: "a", Type: ProbeTensorHash, Status: StatusPass, Score: 0.0},
		{Probe: "b", Type: ProbeECCStatus, Status: StatusFail, Score: 1.0},
	}

	entry := scorer.Score(results)
	if entry.Verdict != VerdictCritical {
		t.Errorf("expected critical with any fail, got %s", entry.Verdict)
	}
}

func TestScoring_DriftWarning(t *testing.T) {
	scorer := NewScoringEngine(nil, 10)
	results := []ProbeResult{
		{Probe: "a", Type: ProbeTensorHash, Status: StatusDrift, Score: 0.4},
		{Probe: "b", Type: ProbeECCStatus, Status: StatusPass, Score: 0.0},
	}

	entry := scorer.Score(results)
	if entry.Verdict == VerdictCritical {
		t.Error("drift alone should not be critical")
	}
}

func TestScoring_SkippedProducesUnknown(t *testing.T) {
	scorer := NewScoringEngine(nil, 10)
	results := []ProbeResult{
		{Probe: "a", Type: ProbeTensorHash, Status: StatusSkip, Score: 0.0},
		{Probe: "b", Type: ProbeECCStatus, Status: StatusPass, Score: 0.0},
	}

	entry := scorer.Score(results)
	if entry.Verdict != VerdictUnknown {
		t.Errorf("missing evidence should produce unknown, got %s", entry.Verdict)
	}
	if _, ok := entry.ProbeScores["a"]; ok {
		t.Error("skipped probe should not appear in scores")
	}
}

func TestScoring_History(t *testing.T) {
	scorer := NewScoringEngine(nil, 5)

	for i := 0; i < 7; i++ {
		scorer.Score([]ProbeResult{
			{Probe: "a", Type: ProbeTensorHash, Status: StatusPass, Score: float64(i) * 0.1},
		})
	}

	hist := scorer.History()
	if len(hist) != 5 {
		t.Errorf("expected max 5 history entries, got %d", len(hist))
	}
}

func TestScoring_UpdateConfigAndDefensiveCopies(t *testing.T) {
	scorer := NewScoringEngine(map[ProbeType]float64{ProbeTensorHash: 1, ProbeECCStatus: 1}, 5)
	scorer.Score([]ProbeResult{
		{Probe: "hash", Type: ProbeTensorHash, Status: StatusDrift, Score: 1},
		{Probe: "ecc", Type: ProbeECCStatus, Status: StatusPass, Score: 0},
	})
	scorer.UpdateConfig(map[ProbeType]float64{ProbeTensorHash: 9, ProbeECCStatus: 1}, 1)
	entry := scorer.Score([]ProbeResult{
		{Probe: "hash", Type: ProbeTensorHash, Status: StatusDrift, Score: 1},
		{Probe: "ecc", Type: ProbeECCStatus, Status: StatusPass, Score: 0},
	})
	if entry.CompositeScore < 0.89 || len(scorer.History()) != 1 {
		t.Fatalf("reloaded scoring config was not applied: %#v", entry)
	}
	latest := scorer.Latest()
	latest.ProbeScores["hash"] = 0
	if scorer.Latest().ProbeScores["hash"] == 0 {
		t.Fatal("latest score must be a defensive copy")
	}
}

func TestScoring_Trend(t *testing.T) {
	scorer := NewScoringEngine(nil, 100)

	// Add improving scores
	for _, s := range []float64{0.8, 0.6, 0.4, 0.2} {
		scorer.Score([]ProbeResult{
			{Probe: "a", Type: ProbeTensorHash, Status: StatusDrift, Score: s},
		})
	}

	trend := scorer.Trend(4)
	if trend >= 0 {
		t.Errorf("expected negative trend for improving scores, got %f", trend)
	}
}

func TestScoring_TrendInsufficient(t *testing.T) {
	scorer := NewScoringEngine(nil, 10)
	trend := scorer.Trend(5)
	if trend != 0.0 {
		t.Errorf("expected 0.0 trend with no history, got %f", trend)
	}
}

// ---------- action tests ----------

func TestShouldTrigger(t *testing.T) {
	cases := []struct {
		trigger  Verdict
		current  Verdict
		expected bool
	}{
		{VerdictWarning, VerdictCritical, true},
		{VerdictWarning, VerdictWarning, true},
		{VerdictWarning, VerdictHealthy, false},
		{VerdictCritical, VerdictWarning, false},
		{VerdictHealthy, VerdictHealthy, true},
	}

	for _, tc := range cases {
		got := shouldTrigger(tc.trigger, tc.current)
		if got != tc.expected {
			t.Errorf("shouldTrigger(%s, %s) = %v, want %v",
				tc.trigger, tc.current, got, tc.expected)
		}
	}
}

func TestActionExecutor_NoTrigger(t *testing.T) {
	executor := NewActionExecutor([]ActionConfig{
		{Name: "alert-critical", Type: ActionAlert, Trigger: VerdictCritical},
	}, "", "")

	entry := ScoreEntry{Verdict: VerdictHealthy}
	results := executor.Evaluate(entry)
	if len(results) != 0 {
		t.Errorf("expected no actions for healthy verdict, got %d", len(results))
	}
}

func TestActionExecutor_AlertTriggered(t *testing.T) {
	executor := NewActionExecutor([]ActionConfig{
		{Name: "alert-warn", Type: ActionAlert, Trigger: VerdictWarning},
	}, "", "")

	entry := ScoreEntry{Verdict: VerdictCritical, CompositeScore: 0.9}
	results := executor.Evaluate(entry)
	if len(results) != 1 {
		t.Fatalf("expected 1 action, got %d", len(results))
	}
	if !results[0].Triggered {
		t.Error("expected action to be triggered")
	}
	if results[0].Type != ActionAlert {
		t.Errorf("expected alert action, got %s", results[0].Type)
	}
}

func TestActionExecutor_QuarantineMovesFiles(t *testing.T) {
	t.Setenv("GPU_WATCH_ALLOW_QUARANTINE", "true")
	modelDir := t.TempDir()
	qDir := t.TempDir()
	writeFile(t, modelDir, "model.gguf", "model-data")
	writeFile(t, modelDir, "readme.txt", "not a model")

	executor := NewActionExecutor([]ActionConfig{
		{Name: "quarantine", Type: ActionQuarantine, Trigger: VerdictCritical, TargetDir: qDir},
	}, modelDir, "")

	entry := ScoreEntry{Verdict: VerdictCritical}
	results := executor.Evaluate(entry)

	if len(results) != 1 || !results[0].Success {
		t.Fatal("quarantine action should succeed")
	}

	// model.gguf should be moved
	if _, err := os.Stat(filepath.Join(qDir, "model.gguf")); err != nil {
		t.Error("model.gguf should be in quarantine dir")
	}
	// readme.txt should remain (not a model file)
	if _, err := os.Stat(filepath.Join(modelDir, "readme.txt")); err != nil {
		t.Error("readme.txt should remain in model dir")
	}
}

func TestActionExecutor_QuarantineNestedFiles(t *testing.T) {
	t.Setenv("GPU_WATCH_ALLOW_QUARANTINE", "true")
	modelDir := t.TempDir()
	qDir := t.TempDir()
	nested := filepath.Join(modelDir, "versions", "v1")
	if err := os.MkdirAll(nested, 0700); err != nil {
		t.Fatal(err)
	}
	writeFile(t, nested, "model.gguf", "model-data")
	executor := NewActionExecutor([]ActionConfig{{Name: "quarantine", Type: ActionQuarantine, Trigger: VerdictCritical, TargetDir: qDir}}, modelDir, "")
	results := executor.Evaluate(ScoreEntry{Verdict: VerdictCritical})
	if len(results) != 1 || !results[0].Success {
		t.Fatalf("nested quarantine failed: %#v", results)
	}
	if _, err := os.Stat(filepath.Join(qDir, "versions", "v1", "model.gguf")); err != nil {
		t.Fatal("nested model was not quarantined")
	}
}

func TestActionExecutor_ReloadNoURL(t *testing.T) {
	executor := NewActionExecutor([]ActionConfig{
		{Name: "reload", Type: ActionReload, Trigger: VerdictWarning},
	}, "", "")

	entry := ScoreEntry{Verdict: VerdictWarning}
	results := executor.Evaluate(entry)
	if len(results) != 1 {
		t.Fatal("expected 1 action result")
	}
	if results[0].Success {
		t.Error("reload without URL should not succeed")
	}
}

// ---------- RunAll integration ----------

func TestRunAll_Integration(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "test.gguf", "test-model-data")
	hash := sha256Hex("test-model-data")

	profile := &IntegrityProfile{
		ModelDir: dir,
		Probes: []ProbeConfig{
			{Name: "hash-check", Type: ProbeTensorHash, Enabled: true},
			{Name: "ecc-check", Type: ProbeECCStatus, Enabled: true,
				Settings: map[string]string{"nvidia_smi_path": "/nonexistent"}},
			{Name: "disabled-probe", Type: ProbeSentinelInfer, Enabled: false},
		},
	}
	baseline := &Baseline{TensorHashes: map[string]string{"test.gguf": hash}}
	runner := NewProbeRunner(profile, baseline)
	results := runner.RunAll()

	if len(results) != 2 {
		t.Errorf("expected 2 results (disabled excluded), got %d", len(results))
	}

	for _, r := range results {
		if r.Probe == "hash-check" && r.Status != StatusPass {
			t.Errorf("hash-check expected pass, got %s", r.Status)
		}
		if r.Probe == "ecc-check" && r.Status != StatusUnknown {
			t.Errorf("ecc-check expected unknown, got %s", r.Status)
		}
	}
}

func TestFullPipeline(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "model.gguf", "good-model-data")
	hash := sha256Hex("good-model-data")

	profile := &IntegrityProfile{
		ModelDir: dir,
		Probes: []ProbeConfig{
			{Name: "tensor", Type: ProbeTensorHash, Enabled: true},
		},
		Actions: []ActionConfig{
			{Name: "alert-critical", Type: ActionAlert, Trigger: VerdictCritical},
		},
	}
	baseline := &Baseline{TensorHashes: map[string]string{"model.gguf": hash}}

	runner := NewProbeRunner(profile, baseline)
	scorer := NewScoringEngine(nil, 10)
	executor := NewActionExecutor(profile.Actions, dir, "")

	// First run: should pass
	results := runner.RunAll()
	entry := scorer.Score(results)
	actions := executor.Evaluate(entry)

	if entry.Verdict != VerdictHealthy {
		t.Errorf("expected healthy on first run, got %s", entry.Verdict)
	}
	if len(actions) != 0 {
		t.Errorf("expected no actions on healthy, got %d", len(actions))
	}

	// Corrupt the model
	writeFile(t, dir, "model.gguf", "corrupted-model-data")

	results = runner.RunAll()
	entry = scorer.Score(results)
	actions = executor.Evaluate(entry)

	if entry.Verdict == VerdictHealthy {
		t.Error("should not be healthy after corruption")
	}
}

// ---------- HTTP handler tests ----------

func TestHTTP_Health(t *testing.T) {
	mux := buildTestMux(t)
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var body map[string]string
	json.NewDecoder(w.Body).Decode(&body)
	if body["status"] != "ok" {
		t.Errorf("expected ok, got %s", body["status"])
	}
}

func TestHTTP_Check(t *testing.T) {
	mux := buildTestMux(t)
	req := authorizedGPURequest("POST", "/v1/check")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var body map[string]interface{}
	json.NewDecoder(w.Body).Decode(&body)
	if _, ok := body["probes"]; !ok {
		t.Error("expected probes in response")
	}
	if _, ok := body["score"]; !ok {
		t.Error("expected score in response")
	}
}

func TestHTTP_CheckMethodNotAllowed(t *testing.T) {
	mux := buildTestMux(t)
	req := authorizedGPURequest("GET", "/v1/check")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != 405 {
		t.Errorf("expected 405 for GET on /v1/check, got %d", w.Code)
	}
}

func TestHTTP_Status(t *testing.T) {
	mux := buildTestMux(t)
	req := authorizedGPURequest("GET", "/v1/status")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestHTTP_History(t *testing.T) {
	mux := buildTestMux(t)
	req := authorizedGPURequest("GET", "/v1/history")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestHTTP_Metrics(t *testing.T) {
	mux := buildTestMux(t)
	req := authorizedGPURequest("GET", "/v1/metrics")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var body map[string]interface{}
	json.NewDecoder(w.Body).Decode(&body)
	if _, ok := body["checks_total"]; !ok {
		t.Error("expected checks_total in metrics")
	}
}

func TestHTTP_ReloadRequiresToken(t *testing.T) {
	mux := buildTestMuxWithToken(t, "test-token-123")

	// Without token
	req := httptest.NewRequest("POST", "/v1/reload", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != 401 {
		t.Errorf("expected 401 without token, got %d", w.Code)
	}

	// With wrong token
	req = httptest.NewRequest("POST", "/v1/reload", nil)
	req.Header.Set("Authorization", "Bearer wrong-token")
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != 401 {
		t.Errorf("expected 401 with wrong token, got %d", w.Code)
	}

	// With correct token
	req = httptest.NewRequest("POST", "/v1/reload", nil)
	req.Header.Set("Authorization", "Bearer test-token-123")
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Errorf("expected 200 with correct token, got %d", w.Code)
	}
}

func TestHTTP_BaselineRequiresToken(t *testing.T) {
	mux := buildTestMuxWithToken(t, "secret")

	req := httptest.NewRequest("POST", "/v1/baseline", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != 401 {
		t.Errorf("expected 401 without token, got %d", w.Code)
	}
}

// ---------- token auth ----------

func TestCheckToken_Empty(t *testing.T) {
	req := httptest.NewRequest("GET", "/", nil)
	if checkToken(req, "") {
		t.Error("empty expected token must fail closed")
	}
}

func TestCheckToken_Valid(t *testing.T) {
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "Bearer my-secret")
	if !checkToken(req, "my-secret") {
		t.Error("valid token should be accepted")
	}
}

func TestCheckToken_Invalid(t *testing.T) {
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	if checkToken(req, "my-secret") {
		t.Error("invalid token should be rejected")
	}
}

func TestReadCredentialRequiresOwnerOnlyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "credential")
	credential := strings.Repeat("x", 32)
	if err := os.WriteFile(path, []byte(credential), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := readCredentialFile(path); err == nil {
		t.Fatal("group/world-readable credential must be rejected")
	}
	if err := os.Chmod(path, 0600); err != nil {
		t.Fatal(err)
	}
	if value, err := readCredentialFile(path); err != nil || value != credential {
		t.Fatalf("owner-only credential rejected: value=%q err=%v", value, err)
	}
}

// ---------- helpers ----------

func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644)
	if err != nil {
		t.Fatal(err)
	}
}

func sha256Hex(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func authorizedGPURequest(method, target string) *http.Request {
	req := httptest.NewRequest(method, target, nil)
	req.Header.Set("Authorization", "Bearer test-token")
	return req
}

func buildTestMux(t *testing.T) http.Handler {
	t.Helper()
	return buildTestMuxWithToken(t, "test-token")
}

func buildTestMuxWithToken(t *testing.T, token string) http.Handler {
	t.Helper()

	dir := t.TempDir()
	profilePath := filepath.Join(dir, "profile.yaml")
	os.WriteFile(profilePath, []byte(`
version: 1
model_dir: ""
probes:
  - name: test-hash
    type: tensor_hash
    enabled: true
scoring:
  max_history: 10
daemon:
  bind_addr: "127.0.0.1:8505"
`), 0o644)

	// Set env for loadProfile
	os.Setenv("INTEGRITY_PROFILE", profilePath)
	defer os.Unsetenv("INTEGRITY_PROFILE")

	profile := &IntegrityProfile{
		Probes: []ProbeConfig{
			{Name: "test-hash", Type: ProbeTensorHash, Enabled: true},
		},
		Scoring: ScoringConfig{MaxHistory: 10},
	}

	runner := NewProbeRunner(profile, nil)
	scorer := NewScoringEngine(nil, 10)
	executor := NewActionExecutor(nil, "", "")

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("/v1/check", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		results := runner.RunAll()
		entry := scorer.Score(results)
		actions := executor.Evaluate(entry)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"probes": results, "score": entry, "actions": actions,
		})
	})

	mux.HandleFunc("/v1/status", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"latest": scorer.Latest(), "trend": scorer.Trend(10),
		})
	})

	mux.HandleFunc("/v1/history", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(scorer.History())
	})

	mux.HandleFunc("/v1/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]int64{
			"checks_total": metricChecks.Load(),
		})
	})

	mux.HandleFunc("/v1/reload", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !checkToken(r, token) {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "reloaded"})
	})

	mux.HandleFunc("/v1/baseline", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !checkToken(r, token) {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "baseline captured"})
	})

	return authenticatedGPUHandler(mux, token)
}

func TestTensorHashRejectsSymlinkEvidence(t *testing.T) {
	dir := t.TempDir()
	realPath := filepath.Join(dir, "real.gguf")
	if err := os.WriteFile(realPath, []byte("model"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(realPath, filepath.Join(dir, "linked.gguf")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	runner := NewProbeRunner(&IntegrityProfile{ModelDir: dir}, &Baseline{TensorHashes: map[string]string{"real.gguf": sha256Hex("model")}})
	result := runner.runTensorHash(ProbeConfig{Name: "hash", Type: ProbeTensorHash, Enabled: true})
	if result.Status != StatusError {
		t.Fatalf("unsafe model evidence must produce error, got %s", result.Status)
	}
}

func TestDriverFingerprintBaseline(t *testing.T) {
	dir := t.TempDir()
	versionPath := filepath.Join(dir, "version")
	if err := os.WriteFile(versionPath, []byte("550.90.07\n"), 0600); err != nil {
		t.Fatal(err)
	}
	baseline := &Baseline{DriverFingerprint: &DriverBaseline{DriverVersion: "550.90.07", KernelModule: "nvidia"}}
	runner := NewProbeRunner(&IntegrityProfile{}, baseline)
	probe := ProbeConfig{Name: "driver", Type: ProbeDriverFingerprint, Enabled: true, Settings: map[string]string{
		"driver_version_path": versionPath,
		"kernel_module":       "nvidia",
	}}
	if result := runner.runDriverFingerprint(probe); result.Status != StatusPass {
		t.Fatalf("expected matching driver baseline, got %s: %#v", result.Status, result.Findings)
	}
	baseline.DriverFingerprint.DriverVersion = "old-version"
	if result := runner.runDriverFingerprint(probe); result.Status != StatusFail {
		t.Fatalf("expected driver mismatch failure, got %s", result.Status)
	}
}

func TestDeviceAllowlistMissingBaselineFailsClosed(t *testing.T) {
	runner := NewProbeRunner(&IntegrityProfile{}, nil)
	result := runner.runDeviceAllowlist(ProbeConfig{Name: "devices", Type: ProbeDeviceAllowlist, Enabled: true, Settings: map[string]string{"device_dir": t.TempDir()}})
	if result.Status != StatusError {
		t.Fatalf("missing device baseline must error, got %s", result.Status)
	}
}

func TestUnknownVerdictNeverTriggersDestructiveAction(t *testing.T) {
	t.Setenv("GPU_WATCH_ALLOW_SHUTDOWN", "true")
	executor := NewActionExecutor([]ActionConfig{{Name: "contain", Type: ActionFailClosed, Trigger: VerdictCritical}}, "", "")
	results := executor.Evaluate(ScoreEntry{Verdict: VerdictUnknown})
	if len(results) != 0 {
		t.Fatal("unknown evidence must never trigger destructive containment")
	}
}

func TestUnknownVerdictOnlyAllowsAlerts(t *testing.T) {
	executor := NewActionExecutor([]ActionConfig{
		{Name: "reload", Type: ActionReload, Trigger: VerdictWarning},
		{Name: "alert", Type: ActionAlert, Trigger: VerdictWarning},
	}, "", "")
	results := executor.Evaluate(ScoreEntry{Verdict: VerdictUnknown})
	if len(results) != 1 || results[0].Type != ActionAlert {
		t.Fatalf("unknown evidence must only notify: %#v", results)
	}
}

func TestDestructiveActionsRequireSeparateOptIns(t *testing.T) {
	executor := NewActionExecutor([]ActionConfig{
		{Name: "quarantine", Type: ActionQuarantine, Trigger: VerdictCritical},
		{Name: "shutdown", Type: ActionFailClosed, Trigger: VerdictCritical},
	}, t.TempDir(), "")
	results := executor.Evaluate(ScoreEntry{Verdict: VerdictCritical})
	if len(results) != 2 || results[0].Triggered || results[1].Triggered {
		t.Fatalf("destructive actions must be inert by default: %#v", results)
	}
}

func TestDestructiveActionCooldownIsIdempotent(t *testing.T) {
	t.Setenv("GPU_WATCH_ALLOW_QUARANTINE", "true")
	modelDir := t.TempDir()
	quarantineDir := t.TempDir()
	writeFile(t, modelDir, "model.gguf", "model")
	executor := NewActionExecutor([]ActionConfig{{
		Name: "quarantine", Type: ActionQuarantine, Trigger: VerdictCritical,
		TargetDir: quarantineDir, Cooldown: "1h",
	}}, modelDir, "")
	first := executor.Evaluate(ScoreEntry{Verdict: VerdictCritical})
	second := executor.Evaluate(ScoreEntry{Verdict: VerdictCritical})
	if len(first) != 1 || !first[0].Success || len(second) != 1 || second[0].Triggered {
		t.Fatalf("destructive action was not idempotently suppressed: first=%#v second=%#v", first, second)
	}
}

func TestCommandActionDoesNotInvokeShell(t *testing.T) {
	t.Setenv("GPU_WATCH_ALLOW_ACTION_COMMANDS", "true")
	marker := filepath.Join(t.TempDir(), "should-not-exist")
	result := executeCommand(ActionConfig{Name: "safe", Type: ActionAlert, Command: "/bin/echo safe ; /usr/bin/touch " + marker})
	if !result.Success {
		t.Fatalf("direct command should execute: %s", result.Message)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("command arguments were interpreted by a shell")
	}
}

func TestServiceURLRequiresTLSOffLoopback(t *testing.T) {
	if _, err := parseServiceURL("http://example.com"); err == nil {
		t.Fatal("remote plaintext service URL must be rejected")
	}
	if _, err := parseServiceURL("http://127.0.0.1:8505"); err != nil {
		t.Fatalf("loopback service URL rejected: %v", err)
	}
	if _, err := parseServiceURL("https://example.com"); err != nil {
		t.Fatalf("TLS service URL rejected: %v", err)
	}
}

func TestIncidentReporterAuthenticatesAndDeduplicates(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer recorder-secret" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		calls.Add(1)
		w.WriteHeader(http.StatusCreated)
	}))
	defer server.Close()
	reporter := newIncidentReporter()
	entry := ScoreEntry{Timestamp: time.Now(), Verdict: VerdictCritical, CompositeScore: 1, ProbeStatuses: map[string]ProbeStatus{"hash": StatusFail}}
	results := []ProbeResult{{Probe: "hash", Type: ProbeTensorHash, Status: StatusFail}}
	if err := reporter.Report(server.URL, "recorder-secret", entry, results); err != nil {
		t.Fatal(err)
	}
	if err := reporter.Report(server.URL, "recorder-secret", entry, results); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("expected one deduplicated report, got %d", calls.Load())
	}
}

func TestAuthenticatedGPUHandlerProtectsReadEndpoints(t *testing.T) {
	handler := authenticatedGPUHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), "secret")
	request := httptest.NewRequest(http.MethodGet, "/v1/status", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("expected authenticated read endpoint, got %d", response.Code)
	}
}

func TestVerifyGPUAuditRejectsUnknownAndTrailingJSON(t *testing.T) {
	dir := t.TempDir()
	entry := GPUAuditEntry{Timestamp: time.Now().UTC().Format(time.RFC3339Nano), Event: "test"}
	entry.Hash = computeGPUAuditHash(entry)
	data, err := json.Marshal(entry)
	if err != nil {
		t.Fatal(err)
	}
	for name, line := range map[string][]byte{
		"unknown":  append(append([]byte{}, data[:len(data)-1]...), []byte(`,"unexpected":true}`)...),
		"trailing": append(append([]byte{}, data...), []byte(` {}`)...),
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, name+".jsonl")
			if err := os.WriteFile(path, append(line, '\n'), 0600); err != nil {
				t.Fatal(err)
			}
			if err := verifyGPUAudit(path); err == nil {
				t.Fatal("non-strict audit JSON must be rejected")
			}
		})
	}
}

func TestGPUAuditWriteFailurePoisonsSensitiveEndpoints(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	auditFile = f
	auditRequired.Store(true)
	auditHealthy.Store(true)
	auditLastHash = ""
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		auditFile = nil
		auditRequired.Store(false)
		auditHealthy.Store(false)
		auditLastHash = ""
	})
	if err := auditLog(GPUAuditEntry{Event: "test"}); err == nil {
		t.Fatal("closed audit file must fail")
	}
	handler := authenticatedGPUHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unhealthy audit must block protected endpoints")
	}), strings.Repeat("t", 32))
	req := httptest.NewRequest(http.MethodPost, "/v1/check", nil)
	req.Header.Set("Authorization", "Bearer "+strings.Repeat("t", 32))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

func TestProfileAndBaselineRejectGroupWritableFiles(t *testing.T) {
	dir := t.TempDir()
	profilePath := filepath.Join(dir, "profile.yaml")
	if err := os.WriteFile(profilePath, []byte("version: 1\n"), 0620); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(profilePath, 0620); err != nil {
		t.Fatal(err)
	}
	if _, err := loadProfileFile(profilePath); err == nil {
		t.Fatal("group-writable profile must be rejected")
	}
	baselinePath := filepath.Join(dir, "baseline.yaml")
	if err := os.WriteFile(baselinePath, []byte("captured_at: 2026-08-02T00:00:00Z\n"), 0620); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(baselinePath, 0620); err != nil {
		t.Fatal(err)
	}
	if _, err := loadBaselineFile(&IntegrityProfile{BaselineFile: baselinePath}); err == nil {
		t.Fatal("group-writable baseline must be rejected")
	}
}
