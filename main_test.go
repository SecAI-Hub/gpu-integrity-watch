package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
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

func TestECCStatus_Skip(t *testing.T) {
	profile := &IntegrityProfile{}
	runner := NewProbeRunner(profile, nil)

	result := runner.runECCStatus(ProbeConfig{
		Name:     "ecc-test",
		Type:     ProbeECCStatus,
		Enabled:  true,
		Settings: map[string]string{"nvidia_smi_path": "/nonexistent/nvidia-smi"},
	})

	if result.Status != StatusSkip {
		t.Errorf("expected skip without nvidia-smi, got %s", result.Status)
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
	if result.Status != StatusSkip {
		t.Errorf("expected skip for unsupported ECC, got %s", result.Status)
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

func TestScoring_SkippedIgnored(t *testing.T) {
	scorer := NewScoringEngine(nil, 10)
	results := []ProbeResult{
		{Probe: "a", Type: ProbeTensorHash, Status: StatusSkip, Score: 0.0},
		{Probe: "b", Type: ProbeECCStatus, Status: StatusPass, Score: 0.0},
	}

	entry := scorer.Score(results)
	if entry.Verdict != VerdictHealthy {
		t.Errorf("skipped probes should not affect verdict, got %s", entry.Verdict)
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
		if r.Probe == "ecc-check" && r.Status != StatusSkip {
			t.Errorf("ecc-check expected skip, got %s", r.Status)
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
	req := httptest.NewRequest("POST", "/v1/check", nil)
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
	req := httptest.NewRequest("GET", "/v1/check", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != 405 {
		t.Errorf("expected 405 for GET on /v1/check, got %d", w.Code)
	}
}

func TestHTTP_Status(t *testing.T) {
	mux := buildTestMux(t)
	req := httptest.NewRequest("GET", "/v1/status", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestHTTP_History(t *testing.T) {
	mux := buildTestMux(t)
	req := httptest.NewRequest("GET", "/v1/history", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestHTTP_Metrics(t *testing.T) {
	mux := buildTestMux(t)
	req := httptest.NewRequest("GET", "/v1/metrics", nil)
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
	if !checkToken(req, "") {
		t.Error("empty expected token should allow all requests")
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

func buildTestMux(t *testing.T) *http.ServeMux {
	t.Helper()
	return buildTestMuxWithToken(t, "")
}

func buildTestMuxWithToken(t *testing.T, token string) *http.ServeMux {
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

	return mux
}
