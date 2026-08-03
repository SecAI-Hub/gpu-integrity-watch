package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var outboundHTTPClient = &http.Client{
	Timeout: 15 * time.Second,
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 3 || len(via) == 0 || req.URL.Scheme != via[0].URL.Scheme || req.URL.Host != via[0].URL.Host {
			return fmt.Errorf("cross-origin or excessive redirect refused")
		}
		return nil
	},
}

func validatedActionURL(raw, suffix string) (string, error) {
	parsed, err := parseServiceURL(raw)
	if err != nil {
		return "", err
	}
	if suffix != "" {
		parsed.Path = strings.TrimRight(parsed.Path, "/") + suffix
		parsed.RawQuery = ""
	}
	return parsed.String(), nil
}

func parseServiceURL(raw string) (*url.URL, error) {
	parsed, err := url.Parse(raw)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, fmt.Errorf("invalid service URL")
	}
	if parsed.Scheme == "http" {
		host := parsed.Hostname()
		ip := net.ParseIP(host)
		if host != "localhost" && (ip == nil || !ip.IsLoopback()) {
			return nil, fmt.Errorf("unencrypted service URLs are limited to loopback")
		}
	}
	return parsed, nil
}

func drainActionResponse(body io.Reader) {
	_, _ = io.Copy(io.Discard, io.LimitReader(body, 64<<10))
}

// ActionType identifies a response action.
type ActionType string

const (
	ActionReload     ActionType = "reload"
	ActionQuarantine ActionType = "quarantine"
	ActionAlert      ActionType = "alert"
	ActionFailClosed ActionType = "fail_closed"
)

// ActionConfig defines when and how to trigger a response action.
type ActionConfig struct {
	Name      string     `yaml:"name" json:"name"`
	Type      ActionType `yaml:"type" json:"type"`
	Trigger   Verdict    `yaml:"trigger" json:"trigger"` // verdict level that triggers this action
	Webhook   string     `yaml:"webhook,omitempty" json:"webhook,omitempty"`
	Command   string     `yaml:"command,omitempty" json:"command,omitempty"`
	TargetDir string     `yaml:"target_dir,omitempty" json:"target_dir,omitempty"` // for quarantine
	Cooldown  string     `yaml:"cooldown,omitempty" json:"cooldown,omitempty"`
}

// ActionResult records the outcome of an executed action.
type ActionResult struct {
	Action    string     `json:"action"`
	Type      ActionType `json:"type"`
	Triggered bool       `json:"triggered"`
	Success   bool       `json:"success"`
	Message   string     `json:"message"`
	Timestamp time.Time  `json:"timestamp"`
}

// ActionExecutor evaluates scoring results and triggers configured actions.
type ActionExecutor struct {
	mu       sync.Mutex
	actions  []ActionConfig
	modelDir string
	inferURL string
	lastRun  map[string]time.Time
}

// NewActionExecutor creates an executor with the given action configs.
func NewActionExecutor(actions []ActionConfig, modelDir, inferURL string) *ActionExecutor {
	return &ActionExecutor{
		actions:  actions,
		modelDir: modelDir,
		inferURL: inferURL,
		lastRun:  make(map[string]time.Time),
	}
}

func (e *ActionExecutor) Update(actions []ActionConfig, modelDir, inferURL string) {
	e.mu.Lock()
	e.actions = actions
	e.modelDir = modelDir
	e.inferURL = inferURL
	e.mu.Unlock()
}

// Evaluate checks the score entry against action triggers and executes matching actions.
func (e *ActionExecutor) Evaluate(entry ScoreEntry) []ActionResult {
	e.mu.Lock()
	defer e.mu.Unlock()
	var results []ActionResult

	for _, ac := range e.actions {
		if !shouldTrigger(ac.Trigger, entry.Verdict) {
			continue
		}
		// Unknown evidence can notify an operator, but must not mutate inference
		// state or model storage.
		if entry.Verdict == VerdictUnknown && ac.Type != ActionAlert {
			continue
		}
		if isDestructiveAction(ac.Type) {
			if !destructiveActionEnabled(ac.Type) {
				results = append(results, ActionResult{
					Action: ac.Name, Type: ac.Type, Triggered: false, Success: false,
					Message: "destructive action requires its explicit opt-in", Timestamp: time.Now(),
				})
				continue
			}
			cooldown := 15 * time.Minute
			if ac.Cooldown != "" {
				cooldown, _ = time.ParseDuration(ac.Cooldown)
			}
			key := string(ac.Type) + ":" + ac.Name
			if last := e.lastRun[key]; !last.IsZero() && time.Since(last) < cooldown {
				results = append(results, ActionResult{
					Action: ac.Name, Type: ac.Type, Triggered: false, Success: false,
					Message: "destructive action suppressed by cooldown", Timestamp: time.Now(),
				})
				continue
			}
			e.lastRun[key] = time.Now()
		}

		var ar ActionResult
		switch ac.Type {
		case ActionReload:
			ar = e.executeReload(ac)
		case ActionQuarantine:
			ar = e.executeQuarantine(ac)
		case ActionAlert:
			ar = e.executeAlert(ac, entry)
		case ActionFailClosed:
			ar = e.executeFailClosed(ac)
		default:
			ar = ActionResult{
				Action:    ac.Name,
				Type:      ac.Type,
				Triggered: true,
				Success:   false,
				Message:   fmt.Sprintf("unknown action type: %s", ac.Type),
			}
		}

		ar.Timestamp = time.Now()
		results = append(results, ar)
	}

	return results
}

func isDestructiveAction(actionType ActionType) bool {
	return actionType == ActionQuarantine || actionType == ActionFailClosed
}

func destructiveActionEnabled(actionType ActionType) bool {
	switch actionType {
	case ActionQuarantine:
		return os.Getenv("GPU_WATCH_ALLOW_QUARANTINE") == "true"
	case ActionFailClosed:
		return os.Getenv("GPU_WATCH_ALLOW_SHUTDOWN") == "true"
	default:
		return true
	}
}

// shouldTrigger returns true if the current verdict meets the trigger threshold.
func shouldTrigger(trigger Verdict, current Verdict) bool {
	order := map[Verdict]int{
		VerdictHealthy:  0,
		VerdictWarning:  1,
		VerdictCritical: 2,
		VerdictUnknown:  2,
	}

	return order[current] >= order[trigger]
}

// executeReload signals the inference server to reload the model.
func (e *ActionExecutor) executeReload(ac ActionConfig) ActionResult {
	ar := ActionResult{Action: ac.Name, Type: ActionReload, Triggered: true}

	target := e.inferURL
	if target == "" {
		ar.Success = false
		ar.Message = "no inference URL configured for reload"
		return ar
	}

	// Try llama.cpp-style reload endpoint
	actionURL, err := validatedActionURL(target, "/reload")
	if err != nil {
		ar.Success = false
		ar.Message = err.Error()
		return ar
	}
	resp, err := outboundHTTPClient.Post(actionURL, "application/json", strings.NewReader("{}"))
	if err != nil {
		// Fall back to command if configured
		if ac.Command != "" {
			return executeCommand(ac)
		}
		ar.Success = false
		ar.Message = fmt.Sprintf("reload request failed: %v", err)
		return ar
	}
	defer resp.Body.Close()
	drainActionResponse(resp.Body)

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		ar.Success = true
		ar.Message = "model reload triggered"
	} else {
		ar.Success = false
		ar.Message = fmt.Sprintf("reload returned status %d", resp.StatusCode)
	}

	return ar
}

// executeQuarantine moves model files to a quarantine directory.
func (e *ActionExecutor) executeQuarantine(ac ActionConfig) ActionResult {
	ar := ActionResult{Action: ac.Name, Type: ActionQuarantine, Triggered: true}
	if os.Getenv("GPU_WATCH_ALLOW_QUARANTINE") != "true" {
		ar.Message = "quarantine disabled; set GPU_WATCH_ALLOW_QUARANTINE=true"
		return ar
	}

	srcDir := e.modelDir
	if srcDir == "" {
		ar.Success = false
		ar.Message = "no model directory configured for quarantine"
		return ar
	}

	qDir := ac.TargetDir
	if qDir == "" {
		qDir = filepath.Join(filepath.Dir(srcDir), "quarantine")
	}
	srcDir, err := filepath.Abs(srcDir)
	if err != nil {
		ar.Message = "invalid model directory"
		return ar
	}
	qDir, err = filepath.Abs(qDir)
	if err != nil || qDir == srcDir || strings.HasPrefix(qDir, srcDir+string(os.PathSeparator)) {
		ar.Message = "quarantine directory must be separate from model directory"
		return ar
	}
	if info, err := os.Lstat(srcDir); err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		ar.Message = "model directory is unsafe"
		return ar
	}

	if err := os.MkdirAll(qDir, 0o700); err != nil {
		ar.Success = false
		ar.Message = fmt.Sprintf("failed to create quarantine dir: %v", err)
		return ar
	}
	if info, err := os.Lstat(qDir); err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		ar.Message = "quarantine directory is unsafe"
		return ar
	}

	moved := 0
	failed := 0
	touchedDirs := map[string]bool{srcDir: true, qDir: true}
	walkErr := filepath.WalkDir(srcDir, func(src string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			failed++
			return filepath.SkipDir
		}
		if src == srcDir {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			failed++
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.IsDir() {
			return nil
		}
		lowerName := strings.ToLower(entry.Name())
		if !strings.HasSuffix(lowerName, ".gguf") && !strings.HasSuffix(lowerName, ".bin") &&
			!strings.HasSuffix(lowerName, ".safetensors") {
			return nil
		}
		srcInfo, err := os.Lstat(src)
		if err != nil || !srcInfo.Mode().IsRegular() || srcInfo.Mode()&os.ModeSymlink != 0 {
			failed++
			return nil
		}
		rel, err := filepath.Rel(srcDir, src)
		if err != nil || rel == "." || strings.HasPrefix(rel, "..") {
			failed++
			return nil
		}
		dstDir, err := ensureSafeSubdir(qDir, filepath.Dir(rel))
		if err != nil {
			failed++
			return nil
		}
		dst := filepath.Join(dstDir, filepath.Base(rel))
		if err := os.Link(src, dst); err != nil {
			log.Printf("[action] quarantine link failed %s: %v", rel, err)
			failed++
			return nil
		}
		dstInfo, dstErr := os.Lstat(dst)
		srcAfter, srcErr := os.Lstat(src)
		if dstErr != nil || srcErr != nil || !os.SameFile(srcInfo, srcAfter) || !os.SameFile(srcAfter, dstInfo) {
			failed++
			return nil
		}
		if err := os.Remove(src); err != nil {
			log.Printf("[action] quarantine unlink failed %s: %v", rel, err)
			failed++
			return nil
		}
		moved++
		touchedDirs[filepath.Dir(src)] = true
		touchedDirs[dstDir] = true
		return nil
	})
	if walkErr != nil {
		failed++
	}

	for dirPath := range touchedDirs {
		// #nosec G304 -- every directory is derived from validated source/quarantine roots and is opened only for fsync.
		if dir, err := os.Open(dirPath); err == nil {
			_ = dir.Sync()
			_ = dir.Close()
		}
	}
	ar.Success = failed == 0 && moved > 0
	ar.Message = fmt.Sprintf("quarantined %d model files (%d failed)", moved, failed)
	return ar
}

func ensureSafeSubdir(root, relative string) (string, error) {
	current := root
	if relative == "." || relative == "" {
		return current, nil
	}
	for _, component := range strings.Split(filepath.Clean(relative), string(os.PathSeparator)) {
		if component == "" || component == "." || component == ".." {
			return "", fmt.Errorf("unsafe quarantine path")
		}
		current = filepath.Join(current, component)
		if err := os.Mkdir(current, 0o700); err != nil && !os.IsExist(err) {
			return "", err
		}
		info, err := os.Lstat(current)
		if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return "", fmt.Errorf("unsafe quarantine directory")
		}
	}
	return current, nil
}

// executeAlert sends an alert via webhook or command.
func (e *ActionExecutor) executeAlert(ac ActionConfig, entry ScoreEntry) ActionResult {
	ar := ActionResult{Action: ac.Name, Type: ActionAlert, Triggered: true}

	payload := map[string]interface{}{
		"source":    "gpu-integrity-watch",
		"verdict":   entry.Verdict,
		"score":     entry.CompositeScore,
		"probes":    entry.ProbeScores,
		"timestamp": entry.Timestamp.Format(time.RFC3339),
	}

	if ac.Webhook != "" {
		webhook, err := validatedActionURL(ac.Webhook, "")
		if err != nil {
			ar.Message = err.Error()
			return ar
		}
		body, _ := json.Marshal(payload)
		resp, err := outboundHTTPClient.Post(webhook, "application/json", strings.NewReader(string(body)))
		if err != nil {
			ar.Success = false
			ar.Message = fmt.Sprintf("webhook failed: %v", err)
			return ar
		}
		defer resp.Body.Close()
		drainActionResponse(resp.Body)

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			ar.Success = true
			ar.Message = "alert sent via webhook"
		} else {
			ar.Success = false
			ar.Message = fmt.Sprintf("webhook returned %d", resp.StatusCode)
		}
		return ar
	}

	if ac.Command != "" {
		return executeCommand(ac)
	}

	// Default: log the alert
	log.Printf("[ALERT] integrity verdict=%s score=%.2f", entry.Verdict, entry.CompositeScore)
	ar.Success = true
	ar.Message = "alert logged (no webhook or command configured)"
	return ar
}

// executeFailClosed attempts to shut down the inference server.
func (e *ActionExecutor) executeFailClosed(ac ActionConfig) ActionResult {
	ar := ActionResult{Action: ac.Name, Type: ActionFailClosed, Triggered: true}
	if os.Getenv("GPU_WATCH_ALLOW_SHUTDOWN") != "true" {
		ar.Message = "shutdown disabled; set GPU_WATCH_ALLOW_SHUTDOWN=true"
		return ar
	}

	if ac.Command != "" {
		return executeCommand(ac)
	}

	// Try to signal inference server shutdown
	if e.inferURL != "" {
		actionURL, err := validatedActionURL(e.inferURL, "/shutdown")
		if err != nil {
			ar.Message = err.Error()
			return ar
		}
		resp, err := outboundHTTPClient.Post(actionURL, "application/json", strings.NewReader("{}"))
		if err != nil {
			ar.Success = false
			ar.Message = fmt.Sprintf("fail-closed shutdown request failed: %v", err)
			return ar
		}
		defer resp.Body.Close()
		drainActionResponse(resp.Body)
		ar.Success = resp.StatusCode >= 200 && resp.StatusCode < 300
		ar.Message = fmt.Sprintf("fail-closed shutdown returned HTTP %d", resp.StatusCode)
		return ar
	}

	ar.Success = false
	ar.Message = "fail-closed: no command or inference URL configured"
	return ar
}

// executeCommand runs a shell command for an action.
func executeCommand(ac ActionConfig) ActionResult {
	ar := ActionResult{Action: ac.Name, Type: ac.Type, Triggered: true}

	if os.Getenv("GPU_WATCH_ALLOW_ACTION_COMMANDS") != "true" {
		ar.Success = false
		ar.Message = "command actions disabled; set GPU_WATCH_ALLOW_ACTION_COMMANDS=true"
		return ar
	}

	parts := strings.Fields(ac.Command)
	if len(parts) == 0 || !filepath.IsAbs(parts[0]) || filepath.Clean(parts[0]) != parts[0] {
		ar.Success = false
		ar.Message = "command actions require a canonical absolute executable"
		return ar
	}
	_, err := runBoundedCommand(parts[0], parts[1:]...)
	if err != nil {
		ar.Success = false
		ar.Message = fmt.Sprintf("command failed: %v", err)
	} else {
		ar.Success = true
		ar.Message = "command executed successfully"
	}

	return ar
}
