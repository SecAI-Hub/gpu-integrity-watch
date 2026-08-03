package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

type IntegrationConfig struct {
	IncidentRecorderURL       string `yaml:"incident_recorder_url"`
	IncidentRecorderTokenPath string `yaml:"incident_recorder_token_path"`
}

type GPUAttestState struct {
	Timestamp      time.Time              `json:"timestamp"`
	Verdict        Verdict                `json:"verdict"`
	CompositeScore float64                `json:"composite_score"`
	ProbeStatuses  map[string]ProbeStatus `json:"probe_statuses"`
	ProbeScores    map[string]float64     `json:"probe_scores"`
	Trend          float64                `json:"trend"`
}

func buildAttestState(scorer *ScoringEngine) GPUAttestState {
	state := GPUAttestState{
		Timestamp:     time.Now().UTC(),
		Verdict:       VerdictUnknown,
		ProbeStatuses: map[string]ProbeStatus{},
		ProbeScores:   map[string]float64{},
		Trend:         scorer.Trend(10),
	}
	if latest := scorer.Latest(); latest != nil {
		state.Verdict = latest.Verdict
		state.CompositeScore = latest.CompositeScore
		for key, value := range latest.ProbeStatuses {
			state.ProbeStatuses[key] = value
		}
		for key, value := range latest.ProbeScores {
			state.ProbeScores[key] = value
		}
	}
	return state
}

type incidentReporter struct {
	mu              sync.Mutex
	lastFingerprint string
	lastReport      time.Time
	cooldown        time.Duration
}

func newIncidentReporter() *incidentReporter {
	return &incidentReporter{cooldown: 5 * time.Minute}
}

func (reporter *incidentReporter) Report(baseURL, token string, entry ScoreEntry, results []ProbeResult) error {
	if baseURL == "" || (entry.Verdict != VerdictWarning && entry.Verdict != VerdictCritical && entry.Verdict != VerdictUnknown) {
		return nil
	}
	statusJSON, _ := json.Marshal(entry.ProbeStatuses)
	fingerprint := fmt.Sprintf("%s:%0.4f:%s", entry.Verdict, entry.CompositeScore, statusJSON)
	reporter.mu.Lock()
	if fingerprint == reporter.lastFingerprint && time.Since(reporter.lastReport) < reporter.cooldown {
		reporter.mu.Unlock()
		return nil
	}
	reporter.mu.Unlock()

	parsed, err := parseServiceURL(baseURL)
	if err != nil {
		return fmt.Errorf("invalid incident recorder URL: %w", err)
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/v1/event"
	parsed.RawQuery = ""
	parsed.Fragment = ""

	severity := "warning"
	if entry.Verdict == VerdictCritical || entry.Verdict == VerdictUnknown {
		severity = "critical"
	}
	failed := make([]string, 0)
	for _, result := range results {
		if result.Status != StatusPass {
			failed = append(failed, result.Probe)
		}
	}
	event := map[string]any{
		"session_id": "gpu-integrity-watch",
		"source":     "gpu-integrity-watch",
		"type":       "gpu.integrity",
		"severity":   severity,
		"data": map[string]any{
			"verdict":         entry.Verdict,
			"composite_score": entry.CompositeScore,
			"failed_probes":   failed,
		},
	}
	body, err := json.Marshal(event)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPost, parsed.String(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := outboundHTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("incident recorder returned HTTP %d", resp.StatusCode)
	}
	reporter.mu.Lock()
	reporter.lastFingerprint = fingerprint
	reporter.lastReport = time.Now()
	reporter.mu.Unlock()
	return nil
}
