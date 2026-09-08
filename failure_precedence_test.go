package main

import "testing"

func TestHardFailureDominatesEveryUnavailableProbeOrder(t *testing.T) {
	for _, status := range []ProbeStatus{StatusSkip, StatusError, StatusUnknown} {
		for _, results := range [][]ProbeResult{
			{{Status: status}, {Status: StatusFail}},
			{{Status: StatusFail}, {Status: status}},
		} {
			if got := classifyVerdict(0, results); got != VerdictCritical {
				t.Fatalf("%+v classified as %s", results, got)
			}
		}
	}
}
