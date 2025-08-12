package data_test

import (
	"os"
	"reflect"
	"sort"
	"testing"

	"go-neuralnetwork/internal/data"
	"go-neuralnetwork/internal/tempfile"
)

func TestLoadCSV(t *testing.T) {
	// Test case 1: Valid CSV with header
	csvContent1 := `header1,header2,header3
1.0,2.0,3.0
4.0,5.0,6.0`
	filePath1, err := tempfile.CreateTempFileWithContent("testdata-*.csv", csvContent1)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath1)

	dataset, err := data.LoadCSV(filePath1, 1.0)
	if err != nil {
		t.Fatalf("Test Case 1: Unexpected error: %v", err)
	}

	// The LoadCSV function shuffles the data, so we need to sort it to have a deterministic test.
	type labeledData struct {
		input  []float64
		target []float64
	}

	var combined []labeledData
	for i := range dataset.TrainInputs {
		combined = append(combined, labeledData{input: dataset.TrainInputs[i], target: dataset.TrainTargets[i]})
	}

	// Sort based on the first element of the input slice.
	sort.Slice(combined, func(i, j int) bool {
		if len(combined[i].input) == 0 || len(combined[j].input) == 0 {
			return false
		}
		return combined[i].input[0] < combined[j].input[0]
	})

	sortedInputs := make([][]float64, len(combined))
	sortedTargets := make([][]float64, len(combined))
	for i, d := range combined {
		sortedInputs[i] = d.input
		sortedTargets[i] = d.target
	}

	expectedInputs1 := [][]float64{{0.0, 0.0}, {1.0, 1.0}}
	expectedTargets1 := [][]float64{{0.0}, {1.0}}
	expectedTargetMins1 := []float64{3.0}
	expectedTargetMaxs1 := []float64{6.0}
	expectedInputMins1 := []float64{1.0, 2.0}
	expectedInputMaxs1 := []float64{4.0, 5.0}

	if !reflect.DeepEqual(sortedInputs, expectedInputs1) {
		t.Errorf("Test Case 1: Inputs mismatch. Got %v, Expected %v", sortedInputs, expectedInputs1)
	}
	if !reflect.DeepEqual(sortedTargets, expectedTargets1) {
		t.Errorf("Test Case 1: Targets mismatch. Got %v, Expected %v", sortedTargets, expectedTargets1)
	}
	if !reflect.DeepEqual(dataset.TargetMins, expectedTargetMins1) {
		t.Errorf("Test Case 1: TargetMins mismatch. Got %v, Expected %v", dataset.TargetMins, expectedTargetMins1)
	}
	if !reflect.DeepEqual(dataset.TargetMaxs, expectedTargetMaxs1) {
		t.Errorf("Test Case 1: TargetMaxs mismatch. Got %v, Expected %v", dataset.TargetMaxs, expectedTargetMaxs1)
	}
	if !reflect.DeepEqual(dataset.InputMins, expectedInputMins1) {
		t.Errorf("Test Case 1: InputMins mismatch. Got %v, Expected %v", dataset.InputMins, expectedInputMins1)
	}
	if !reflect.DeepEqual(dataset.InputMaxs, expectedInputMaxs1) {
		t.Errorf("Test Case 1: InputMaxs mismatch. Got %v, Expected %v", dataset.InputMaxs, expectedInputMaxs1)
	}
	if len(dataset.TestInputs) != 0 {
		t.Errorf("Test Case 1: TestInputs should be empty with splitRatio 1.0, got %v", dataset.TestInputs)
	}
	if len(dataset.TestTargets) != 0 {
		t.Errorf("Test Case 1: TestTargets should be empty with splitRatio 1.0, got %v", dataset.TestTargets)
	}

	// Test case 2: Invalid file path
	_, err = data.LoadCSV("nonexistent.csv", 1.0)
	if err == nil {
		t.Errorf("Test Case 2: Expected an error for invalid file path, got nil")
	}

	// Test case 3: Empty CSV file
	csvContent3 := ``
	filePath3, err := tempfile.CreateTempFileWithContent("testdata-*.csv", csvContent3)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath3)

	_, err = data.LoadCSV(filePath3, 1.0)
	if err == nil {
		t.Errorf("Test Case 3: Expected an error for empty CSV, got nil")
	}

	// Test case 4: CSV with non-numeric data
	csvContent4 := `header1,header2
abc,1.0`
	filePath4, err := tempfile.CreateTempFileWithContent("testdata-*.csv", csvContent4)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath4)

	_, err = data.LoadCSV(filePath4, 1.0)
	if err == nil {
		t.Errorf("Test Case 4: Expected an error for non-numeric data, got nil")
	}
}
