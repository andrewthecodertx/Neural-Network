package data_test

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"go-neuralnetwork/internal/data"
)

func createTempCSV(content string) (string, error) {
	tmpfile, err := ioutil.TempFile("", "testdata-*.csv")
	if err != nil {
		return "", err
	}
	defer tmpfile.Close()

	_, err = tmpfile.WriteString(content)
	if err != nil {
		return "", err
	}
	return tmpfile.Name(), nil
}

func TestLoadCSV(t *testing.T) {
	// Test case 1: Valid CSV with header
	csvContent1 := `header1,header2,header3
1.0,2.0,3.0
4.0,5.0,6.0`
	filePath1, err := createTempCSV(csvContent1)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath1)

	inputs1, targets1, _, _, inputMins1, inputMaxs1, targetMins1, targetMaxs1, err1 := data.LoadCSV(filePath1)
	if err1 != nil {
		t.Fatalf("Test Case 1: Unexpected error: %v", err1)
	}

	expectedInputs1 := [][]float64{{0.0, 0.0}, {1.0, 1.0}}
	expectedTargets1 := [][]float64{{0.0}, {1.0}}
	expectedTargetMins1 := []float64{3.0}
	expectedTargetMaxs1 := []float64{6.0}
	expectedInputMins1 := []float64{1.0, 2.0}
	expectedInputMaxs1 := []float64{4.0, 5.0}

	if !reflect.DeepEqual(inputs1, expectedInputs1) {
		t.Errorf("Test Case 1: Inputs mismatch. Got %v, Expected %v", inputs1, expectedInputs1)
	}
	if !reflect.DeepEqual(targets1, expectedTargets1) {
		t.Errorf("Test Case 1: Targets mismatch. Got %v, Expected %v", targets1, expectedTargets1)
	}
	if !reflect.DeepEqual(targetMins1, expectedTargetMins1) {
		t.Errorf("Test Case 1: TargetMins mismatch. Got %v, Expected %v", targetMins1, expectedTargetMins1)
	}
	if !reflect.DeepEqual(targetMaxs1, expectedTargetMaxs1) {
		t.Errorf("Test Case 1: TargetMaxs mismatch. Got %v, Expected %v", targetMaxs1, expectedTargetMaxs1)
	}
	if !reflect.DeepEqual(inputMins1, expectedInputMins1) {
		t.Errorf("Test Case 1: InputMins mismatch. Got %v, Expected %v", inputMins1, expectedInputMins1)
	}
	if !reflect.DeepEqual(inputMaxs1, expectedInputMaxs1) {
		t.Errorf("Test Case 1: InputMaxs mismatch. Got %v, Expected %v", inputMaxs1, expectedInputMaxs1)
	}

	// Test case 2: Invalid file path
	_, _, _, _, _, _, _, _, err2 := data.LoadCSV("nonexistent.csv")
	if err2 == nil {
		t.Errorf("Test Case 2: Expected an error for invalid file path, got nil")
	}

	// Test case 3: Empty CSV file
	csvContent3 := ``
	filePath3, err := createTempCSV(csvContent3)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath3)

	_, _, _, _, _, _, _, _, err3 := data.LoadCSV(filePath3)
	if err3 == nil {
		t.Errorf("Test Case 3: Expected an error for empty CSV, got nil")
	}

	// Test case 4: CSV with non-numeric data
	csvContent4 := `header1,header2
abc,1.0`
	filePath4, err := createTempCSV(csvContent4)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath4)

	_, _, _, _, _, _, _, _, err4 := data.LoadCSV(filePath4)
	if err4 == nil {
		t.Errorf("Test Case 4: Expected an error for non-numeric data, got nil")
	}
}
