package main

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"
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

	inputs1, targets1, targetMins1, targetMaxs1, inputMins1, inputMaxs1 := loadCSV(filePath1, 2, 1)

	expectedInputs1 := [][]float64{{
		0.0, 0.0,
	},
	{
		1.0, 1.0,
	},}
	expectedTargets1 := [][]float64{{
		0.0,
	},
	{
		1.0,
	},}
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
	inputs2, targets2, targetMins2, targetMaxs2, inputMins2, inputMaxs2 := loadCSV("nonexistent.csv", 1, 1)
	if inputs2 != nil || targets2 != nil || targetMins2 != nil || targetMaxs2 != nil || inputMins2 != nil || inputMaxs2 != nil {
		t.Errorf("Test Case 2: Expected nil for invalid file path, got non-nil values")
	}

	// Test case 3: Empty CSV file
	csvContent3 := ``
	filePath3, err := createTempCSV(csvContent3)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath3)

	inputs3, targets3, targetMins3, targetMaxs3, inputMins3, inputMaxs3 := loadCSV(filePath3, 1, 1)
	if inputs3 != nil || targets3 != nil || targetMins3 != nil || targetMaxs3 != nil || inputMins3 != nil || inputMaxs3 != nil {
		t.Errorf("Test Case 3: Expected nil for empty CSV, got non-nil values")
	}

	// Test case 4: CSV with non-numeric data
	csvContent4 := `header1,header2
abc,1.0`
	filePath4, err := createTempCSV(csvContent4)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath4)

	inputs4, targets4, targetMins4, targetMaxs4, inputMins4, inputMaxs4 := loadCSV(filePath4, 1, 1)
	if inputs4 != nil || targets4 != nil || targetMins4 != nil || targetMaxs4 != nil || inputMins4 != nil || inputMaxs4 != nil {
		t.Errorf("Test Case 4: Expected nil for non-numeric data, got non-nil values")
	}
}
