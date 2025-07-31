package neuralnetwork_test

import (
	"io/ioutil"
	"math"
	"os"
	"reflect"
	"testing"

	"go-neuralnetwork/internal/data"
	"go-neuralnetwork/internal/neuralnetwork"
)

func TestInitNetwork(t *testing.T) {
	inputs := 2
	hidden := 3
	outputs := 1
	hiddenActivation := "relu"
	outputActivation := "linear"

	nn := neuralnetwork.InitNetwork(inputs, hidden, outputs, hiddenActivation, outputActivation)

	if nn.NumInputs != inputs {
		t.Errorf("Expected NumInputs to be %d, got %d", inputs, nn.NumInputs)
	}
	if nn.NumHidden != hidden {
		t.Errorf("Expected NumHidden to be %d, got %d", hidden, nn.NumHidden)
	}
	if nn.NumOutputs != outputs {
		t.Errorf("Expected NumOutputs to be %d, got %d", outputs, nn.NumOutputs)
	}
	if nn.HiddenActivation != hiddenActivation {
		t.Errorf("Expected HiddenActivation to be %s, got %s", hiddenActivation, nn.HiddenActivation)
	}
	if nn.OutputActivation != outputActivation {
		t.Errorf("Expected OutputActivation to be %s, got %s", outputActivation, nn.OutputActivation)
	}

	if len(nn.HiddenWeights) != hidden || len(nn.HiddenWeights[0]) != inputs {
		t.Errorf("HiddenWeights dimensions mismatch")
	}
	if len(nn.OutputWeights) != outputs || len(nn.OutputWeights[0]) != hidden {
		t.Errorf("OutputWeights dimensions mismatch")
	}
	if len(nn.HiddenBiases) != hidden {
		t.Errorf("HiddenBiases dimensions mismatch")
	}
	if len(nn.OutputBiases) != outputs {
		t.Errorf("OutputBiases dimensions mismatch")
	}

	// Check if weights and biases are initialized (not all zeros)
	hasNonZeroWeight := false
	for i := range nn.HiddenWeights {
		for j := range nn.HiddenWeights[i] {
			if nn.HiddenWeights[i][j] != 0.0 {
				hasNonZeroWeight = true
				break
			}
		}
		if hasNonZeroWeight {
			break
		}
	}
	if !hasNonZeroWeight {
		t.Errorf("HiddenWeights are all zeros")
	}
}

func TestFeedForward(t *testing.T) {
	// Create a simple network with known weights and biases
	nn := &neuralnetwork.NeuralNetwork{
		NumInputs:        2,
		NumHidden:        2,
		NumOutputs:       1,
		HiddenWeights:    [][]float64{{0.1, 0.2}, {0.3, 0.4}},
		OutputWeights:    [][]float64{{0.5, 0.6}},
		HiddenBiases:     []float64{0.0, 0.0},
		OutputBiases:     []float64{0.0},
		HiddenActivation: "relu",
		OutputActivation: "linear",
	}
	nn.SetActivationFunctions()

	inputs := []float64{1.0, 1.0}

	// Expected calculations:
	// Hidden layer input 1: (1.0 * 0.1) + (1.0 * 0.2) + 0.0 = 0.3 -> utils.Relu(0.3) = 0.3
	// Hidden layer input 2: (1.0 * 0.3) + (1.0 * 0.4) + 0.0 = 0.7 -> utils.Relu(0.7) = 0.7
	// Hidden outputs: [0.3, 0.7]

	// Output layer input: (0.3 * 0.5) + (0.7 * 0.6) + 0.0 = 0.15 + 0.42 = 0.57
	// Final output: 0.57

	hiddenOutputs, finalOutputs := nn.FeedForward(inputs)

	expectedHiddenOutputs := []float64{0.3, 0.7}
	expectedFinalOutputs := []float64{0.57}

	for i := range expectedHiddenOutputs {
		if math.Abs(hiddenOutputs[i]-expectedHiddenOutputs[i]) > 1e-9 {
			t.Errorf("Hidden output mismatch at index %d: Expected %f, got %f", i, expectedHiddenOutputs[i], hiddenOutputs[i])
		}
	}

	for i := range expectedFinalOutputs {
		if math.Abs(finalOutputs[i]-expectedFinalOutputs[i]) > 1e-9 {
			t.Errorf("Final output mismatch at index %d: Expected %f, got %f", i, expectedFinalOutputs[i], finalOutputs[i])
		}
	}
}

func TestSaveAndLoadModel(t *testing.T) {
	// Create a dummy ModelData
	originalNN := neuralnetwork.InitNetwork(2, 2, 1, "relu", "linear")
	originalNN.HiddenWeights = [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	originalNN.OutputWeights = [][]float64{{0.5, 0.6}}
	originalNN.HiddenBiases = []float64{0.7, 0.8}
	originalNN.OutputBiases = []float64{0.9}

	originalMD := &data.ModelData{
		NN:         originalNN,
		TargetMins: []float64{1.0},
		TargetMaxs: []float64{10.0},
		InputMins:  []float64{0.0, 0.0},
		InputMaxs:  []float64{1.0, 1.0},
	}

	// Create a temporary file
	tmpfile, err := ioutil.TempFile("", "model-*.json")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	filePath := tmpfile.Name()
	tmpfile.Close()
	defer os.Remove(filePath)

	// Save the model
	err = originalMD.SaveModel(filePath)
	if err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	// Load the model
	loadedMD, err := data.LoadModel(filePath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	loadedMD.NN.SetActivationFunctions()

	// Compare loaded model with original
	if !reflect.DeepEqual(originalMD.NN.NumInputs, loadedMD.NN.NumInputs) ||
		!reflect.DeepEqual(originalMD.NN.NumHidden, loadedMD.NN.NumHidden) ||
		!reflect.DeepEqual(originalMD.NN.NumOutputs, loadedMD.NN.NumOutputs) ||
		!reflect.DeepEqual(originalMD.NN.HiddenActivation, loadedMD.NN.HiddenActivation) ||
		!reflect.DeepEqual(originalMD.NN.OutputActivation, loadedMD.NN.OutputActivation) ||
		!reflect.DeepEqual(originalMD.NN.HiddenWeights, loadedMD.NN.HiddenWeights) ||
		!reflect.DeepEqual(originalMD.NN.OutputWeights, loadedMD.NN.OutputWeights) ||
		!reflect.DeepEqual(originalMD.NN.HiddenBiases, loadedMD.NN.HiddenBiases) ||
		!reflect.DeepEqual(originalMD.NN.OutputBiases, loadedMD.NN.OutputBiases) ||
		!reflect.DeepEqual(originalMD.TargetMins, loadedMD.TargetMins) ||
		!reflect.DeepEqual(originalMD.TargetMaxs, loadedMD.TargetMaxs) ||
		!reflect.DeepEqual(originalMD.InputMins, loadedMD.InputMins) ||
		!reflect.DeepEqual(originalMD.InputMaxs, loadedMD.InputMaxs) {
		t.Errorf("Loaded model does not match original model")
	}
}
