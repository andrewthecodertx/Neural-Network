package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

type NeuralNetwork struct {
	NumInputs     int         `json:"numInputs"`
	NumHidden     int         `json:"numHidden"`
	NumOutputs    int         `json:"numOutputs"`
	HiddenWeights [][]float64 `json:"hiddenWeights"`
	OutputWeights [][]float64 `json:"outputWeights"`
	HiddenBiases  []float64   `json:"hiddenBiases"`
	OutputBiases  []float64   `json:"outputBiases"`
}

func initNetwork(inputs, hidden, outputs int) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

	hiddenWeights := make([][]float64, hidden)
	heInitHidden := math.Sqrt(2.0 / float64(inputs))
	for i := range hiddenWeights {
		hiddenWeights[i] = make([]float64, inputs)
		for j := range hiddenWeights[i] {
			hiddenWeights[i][j] = rand.NormFloat64() * heInitHidden
		}
	}

	outputWeights := make([][]float64, outputs)
	heInitOutput := math.Sqrt(2.0 / float64(hidden))
	for i := range outputWeights {
		outputWeights[i] = make([]float64, hidden)
		for j := range outputWeights[i] {
			outputWeights[i][j] = rand.NormFloat64() * heInitOutput
		}
	}

	hiddenBiases := make([]float64, hidden)
	outputBiases := make([]float64, outputs)

	return &NeuralNetwork{
		NumInputs:     inputs,
		NumHidden:     hidden,
		NumOutputs:    outputs,
		HiddenWeights: hiddenWeights,
		OutputWeights: outputWeights,
		HiddenBiases:  hiddenBiases,
		OutputBiases:  outputBiases,
	}
}

func (nn *NeuralNetwork) feedForward(inputs []float64) ([]float64, []float64) {
	// Calculate hidden layer outputs
	hiddenOutputs := make([]float64, nn.NumHidden)
	for i := 0; i < nn.NumHidden; i++ {
		sum := 0.0
		for j := 0; j < nn.NumInputs; j++ {
			sum += inputs[j] * nn.HiddenWeights[i][j]
		}
		hiddenOutputs[i] = relu(sum + nn.HiddenBiases[i])
	}

	// Calculate final outputs
	finalOutputs := make([]float64, nn.NumOutputs)
	for i := 0; i < nn.NumOutputs; i++ {
		sum := 0.0
		for j := 0; j < nn.NumHidden; j++ {
			sum += hiddenOutputs[j] * nn.OutputWeights[i][j]
		}
		finalOutputs[i] = sum + nn.OutputBiases[i] // Linear output
	}

	return hiddenOutputs, finalOutputs
}

func (nn *NeuralNetwork) backpropagate(inputs, targets, hiddenOutputs, finalOutputs []float64, learningRate float64) {
	// Calculate output errors and deltas
	outputErrors := make([]float64, nn.NumOutputs)
	outputDeltas := make([]float64, nn.NumOutputs)
	for i := 0; i < nn.NumOutputs; i++ {
		outputErrors[i] = targets[i] - finalOutputs[i]
		outputDeltas[i] = outputErrors[i] // Derivative of linear function is 1
	}

	// Calculate hidden layer errors and deltas
	hiddenErrors := make([]float64, nn.NumHidden)
	hiddenDeltas := make([]float64, nn.NumHidden)
	for i := 0; i < nn.NumHidden; i++ {
		sum := 0.0
		for j := 0; j < nn.NumOutputs; j++ {
			sum += outputDeltas[j] * nn.OutputWeights[j][i]
		}
		hiddenErrors[i] = sum
		hiddenDeltas[i] = hiddenErrors[i] * reluDerivative(hiddenOutputs[i])
	}

	// Update output weights and biases
	for i := 0; i < nn.NumOutputs; i++ {
		for j := 0; j < nn.NumHidden; j++ {
			nn.OutputWeights[i][j] += learningRate * outputDeltas[i] * hiddenOutputs[j]
		}
		nn.OutputBiases[i] += learningRate * outputDeltas[i]
	}

	// Update hidden weights and biases
	for i := 0; i < nn.NumHidden; i++ {
		for j := 0; j < nn.NumInputs; j++ {
			nn.HiddenWeights[i][j] += learningRate * hiddenDeltas[i] * inputs[j]
		}
		nn.HiddenBiases[i] += learningRate * hiddenDeltas[i]
	}
}

func (nn *NeuralNetwork) train(inputs, targets [][]float64, epochs int, learningRate float64, errorGoal float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		for i := range inputs {
			// Forward pass
			hiddenOutputs, finalOutputs := nn.feedForward(inputs[i])

			// Backpropagation
			nn.backpropagate(inputs[i], targets[i], hiddenOutputs, finalOutputs, learningRate)

			// Calculate error for logging
			for j := range targets[i] {
				totalError += 0.5 * (targets[i][j] - finalOutputs[j]) * (targets[i][j] - finalOutputs[j])
			}
		}
		avgError := totalError / float64(len(inputs))
		fmt.Printf("\rEpoch %d/%d, Error: %f", epoch+1, epochs, avgError)

		if avgError < errorGoal {
			fmt.Printf("\nError goal reached at epoch %d\n", epoch+1)
			break
		}
	}
	fmt.Println()
}

type ModelData struct {
	NN         *NeuralNetwork `json:"neuralNetwork"`
	TargetMins []float64      `json:"targetMins"`
	TargetMaxs []float64      `json:"targetMaxs"`
	InputMins  []float64      `json:"inputMins"`
	InputMaxs  []float64      `json:"inputMaxs"`
}

func (md *ModelData) saveModel(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(md)
}

func loadModel(filePath string) (*ModelData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var md ModelData
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&md)
	if err != nil {
		return nil, err
	}
	return &md, nil
}

func main() {
	var choice string
	fmt.Print("Do you want to (t)rain a new model or (l)oad an existing model? (t/l): ")
	fmt.Scanln(&choice)

	if choice == "t" || choice == "T" {
		var file string
		var inputCount int
		var outputCount int
		var hiddenCount int
		var epochs int
		var learningRate float64
		var errorGoal float64

		defaultFile := "data.csv"
		defaultInputCount := 11
		defaultOutputCount := 1
		defaultHiddenCount := 16
		defaultEpochs := 200
		defaultLearningRate := 0.05
		defaultErrorGoal := 0.005

		fmt.Printf("data file (default: %s): ", defaultFile)
		fmt.Scanln(&file)
		if file == "" {
			file = defaultFile
		}

		fmt.Printf("inputs (default: %d): ", defaultInputCount)
		fmt.Scanln(&inputCount)
		if inputCount == 0 {
			inputCount = defaultInputCount
		}

		fmt.Printf("outputs (default: %d): ", defaultOutputCount)
		fmt.Scanln(&outputCount)
		if outputCount == 0 {
			outputCount = defaultOutputCount
		}

		fmt.Printf("hidden neurons (default: %d): ", defaultHiddenCount)
		fmt.Scanln(&hiddenCount)
		if hiddenCount == 0 {
			hiddenCount = defaultHiddenCount
		}

		fmt.Printf("epochs (default: %d): ", defaultEpochs)
		fmt.Scanln(&epochs)
		if epochs == 0 {
			epochs = defaultEpochs
		}

		fmt.Printf("learning rate (default: %.2f): ", defaultLearningRate)
		fmt.Scanln(&learningRate)
		if learningRate == 0.0 {
			learningRate = defaultLearningRate
		}

		fmt.Printf("error goal (default: %.4f): ", defaultErrorGoal)
		fmt.Scanln(&errorGoal)
		if errorGoal == 0.0 {
			errorGoal = defaultErrorGoal
		}

		inputs, targets, targetMins, targetMaxs, inputMins, inputMaxs := loadCSV(file, inputCount, outputCount)

		nn := initNetwork(inputCount, hiddenCount, outputCount)

		nn.train(inputs, targets, epochs, learningRate, errorGoal)

		modelFileName := "model.json"
		md := &ModelData{
			NN:         nn,
			TargetMins: targetMins,
			TargetMaxs: targetMaxs,
			InputMins:  inputMins,
			InputMaxs:  inputMaxs,
		}
		err := md.saveModel(modelFileName)
		if err != nil {
			fmt.Printf("Error saving model: %v\n", err)
		} else {
			fmt.Printf("Model saved to %s\n", modelFileName)
		}
	} else if choice == "l" || choice == "L" {
		modelFileName := "model.json"
		md, err := loadModel(modelFileName)
		if err != nil {
			fmt.Printf("Error loading model: %v", err)
			return
		}
		fmt.Printf("Model loaded from %s", modelFileName)

		fmt.Println("Enter input values for prediction (space-separated):")
		var inputValuesStr string
		fmt.Scanln(&inputValuesStr)

		// Parse input values
		inputStrings := strings.Split(inputValuesStr, ",")
		inputValues := make([]float64, len(inputStrings))
		for i, s := range inputStrings {
			val, err := strconv.ParseFloat(s, 64)
			if err != nil {
				fmt.Printf("Invalid input: %v", err)
				return
			}
			inputValues[i] = val
		}

		// Normalize input values
		if len(inputValues) != md.NN.NumInputs {
			fmt.Printf("Expected %d input values, got %d", md.NN.NumInputs, len(inputValues))
			return
		}
		normalizedInput := make([]float64, md.NN.NumInputs)
		for i := range inputValues {
			if md.InputMaxs[i]-md.InputMins[i] == 0 {
				normalizedInput[i] = 0 // Avoid division by zero
			} else {
				normalizedInput[i] = (inputValues[i] - md.InputMins[i]) / (md.InputMaxs[i] - md.InputMins[i])
			}
		}

		_, prediction := md.NN.feedForward(normalizedInput)
		denormalizedPrediction := prediction[0]*(md.TargetMaxs[0]-md.TargetMins[0]) + md.TargetMins[0]
		fmt.Printf("Prediction for input: %v", denormalizedPrediction)

	} else {
		fmt.Println("Invalid choice. Please enter 't' or 'l'.")
	}
}
