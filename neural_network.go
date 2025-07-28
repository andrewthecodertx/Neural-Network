package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
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

func InitNetwork(inputs, hidden, outputs int) *NeuralNetwork {
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

func (nn *NeuralNetwork) FeedForward(inputs []float64) ([]float64, []float64) {
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

func (nn *NeuralNetwork) Backpropagate(inputs, targets, hiddenOutputs, finalOutputs []float64, learningRate float64) {
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

func (nn *NeuralNetwork) Train(inputs, targets [][]float64, epochs int, learningRate float64, errorGoal float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		for i := range inputs {
			// Forward pass
			hiddenOutputs, finalOutputs := nn.FeedForward(inputs[i])

			// Backpropagation
			nn.Backpropagate(inputs[i], targets[i], hiddenOutputs, finalOutputs, learningRate)

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

func (md *ModelData) SaveModel(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(md)
}

func LoadModel(filePath string) (*ModelData, error) {
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

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
