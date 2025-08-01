package neuralnetwork

import (
	"fmt"
	"math"
	"math/rand"

	"go-neuralnetwork/internal/utils"
)

type Activation func(float64) float64

var activationFunctions = map[string]Activation{
	"relu":    utils.Relu,
	"sigmoid": utils.Sigmoid,
	"tanh":    utils.Tanh,
	"linear":  utils.Linear,
}

var activationDerivatives = map[string]Activation{
	"relu":    utils.ReluDerivative,
	"sigmoid": utils.SigmoidDerivative,
	"tanh":    utils.TanhDerivative,
	"linear":  utils.LinearDerivative,
}

type NeuralNetwork struct {
	NumInputs             int           `json:"numInputs"`
	HiddenLayers          []int         `json:"hiddenLayers"`
	NumOutputs            int           `json:"numOutputs"`
	HiddenWeights         [][][]float64 `json:"hiddenWeights"`
	OutputWeights         [][]float64   `json:"outputWeights"`
	HiddenBiases          [][]float64   `json:"hiddenBiases"`
	OutputBiases          []float64     `json:"outputBiases"`
	HiddenActivations     []string      `json:"hiddenActivations"`
	OutputActivation      string        `json:"outputActivation"`
	hiddenActivationFuncs []Activation  `json:"-"`
	outputActivationFunc  Activation    `json:"-"`
	hiddenDerivativeFuncs []Activation  `json:"-"`
	outputDerivativeFunc  Activation    `json:"-"`
}

func InitNetwork(inputs int, hiddenLayers []int, outputs int, hiddenActivations []string, outputActivation string) *NeuralNetwork {
	hiddenWeights := make([][][]float64, len(hiddenLayers))
	hiddenBiases := make([][]float64, len(hiddenLayers))
	prevLayerSize := inputs

	for i, layerSize := range hiddenLayers {
		hiddenWeights[i] = make([][]float64, layerSize)
		heInit := math.Sqrt(2.0 / float64(prevLayerSize))
		for j := range hiddenWeights[i] {
			hiddenWeights[i][j] = make([]float64, prevLayerSize)
			for k := range hiddenWeights[i][j] {
				hiddenWeights[i][j][k] = rand.NormFloat64() * heInit
			}
		}
		hiddenBiases[i] = make([]float64, layerSize)
		prevLayerSize = layerSize
	}

	outputWeights := make([][]float64, outputs)
	heInitOutput := math.Sqrt(2.0 / float64(prevLayerSize))
	for i := range outputWeights {
		outputWeights[i] = make([]float64, prevLayerSize)
		for j := range outputWeights[i] {
			outputWeights[i][j] = rand.NormFloat64() * heInitOutput
		}
	}
	outputBiases := make([]float64, outputs)

	nn := &NeuralNetwork{
		NumInputs:         inputs,
		HiddenLayers:      hiddenLayers,
		NumOutputs:        outputs,
		HiddenWeights:     hiddenWeights,
		OutputWeights:     outputWeights,
		HiddenBiases:      hiddenBiases,
		OutputBiases:      outputBiases,
		HiddenActivations: hiddenActivations,
		OutputActivation:  outputActivation,
	}
	nn.SetActivationFunctions()
	return nn
}

func (nn *NeuralNetwork) SetActivationFunctions() {
	nn.hiddenActivationFuncs = make([]Activation, len(nn.HiddenActivations))
	nn.hiddenDerivativeFuncs = make([]Activation, len(nn.HiddenActivations))
	for i, activation := range nn.HiddenActivations {
		nn.hiddenActivationFuncs[i] = activationFunctions[activation]
		nn.hiddenDerivativeFuncs[i] = activationDerivatives[activation]
	}
	nn.outputActivationFunc = activationFunctions[nn.OutputActivation]
	nn.outputDerivativeFunc = activationDerivatives[nn.OutputActivation]
}

func (nn *NeuralNetwork) FeedForward(inputs []float64) ([][]float64, []float64) {
	hiddenOutputs := make([][]float64, len(nn.HiddenLayers))
	layerInput := inputs

	for i, layerSize := range nn.HiddenLayers {
		hiddenOutputs[i] = make([]float64, layerSize)
		for j := 0; j < layerSize; j++ {
			sum := 0.0
			for k := 0; k < len(layerInput); k++ {
				sum += layerInput[k] * nn.HiddenWeights[i][j][k]
			}
			hiddenOutputs[i][j] = nn.hiddenActivationFuncs[i](sum + nn.HiddenBiases[i][j])
		}
		layerInput = hiddenOutputs[i]
	}

	finalOutputs := make([]float64, nn.NumOutputs)
	for i := 0; i < nn.NumOutputs; i++ {
		sum := 0.0
		for j := 0; j < len(layerInput); j++ {
			sum += layerInput[j] * nn.OutputWeights[i][j]
		}
		finalOutputs[i] = nn.outputActivationFunc(sum + nn.OutputBiases[i])
	}

	return hiddenOutputs, finalOutputs
}

func (nn *NeuralNetwork) Backpropagate(inputs []float64, targets []float64, hiddenOutputs [][]float64, finalOutputs []float64, learningRate float64) {
	outputErrors := make([]float64, nn.NumOutputs)
	outputDeltas := make([]float64, nn.NumOutputs)
	for i := 0; i < nn.NumOutputs; i++ {
		outputErrors[i] = targets[i] - finalOutputs[i]
		outputDeltas[i] = outputErrors[i] * nn.outputDerivativeFunc(finalOutputs[i])
	}

	hiddenErrors := make([][]float64, len(nn.HiddenLayers))
	hiddenDeltas := make([][]float64, len(nn.HiddenLayers))
	nextLayerDeltas := outputDeltas
	nextLayerWeights := nn.OutputWeights

	for i := len(nn.HiddenLayers) - 1; i >= 0; i-- {
		layerSize := nn.HiddenLayers[i]
		hiddenErrors[i] = make([]float64, layerSize)
		hiddenDeltas[i] = make([]float64, layerSize)
		for j := 0; j < layerSize; j++ {
			sum := 0.0
			for k := 0; k < len(nextLayerDeltas); k++ {
				sum += nextLayerDeltas[k] * nextLayerWeights[k][j]
			}
			hiddenErrors[i][j] = sum
			hiddenDeltas[i][j] = hiddenErrors[i][j] * nn.hiddenDerivativeFuncs[i](hiddenOutputs[i][j])
		}
		nextLayerDeltas = hiddenDeltas[i]
		if i > 0 {
			nextLayerWeights = nn.HiddenWeights[i]
		}
	}

	// Update output weights and biases
	lastHiddenLayerOutput := hiddenOutputs[len(hiddenOutputs)-1]
	for i := 0; i < nn.NumOutputs; i++ {
		for j := 0; j < len(lastHiddenLayerOutput); j++ {
			nn.OutputWeights[i][j] += learningRate * outputDeltas[i] * lastHiddenLayerOutput[j]
		}
		nn.OutputBiases[i] += learningRate * outputDeltas[i]
	}

	// Update hidden weights and biases
	for i := len(nn.HiddenLayers) - 1; i >= 0; i-- {
		var prevLayerOutput []float64
		if i == 0 {
			prevLayerOutput = inputs
		} else {
			prevLayerOutput = hiddenOutputs[i-1]
		}
		for j := 0; j < nn.HiddenLayers[i]; j++ {
			for k := 0; k < len(prevLayerOutput); k++ {
				nn.HiddenWeights[i][j][k] += learningRate * hiddenDeltas[i][j] * prevLayerOutput[k]
			}
			nn.HiddenBiases[i][j] += learningRate * hiddenDeltas[i][j]
		}
	}
}

func (nn *NeuralNetwork) Train(inputs, targets [][]float64, epochs int, learningRate float64, errorGoal float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		for i := range inputs {
			hiddenOutputs, finalOutputs := nn.FeedForward(inputs[i])
			nn.Backpropagate(inputs[i], targets[i], hiddenOutputs, finalOutputs, learningRate)
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
