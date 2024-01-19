package main

import (
	"fmt"
	"math"
	"math/rand"
)

type NeuralNetwork struct {
	inputs       int
	hiddenLayers int
	neurons      int
	outputs      int
	weights      [][]float64
	biases       [][]float64
}

func initNetwork(inputs, hiddenLayers, neurons, outputs int) *NeuralNetwork {
	nn := NeuralNetwork{}
	nn.inputs = inputs
	nn.hiddenLayers = hiddenLayers
	nn.neurons = neurons
	nn.outputs = outputs

	nn.weights = make([][]float64, nn.inputs)
	for i := range nn.weights {
		nn.weights[i] = make([]float64, nn.hiddenLayers)
		for j := range nn.weights[i] {
			nn.weights[i][j] = rand.NormFloat64() / math.Sqrt(float64(nn.hiddenLayers))
		}
	}

	nn.biases = make([][]float64, nn.hiddenLayers)
	for i := range nn.biases {
		nn.biases[i] = make([]float64, nn.neurons)
		for j := range nn.biases[i] {
			nn.biases[i][j] = rand.Float64()
		}
	}

	return &nn
}

func (nn *NeuralNetwork) feedForward(input []float64) float64 {
	inputs := make([][]float64, 1)
	inputs[0] = make([]float64, len(input))
	copy(inputs[0], input)

<<<<<<< HEAD
	product := dotProduct(inputs, nn.weights)

	prediction := product
	for i := 0; i < nn.hiddenLayers; i++ {
		for j := 0; j < nn.neurons; j++ {
			prediction += nn.biases[i][j]
=======
	fmt.Println(inputs, "x", nn.weights)
	product := dotProduct(inputs, nn.weights)

	prediction := product
	fmt.Println(prediction)
	for hiddenLayer := 0; hiddenLayer < nn.hiddenLayers; hiddenLayer++ {
		for neuron := 0; neuron < nn.neurons; neuron++ {
			prediction += nn.biases[hiddenLayer][neuron]
			fmt.Println("layer:", hiddenLayer, "neuron:", neuron, prediction)
>>>>>>> 79a9eed (initial upload to github)
		}
	}

	prediction = sigmoid(prediction)
<<<<<<< HEAD
	return prediction
}

func (nn *NeuralNetwork) train(inputs [][]float64, targets [][]float64, learnRate float64) {
	// TODO: set epochs in the man function and send it as a parameter
	for epoch := 0; epoch < 1000; epoch++ {
		for i := range inputs {
			// forward pass
			prediction := nn.feedForward(inputs[i])

			// loss calculation
			// TODO: this assumes a single target! need to iterate in cases where there might be more.
			loss := calculateLoss(prediction, targets[i][0])

			// backward pass
			gradients := calculateGradients(inputs[i], loss)

			// adjust weights and biases
			nn.updateWeightsAndBiases(gradients, learnRate)
=======

	return prediction
}

func (nn *NeuralNetwork) train(inputs, targets [][]float64, learnRate float64) {
	// TODO: set epochs in the man function and send it as a parameter
	epochs := 1

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < 10; i++ {
			prediction := nn.feedForward(inputs[i])
			target := targets[i][0]
			loss := calculateLoss(prediction, target)

			fmt.Println("inputs:", inputs[i])
			fmt.Println("prediction:", prediction, "target:", target)
			fmt.Println("loss:", loss)
			fmt.Println("---")
>>>>>>> 79a9eed (initial upload to github)
		}
	}
}

<<<<<<< HEAD
func (nn *NeuralNetwork) updateWeightsAndBiases(gradients [][]float64, learnRate float64) {
}

func calculateGradients(input []float64, loss float64) [][]float64 {
	return [][]float64{}
=======
func (nn *NeuralNetwork) calculateGradients(inputs []float64, target float64) ([][]float64, [][]float64) {
	prediction := nn.feedForward(inputs)

	weightGradient := make([][]float64, len(nn.weights))
	biasGradient := make([][]float64, len(nn.biases))

	for i := range nn.weights {
		weightGradient[i] = make([]float64, len(nn.weights[i]))
	}

	for i := range nn.biases {
		biasGradient[i] = make([]float64, len(nn.biases[i]))
	}

	for i := 0; i < nn.hiddenLayers; i++ {
		for j := 0; j < nn.neurons; j++ {
			weightGradient[i][j] = inputs[i] * (prediction - target) * sigmoidDerivative(nn.weights[i][j])
			biasGradient[i][j] = (prediction - target) * sigmoidDerivative(prediction)
		}
	}

	return weightGradient, biasGradient
>>>>>>> 79a9eed (initial upload to github)
}

func calculateLoss(prediction, target float64) float64 {
	diff := prediction - target
	return 0.5 * diff * diff
}

func main() {
	var file string
	var inputCount int
	var outputCount int

	defaultFile := "data.csv"
	defaultInputCount := 11
	defaultOutputCount := 1

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

	inputs, targets := loadCSV(file, inputCount, outputCount)

	nn := initNetwork(inputCount, 4, 4, outputCount)

	// TODO: set learn rate in the main function and send it as a parameter
	nn.train(inputs, targets, 0.1)
}
