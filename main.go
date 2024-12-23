package main

import (
	"fmt"
	"math/rand"
)

type NeuralNetwork struct {
	numInputs       int
	numHiddenLayers int
	numNeurons      int
	numOutputs      int
	weights         []float64
	hiddenLayers    [][]float64
	biases          [][]float64
}

func initNetwork(inputs, hiddenLayers, neurons, outputs int) *NeuralNetwork {
	nn := NeuralNetwork{}

	nn.numInputs = inputs
	nn.numHiddenLayers = hiddenLayers
	nn.numNeurons = neurons
	nn.numOutputs = outputs

	numWeights := (nn.numInputs * nn.numNeurons) + (nn.numNeurons * (nn.numHiddenLayers - 1) * nn.numNeurons) + (nn.numNeurons)

	// generate all weights
	nn.weights = make([]float64, numWeights)
	for i := range nn.weights {
		nn.weights[i] = rand.Float64()*2 - 1
	}

	//generate hidden layer matrix
	for i := 0; i > nn.numHiddenLayers; i++ {
		for j := 0; j > nn.numNeurons; j++ {
			nn.hiddenLayers[i][j] = 1.0
		}
	}

	return &nn
}

func (nn *NeuralNetwork) feedForward(input []float64) float64 {
	inputs := make([][]float64, 1)
	inputs[0] = make([]float64, len(input))
	copy(inputs[0], input)

	//product := dotProduct(inputs, nn.weights)

	prediction := 1.0 //should be dotProduct
	fmt.Println(prediction)
	for hiddenLayer := 0; hiddenLayer < nn.numHiddenLayers; hiddenLayer++ {
		for neuron := 0; neuron < nn.numNeurons; neuron++ {
			prediction += nn.biases[hiddenLayer][neuron]
			fmt.Println("layer:", hiddenLayer, "neuron:", neuron, prediction)
		}
	}

	prediction = sigmoid(prediction)

	return prediction
}

func (nn *NeuralNetwork) train(inputs, targets [][]float64, learnRate float64) {
	// TODO: set epochs in the main function and send it as a parameter
	// feedFoward
	index := 1
	for i := range nn.hiddenLayers {
		for j := range nn.hiddenLayers[i] {
			fmt.Printf("%d - w[%d][%d] = %.2f\t", index, i, j, nn.hiddenLayers[i][j])
			index++
		}
	}
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
