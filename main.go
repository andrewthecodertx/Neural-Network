package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	var choice string
	fmt.Print("Do you want to (t)rain a new model or (l)oad an existing model? (t/l): ")
	fmt.Scanln(&choice)

	if choice == "t" || choice == "T" {
		trainModel()
	} else if choice == "l" || choice == "L" {
		loadAndPredict()
	} else {
		fmt.Println("Invalid choice. Please enter 't' or 'l'.")
	}
}

func trainModel() {
	reader := bufio.NewReader(os.Stdin)

	file := getStringWithDefault(reader, "data file", "data.csv")
	inputCount := getIntWithDefault(reader, "inputs", 11)
	outputCount := getIntWithDefault(reader, "outputs", 1)
	hiddenCount := getIntWithDefault(reader, "hidden neurons", 16)
	epochs := getIntWithDefault(reader, "epochs", 200)
	learningRate := getFloatWithDefault(reader, "learning rate", 0.05)
	errorGoal := getFloatWithDefault(reader, "error goal", 0.005)

	inputs, targets, targetMins, targetMaxs, inputMins, inputMaxs := loadCSV(file, inputCount, outputCount)

	nn := InitNetwork(inputCount, hiddenCount, outputCount)

	nn.Train(inputs, targets, epochs, learningRate, errorGoal)

	modelFileName := "model.json"
	md := &ModelData{
		NN:         nn,
		TargetMins: targetMins,
		TargetMaxs: targetMaxs,
		InputMins:  inputMins,
		InputMaxs:  inputMaxs,
	}
	err := md.SaveModel(modelFileName)
	if err != nil {
		fmt.Printf("Error saving model: %v\n", err)
	} else {
		fmt.Printf("Model saved to %s\n", modelFileName)
	}
}

func loadAndPredict() {
	modelFileName := "model.json"
	md, err := LoadModel(modelFileName)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		return
	}
	fmt.Printf("Model loaded from %s\n", modelFileName)

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Enter input values for prediction (comma-separated):")
	inputValuesStr, _ := reader.ReadString('\n')
	inputValuesStr = strings.TrimSpace(inputValuesStr)

	// Parse input values
	inputStrings := strings.Split(inputValuesStr, ",")
	inputValues := make([]float64, len(inputStrings))
	for i, s := range inputStrings {
		val, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil {
			fmt.Printf("Invalid input: %v\n", err)
			return
		}
		inputValues[i] = val
	}

	// Normalize input values
	if len(inputValues) != md.NN.NumInputs {
		fmt.Printf("Expected %d input values, got %d\n", md.NN.NumInputs, len(inputValues))
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

	_, prediction := md.NN.FeedForward(normalizedInput)
	denormalizedPrediction := prediction[0]*(md.TargetMaxs[0]-md.TargetMins[0]) + md.TargetMins[0]
	fmt.Printf("Prediction for input: %v\n", denormalizedPrediction)
}

func getStringWithDefault(reader *bufio.Reader, prompt, defaultValue string) string {
	fmt.Printf("%s (default: %s): ", prompt, defaultValue)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	if input == "" {
		return defaultValue
	}
	return input
}

func getIntWithDefault(reader *bufio.Reader, prompt string, defaultValue int) int {
	fmt.Printf("%s (default: %d): ", prompt, defaultValue)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	if input == "" {
		return defaultValue
	}
	val, err := strconv.Atoi(input)
	if err != nil {
		fmt.Printf("Invalid input, using default value: %d\n", defaultValue)
		return defaultValue
	}
	return val
}

func getFloatWithDefault(reader *bufio.Reader, prompt string, defaultValue float64) float64 {
	fmt.Printf("%s (default: %.4f): ", prompt, defaultValue)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	if input == "" {
		return defaultValue
	}
	val, err := strconv.ParseFloat(input, 64)
	if err != nil {
		fmt.Printf("Invalid input, using default value: %.4f\n", defaultValue)
		return defaultValue
	}
	return val
}
