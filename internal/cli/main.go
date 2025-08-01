package cli

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"go-neuralnetwork/internal/data"
	"go-neuralnetwork/internal/neuralnetwork"
	"go-neuralnetwork/internal/tui"
)

const modelsDir = "saved_models"

func RunCLI() {
	if err := os.MkdirAll(modelsDir, os.ModePerm); err != nil {
		log.Fatalf("Failed to create models directory: %v", err)
	}

	if len(os.Args) < 2 {
		printUsage()
		return
	}

	command := os.Args[1]
	switch command {
	case "train":
		runTrainingWithTUI()
	case "predict":
		runPrediction()
	default:
		fmt.Println("Unknown command:", command)
		printUsage()
	}
}

func printUsage() {
	fmt.Println("Usage: go-neuralnetwork <command>")
	fmt.Println("Commands:")
	fmt.Println("  train      Launch the interactive TUI to train a new model")
	fmt.Println("  predict    Load a model and make a prediction")
}

func runTrainingWithTUI() {
	// Find available CSV files
	csvFiles, err := filepath.Glob("*.csv")
	if err != nil || len(csvFiles) == 0 {
		log.Fatalf("No CSV files found in the current directory.")
	}

	// Launch the TUI
	tuiModel := tui.New(csvFiles)
	params, err := tuiModel.Run()
	if err != nil {
		if err.Error() == "training cancelled" {
			fmt.Println("Training cancelled.")
			os.Exit(0)
		}
		log.Fatalf("Could not start TUI: %v", err)
	}

	// Load the data using the selected CSV
	inputs, targets, inputSize, outputSize, inputMins, inputMaxs, targetMins, targetMaxs, err := data.LoadCSV(params.CsvPath)
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}
	fmt.Printf("Loaded %s: %d inputs, %d outputs\n", params.CsvPath, inputSize, outputSize)

	// Initialize and train the network with parameters from the TUI
	nn := neuralnetwork.InitNetwork(inputSize, params.HiddenLayers, outputSize, params.HiddenActivations, params.OutputActivation)
	nn.Train(inputs, targets, params.Epochs, params.LearningRate, params.ErrorGoal)

	fmt.Println("\nTraining complete.")

	// Prompt to save the model
	fmt.Print("Enter a name to save this model (or press Enter to skip): ")
	reader := bufio.NewReader(os.Stdin)
	modelName, _ := reader.ReadString('\n')
	modelName = strings.TrimSpace(modelName)

	if modelName != "" {
		modelPath := filepath.Join(modelsDir, modelName+".json")
		modelData := &data.ModelData{
			NN:         nn,
			InputMins:  inputMins,
			InputMaxs:  inputMaxs,
			TargetMins: targetMins,
			TargetMaxs: targetMaxs,
		}
		if err := modelData.SaveModel(modelPath); err != nil {
			log.Fatalf("Failed to save model: %v", err)
		}
		fmt.Printf("Model saved to %s\n", modelPath)
	}
}

func runPrediction() {
	files, err := os.ReadDir(modelsDir)
	if err != nil {
		log.Fatalf("Failed to read models directory: %v", err)
	}

	var models []string
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(file.Name(), ".json") {
			models = append(models, file.Name())
		}
	}

	if len(models) == 0 {
		fmt.Println("No saved models found in the 'saved_models' directory.")
		return
	}

	fmt.Println("Please select a model to load:")
	for i, modelName := range models {
		fmt.Printf("  %d: %s\n", i+1, modelName)
	}

	fmt.Print("Enter the number of the model: ")
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	selection := strings.TrimSpace(input)

	selectedIndex := 0
	_, err = fmt.Sscanf(selection, "%d", &selectedIndex)
	if err != nil || selectedIndex < 1 || selectedIndex > len(models) {
		log.Fatalf("Invalid selection.")
	}

	modelName := models[selectedIndex-1]
	modelPath := filepath.Join(modelsDir, modelName)

	modelData, err := data.LoadModel(modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	modelData.NN.SetActivationFunctions()
	fmt.Printf("Model '%s' loaded successfully.\n\n", modelName)

	// Check if the model has the required normalization data
	if modelData.InputMins == nil || modelData.InputMaxs == nil || modelData.TargetMins == nil || modelData.TargetMaxs == nil {
		log.Fatalf("Error: The selected model '%s' is outdated or was not saved correctly. It is missing the required normalization data. Please train and save a new model.", modelName)
	}

	// Get new data from user
	fmt.Printf("Enter the %d input values, separated by commas: ", modelData.NN.NumInputs)
	inputStr, _ := reader.ReadString('\n')
	inputStrs := strings.Split(strings.TrimSpace(inputStr), ",")

	if len(inputStrs) != modelData.NN.NumInputs {
		log.Fatalf("Error: Expected %d input values, but got %d.", modelData.NN.NumInputs, len(inputStrs))
	}

	// Parse and normalize the input
	predictionInput := make([]float64, modelData.NN.NumInputs)
	for i, s := range inputStrs {
		val, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil {
			log.Fatalf("Invalid input value: %v", err)
		}
		// Normalize
		predictionInput[i] = (val - modelData.InputMins[i]) / (modelData.InputMaxs[i] - modelData.InputMins[i])
	}

	// Make the prediction
	_, predictionOutput := modelData.NN.FeedForward(predictionInput)

	// Denormalize the output
	finalPrediction := predictionOutput[0]*(modelData.TargetMaxs[0]-modelData.TargetMins[0]) + modelData.TargetMins[0]

	fmt.Printf("\nPredicted Output: %f\n", finalPrediction)
}