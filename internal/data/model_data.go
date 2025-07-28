package data

import (
	"encoding/json"
	"os"

	"go-neuralnetwork/internal/neuralnetwork"
)

type ModelData struct {
	NN         *neuralnetwork.NeuralNetwork `json:"neuralNetwork"`
	TargetMins []float64                    `json:"targetMins"`
	TargetMaxs []float64                    `json:"targetMaxs"`
	InputMins  []float64                    `json:"inputMins"`
	InputMaxs  []float64                    `json:"inputMaxs"`
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
