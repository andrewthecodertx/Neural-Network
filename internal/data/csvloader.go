package data

import (
	"encoding/csv"
	"os"
	"strconv"
)

func LoadCSV(filePath string) (inputs, targets [][]float64, inputSize, outputSize int, inputMins, inputMaxs, targetMins, targetMaxs []float64, err error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, 0, 0, nil, nil, nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return nil, nil, 0, 0, nil, nil, nil, nil, err
	}

	inputSize = len(header) - 1
	outputSize = 1

	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, 0, 0, nil, nil, nil, nil, err
	}

	// Find min and max for each column to normalize the data
	mins := make([]float64, inputSize+outputSize)
	maxs := make([]float64, inputSize+outputSize)
	for i := range mins {
		mins[i] = 1e9
		maxs[i] = -1e9
	}

	for _, record := range records {
		for i := 0; i < inputSize+outputSize; i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, nil, 0, 0, nil, nil, nil, nil, err
			}
			if val < mins[i] {
				mins[i] = val
			}
			if val > maxs[i] {
				maxs[i] = val
			}
		}
	}

	inputMins = mins[:inputSize]
	inputMaxs = maxs[:inputSize]
	targetMins = mins[inputSize:]
	targetMaxs = maxs[inputSize:]

	for _, record := range records {
		inputRow := make([]float64, inputSize)
		outputRow := make([]float64, outputSize)

		for i := 0; i < inputSize; i++ {
			val, _ := strconv.ParseFloat(record[i], 64)
			if maxs[i]-mins[i] == 0 {
				inputRow[i] = 0
			} else {
				inputRow[i] = (val - mins[i]) / (maxs[i] - mins[i])
			}
		}

		for i := 0; i < outputSize; i++ {
			val, _ := strconv.ParseFloat(record[inputSize+i], 64)
			if maxs[inputSize+i]-mins[inputSize+i] == 0 {
				outputRow[i] = 0
			} else {
				outputRow[i] = (val - mins[inputSize+i]) / (maxs[inputSize+i] - mins[inputSize+i])
			}
		}

		inputs = append(inputs, inputRow)
		targets = append(targets, outputRow)
	}

	return inputs, targets, inputSize, outputSize, inputMins, inputMaxs, targetMins, targetMaxs, nil
}
