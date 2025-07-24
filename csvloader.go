package main

import (
	"encoding/csv"
	"os"
	"strconv"
)

func loadCSV(filePath string, numInputs, numTargets int) ([][]float64, [][]float64, []float64, []float64, []float64, []float64) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, nil, nil, nil, nil
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// skip the first line (header)
	if _, err := reader.Read(); err != nil {
		return nil, nil, nil, nil, nil, nil
	}

	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, nil, nil, nil, nil
	}

	var inputs [][]float64
	var outputs [][]float64

	// Find min and max for each column to normalize the data
	mins := make([]float64, numInputs+numTargets)
	maxs := make([]float64, numInputs+numTargets)
	for i := range mins {
		mins[i] = 1e9
		maxs[i] = -1e9
	}

	for _, record := range records {
		for i := 0; i < numInputs+numTargets; i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, nil, nil, nil, nil, nil
			}
			if val < mins[i] {
				mins[i] = val
			}
			if val > maxs[i] {
				maxs[i] = val
			}
		}
	}

	targetMins := make([]float64, numTargets)
	targetMaxs := make([]float64, numTargets)
	for i := 0; i < numTargets; i++ {
		targetMins[i] = mins[numInputs+i]
		targetMaxs[i] = maxs[numInputs+i]
	}

	for _, record := range records {
		inputRow := make([]float64, numInputs)
		outputRow := make([]float64, numTargets)

		for i := 0; i < numInputs; i++ {
			val, _ := strconv.ParseFloat(record[i], 64)
			inputRow[i] = (val - mins[i]) / (maxs[i] - mins[i])
		}

		for i := 0; i < numTargets; i++ {
			val, _ := strconv.ParseFloat(record[numInputs+i], 64)
			outputRow[i] = (val - mins[numInputs+i]) / (maxs[numInputs+i] - mins[numInputs+i])
		}

		inputs = append(inputs, inputRow)
		outputs = append(outputs, outputRow)
	}

	return inputs, outputs, targetMins, targetMaxs, mins[:numInputs], maxs[:numInputs]
}
