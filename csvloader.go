package main

import (
	"encoding/csv"
	"os"
	"strconv"
)

func loadCSV(filePath string, numInputs, numTargets int) ([][]float64, [][]float64) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// skip the first line (header)
	if _, err := reader.Read(); err != nil {
		return nil, nil
	}

	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil
	}

	var inputs [][]float64
	var outputs [][]float64

	for _, record := range records {
		inputRow := make([]float64, numInputs)
		outputRow := make([]float64, numTargets)

		for i := 0; i < numInputs; i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, nil
			}
			inputRow[i] = val
		}

		for i := 0; i < numTargets; i++ {
			val, err := strconv.ParseFloat(record[numInputs+i], 64)
			if err != nil {
				return nil, nil
			}
			outputRow[i] = val
		}

		inputs = append(inputs, inputRow)
		outputs = append(outputs, outputRow)
	}

	return inputs, outputs
}
