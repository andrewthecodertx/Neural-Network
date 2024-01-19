package main

import "math"

func dotProduct(matrix1 [][]float64, matrix2 [][]float64) float64 {
	rows1, cols1 := len(matrix1), len(matrix1[0])
	rows2, cols2 := len(matrix2), len(matrix2[0])

	if cols1 != rows2 {
		panic("check matrix dimensions")
	}

	product := 0.0

	for row := 0; row < rows1; row++ {
		for col := 0; col < cols2; col++ {
			product += matrix1[row][col] * matrix2[col][row]
		}
	}

	return product
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func relu(x float64) float64 {
	return math.Max(0.0, x)
}

func flatten(matrix [][]float64) []float64 {
	result := []float64{}

	for _, row := range matrix {
		result = append(result, row...)
	}

	return result
}
