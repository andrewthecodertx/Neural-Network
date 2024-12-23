package main

import "math"

func dotProduct(inputs, weights [][]float64) []float64 {


  return make([]float64, 11)
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
