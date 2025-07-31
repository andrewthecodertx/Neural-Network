package utils

import "math"

func Relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func TanhDerivative(x float64) float64 {
	return 1 - x*x
}

func Linear(x float64) float64 {
	return x
}

func LinearDerivative(x float64) float64 {
	return 1
}
