package neuralnetwork

import "math"

// Activation is an interface for activation functions.
type Activation interface {
	Activate(float64) float64
	Derivative(float64) float64
}

// ReLU is the Rectified Linear Unit activation function.
type ReLU struct{}

// Activate applies the ReLU function.
func (r *ReLU) Activate(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// Derivative calculates the derivative of the ReLU function.
func (r *ReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Sigmoid is the sigmoid activation function.
type Sigmoid struct{}

// Activate applies the sigmoid function.
func (s *Sigmoid) Activate(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Derivative calculates the derivative of the sigmoid function.
func (s *Sigmoid) Derivative(x float64) float64 {
	return s.Activate(x) * (1 - s.Activate(x))
}

// Tanh is the hyperbolic tangent activation function.
type Tanh struct{}

// Activate applies the tanh function.
func (t *Tanh) Activate(x float64) float64 {
	return math.Tanh(x)
}

// Derivative calculates the derivative of the tanh function.
func (t *Tanh) Derivative(x float64) float64 {
	return 1 - math.Pow(t.Activate(x), 2)
}

// Linear is the linear activation function.
type Linear struct{}

// Activate applies the linear function.
func (l *Linear) Activate(x float64) float64 {
	return x
}

// Derivative calculates the derivative of the linear function.
func (l *Linear) Derivative(x float64) float64 {
	return 1
}
