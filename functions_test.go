package main

import (
	"math"
	"testing"
)

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.5},
		{1.0, 0.73105857863},
		{-1.0, 0.26894142137},
		{math.Inf(1), 1.0},
		{math.Inf(-1), 0.0},
	}

	for _, test := range tests {
		result := sigmoid(test.input)
		if math.Abs(result-test.expected) > 1e-9 {
			t.Errorf("For input %f, expected %f, but got %f", test.input, test.expected, result)
		}
	}
}

func TestSigmoidDerivative(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.5, 0.25},
		{0.73105857863, 0.19661193324}, // sigmoid(1.0)
		{0.26894142137, 0.19661193324}, // sigmoid(-1.0)
	}

	for _, test := range tests {
		result := sigmoidDerivative(test.input)
		if math.Abs(result-test.expected) > 1e-9 {
			t.Errorf("For input %f, expected %f, but got %f", test.input, test.expected, result)
		}
	}
}

func TestReLU(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{5.0, 5.0},
		{0.0, 0.0},
		{-5.0, 0.0},
	}

	for _, test := range tests {
		result := relu(test.input)
		if result != test.expected {
			t.Errorf("For input %f, expected %f, but got %f", test.input, test.expected, result)
		}
	}
}

func TestReLUDerivative(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{5.0, 1.0},
		{0.0, 0.0},
		{-5.0, 0.0},
	}

	for _, test := range tests {
		result := reluDerivative(test.input)
		if result != test.expected {
			t.Errorf("For input %f, expected %f, but got %f", test.input, test.expected, result)
		}
	}
}
