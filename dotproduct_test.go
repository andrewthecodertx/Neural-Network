package main

import (
	"math"
	"testing"
)

func TestDotProduct(t *testing.T) {
	// Static inputs
	inputs := [][]float64{{7.5, 0.5, 0.36, 6.1, 0.071, 17, 102, 0.9978, 3.35, 0.8, 10.5}}

	// Static 4x4 matrix of weights
	weights := [][]float64{
		{0.17120721002462017, 0.029045519465977776, -0.13990289555710322, -0.16137022245108026},
		{-0.3557162316524689, -0.46813359552338696, 1.5233449834028807, 0.21899387304463852},
		{-0.5398276138243183, -0.5218466291269792, -0.16724989153171838, -0.04249847758721853},
		{0.15630301965016696, -0.1500566510323811, -0.49562493343336933, 0.7527480962049542},
		{0.14551271046950953, -0.4054268516877574, -0.3416088366637221, -0.10136968062567701},
		{-0.2749970180986231, 0.6010470444767034, -0.37133697379150354, 0.24310746499012892},
		{-0.321239538192123, 0.3943678899563966, -0.4636771331095591, -0.5870101800895848},
		{0.8030327851743342, -0.49376047610115864, -0.3345177324844342, 0.7332315012315851},
		{-0.4261134932339863, -0.13065546321413235, -0.23910346034779684, -0.7570578784727243},
		{-0.3411841500569469, -0.04972368774799181, -0.39977932852050113, -0.04979135633587868},
		{-0.5764263376771839, -0.2767648364434821, -0.27299439446235496, -0.43985207635513457},
	}

	// Calculate the dot product manually
	expectedResult := manualDotProduct(inputs[0], weights[0]) // Assuming the weights are for the first neuron

	// Calculate the dot product using the dotProduct function
	result := dotProduct(inputs, weights)
	// Compare the results within a tolerance
	if math.Abs(result-expectedResult) > 1e-9 {
		t.Errorf("Test failed. Expected: %v, Got: %v", expectedResult, result)
	}
}

// Function to manually calculate the dot product
func manualDotProduct(inputs []float64, weights []float64) float64 {
	product := 0.0
	for i := range inputs {
		product += inputs[i] * weights[i]
	}
	return product
}
