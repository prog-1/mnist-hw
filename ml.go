package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func sigmoid(i, j int, v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

// Returns 10 probabilities for each image, representing probability of each image being each digit.
// Dimensions(rows x columns): pixels - N x 784, weights - 784 x 10, biases - 1 x 10, predictions - N x 10.
// N - image count.
func inference(pixels, weights, biases *mat.Dense) (predictions *mat.Dense) {
	predictions.Mul(pixels, weights) // (N x 784) * (784 x 10) = (N x 10)
	// predictions.Add(predictions, biases)// (N x 10) + (1 x 10) = panic
	predictions.Apply(func(i, j int, v float64) float64 {
		return v + biases.At(1, j)
	}, predictions)
	predictions.Apply(sigmoid, predictions)
	return predictions
}

// Returns gradient of the loss function, i.e. derivatives of all the weights and biases.
// Dimensions(rows x columns): pixels - N x 784, dw - 784 x 10, db -
func dCost(pixels *mat.Dense) (dw, db *mat.Dense) {

}
