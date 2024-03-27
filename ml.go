package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

const (
	digitCount = 10
)

func sigmoid(_, _ int, v float64) float64 {
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

// Converts original label/digit, into 10 element array of chances. Same size as prediciton.
// Dimensions: original - N x 1, converted - N x 10
func convertLabels(original *mat.Dense) (converted *mat.Dense) {
	N, _ := original.Dims()
	converted = mat.NewDense(N, digitCount, nil)
	for i := 0; i < N; i++ {
		converted.Set(i, int(original.At(i, 1)), 1)
	}
	return converted
}

// Returns gradient of the loss function, i.e. derivatives of all the weights and biases.
// Dimensions(rows x columns): pixels - N x 784, labels - N x 10, predictions - N x 10, dw - 1 x 784, db - 1 x 10
func dCost(pixels, labels, predictions *mat.Dense) (dw, db *mat.Dense) {
	// RowCount, ColCount := pixels.Dims()
	N, PC := pixels.Dims() // PC - pixel count
	diff := mat.NewDense(N, digitCount, nil)
	diff.Sub(labels, predictions)
	// for c := 0; c < ColCount; c++ {// Parameters/Pixels
	// 	for r := 0; r < RowCount; r++ {// Images
	// 	}
	// }
	dw = mat.NewDense(1, PC, nil)
	db = mat.NewDense(1, digitCount, nil)
	dw.Apply(func(i, j int, v float64) float64 {
		dw.At(i, j) += 2 / N * diff.At(i)
	}, dw)
}
