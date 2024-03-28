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
	imageCount, _ := pixels.Dims()
	diff := mat.NewDense(imageCount, digitCount, nil)
	diff.Sub(predictions, labels)

	dw.Mul(pixels.T(), diff) // dw = Xt * diff -- 784 x 10
	dw.Scale(2/float64(imageCount), dw)

	gradientB := make([]float64, digitCount)
	for i := range gradientB {
		gradientB[i] = mat.Sum(diff.ColView(i))
	}
	db = mat.NewDense(1, digitCount, gradientB)
	db.Scale(2/float64(imageCount), db)

	return dw, db
}
