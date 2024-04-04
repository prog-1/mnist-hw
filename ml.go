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
func inference(pixels, weights, biases *mat.Dense) *mat.Dense {
	var predictions mat.Dense
	predictions.Mul(pixels, weights) // (N x 784) * (784 x 10) = (N x 10)
	// predictions.Add(predictions, biases)// (N x 10) + (1 x 10) = panic
	predictions.Apply(func(i, j int, v float64) float64 {
		return v + biases.At(0, j)
	}, &predictions)
	predictions.Apply(sigmoid, &predictions)
	return &predictions
}

// Converts original label/digit, into 10 element array of chances. Same size as prediciton.
// Dimensions: original - 1 x N, converted - N x 10
func convertLabels(original *mat.Dense) (converted *mat.Dense) {
	_, N := original.Dims()
	converted = mat.NewDense(N, digitCount, nil)
	for i := 0; i < N; i++ {
		converted.Set(i, int(original.At(0, i)), 1)
	}
	return converted
}

// Returns gradient of the loss function, i.e. derivatives of all the weights and biases.
// Dimensions(rows x columns): pixels - N x 784, labels - N x 10, predictions - N x 10, dw - 784 x 10, db - 1 x 10
func dCost(pixels, labels, predictions *mat.Dense) (dw, db *mat.Dense) {
	_, pixelCount := pixels.Dims()
	dw, db = mat.NewDense(pixelCount, digitCount, nil), mat.NewDense(1, digitCount, nil)
	// RowCount, ColCount := pixels.Dims()
	imageCount, _ := pixels.Dims()
	diff := mat.NewDense(imageCount, digitCount, nil)

	diff.Sub(predictions, convertLabels(labels))

	dw.Mul(pixels.T(), diff) // dw = Xt * diff -- 784 x 10
	dw.Scale(2/float64(imageCount), dw)

	tmpdb := make([]float64, digitCount)
	for i := range tmpdb {
		tmpdb[i] = mat.Sum(diff.ColView(i))
	}
	db = mat.NewDense(1, digitCount, tmpdb)
	db.Scale(2/float64(imageCount), db)

	return dw, db
}

// Dims(r x c): w - 784 x 10 , dw - 784 x 10, b - 1 x 10, db - 1 x 10
func train(epochCount int, pixels, labels *mat.Dense, lrw, lrb float64,
	sink func(epoch int, w, dw, b, db *mat.Dense)) (w, b *mat.Dense, err error) {
	_, pixelCount := pixels.Dims()
	w = mat.NewDense(pixelCount, digitCount, nil) // w - 784 x 10, initialized with zeroes
	b = mat.NewDense(1, digitCount, nil)
	for epoch := 0; epoch < epochCount; epoch++ {
		dw, db := dCost(pixels, labels, inference(pixels, w, b))

		// Adjusting weights
		db.Scale(lrb, db)
		b.Sub(b, db)

		dw.Scale(lrw, dw)
		w.Sub(w, dw)

		if sink != nil {
			sink(epoch, w, dw, b, db)
		}
	}
	return w, b, nil
}

// xTest - N x 784, yTest - N x 1, w = 784 x 10, b - 1 x 10,
func accuracy(xTest, yTest, w, b *mat.Dense) float64 {
	predictions := inference(xTest, w, b)   // N x 10
	convertedLabels := convertLabels(yTest) // N x 10

	// Converting predicitons into matrix with 1 and 0:
	predictions.Apply(func(i, j int, v float64) float64 {
		if predictions.At(i, j) > 0.5 {
			return 1
		} else {
			return 0
		}
	}, predictions)

	var correctCount float64
	equal := func(a, b, epsilon float64) bool {
		return math.Abs(a-b) < epsilon
	}
	r, c := predictions.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if equal(predictions.At(i, j), convertedLabels.At(i, j), 1e-10) {
				correctCount++
			}
		}
	}
	return float64(r) / correctCount
}
