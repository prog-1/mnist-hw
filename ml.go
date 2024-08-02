package main

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

/*
All vectors are single row matrices, because it is easier to write such in code.
All matrices store vectors as rows, because in gonum/mat matrices are stored in row-major order.
*/

// TODO:
// 1. Vectors as row matrices
// 2. Panic message verification

const (
	digitCount = 10
)

// Creates a handwritten digit recognition machine learning model trained on the input images and labels.
// The x matrix contains pixels. Rows are all pixels of an image and cols are pixels with the same position of all images.
// The matrix called labels is a vector containing the actual digits that is depicted on the image with the same index.
// Dims(rows x cols): x - N x M, labels - 1 x N, where N - image count, M - pixel count for a single image.
// Values w, b, lrw and lrb stand for weights, biases, learning rate weights and learning rate biases.
func Train(epochCount int, x, labels *mat.Dense, lrw, lrb float64) (w, b *mat.Dense, err error) {
	// The reason behind calling the matrix with pixels 'x' and not 'pixels' is for versitality if the inference funciton,
	// which can work not only with a digit recognition model

	// Asserting whether dimensions of the input matrices are appropriate
	imageCount, pixelCount := x.Dims()
	if r, c := labels.Dims(); r != 1 || c != imageCount {
		return nil, nil, fmt.Errorf("labels.Dims() = %v, %v, want = 1, %v", r, c, imageCount)
	}

	//w - M x 10 , dw - M x 10, b - 1 x 10, db - 1 x 10
	w = mat.NewDense(pixelCount, digitCount, nil) // w - M x 10, initialized with zeroes
	b = mat.NewDense(1, digitCount, nil)
	// The decision to use column vectors in justified by the absence of the necessity to transpose it,
	// while performing matrix-vector multiplication.

	for epoch := 0; epoch < epochCount; epoch++ {
		dw, db := dCost(x, labels, inference(x, w, b))

		// Adjusting weights
		db.Scale(lrb, db)
		b.Sub(b, db)

		dw.Scale(lrw, dw)
		w.Sub(w, dw)
	}
	return w, b, nil
}

// Returns 10 probabilities for each image, representing probability of each image being each digit.
// Dims(rows x cols): x - N x M, w - M x 10, b - 1 x 10, predictions - N x 10
func inference(x, w, b *mat.Dense) *mat.Dense {
	var predictions mat.Dense
	predictions.Mul(x, w) // (N x M) * (M x 10) = (N x 10)
	// predictions.Add(predictions, biases)// (N x 10) + (1 x 10) = panic
	predictions.Apply(func(_, j int, v float64) float64 {
		return v + b.At(j, 0)
	}, &predictions)
	predictions.Apply(sigmoid, &predictions)
	return &predictions
}

func sigmoid(_, _ int, v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

// Converts original label/digit, into 10 element array of chances. Same size as prediciton.
// Dimensions: original - 1 x N, converted - N x 10
func convertLabels(original *mat.Dense) (converted *mat.Dense) {
	r, N := original.Dims()
	if r != 1 {
		panic(errors.New("original is not a row vector"))
	}
	converted = mat.NewDense(N, digitCount, nil)
	for i := 0; i < N; i++ {
		converted.Set(i, int(original.At(0, i)), 1)
	}
	return converted
}

// Converts 1 x 10 matrix of chances from 0 or 1 to a digit of the highest chance
func convertPrediction(original *mat.Dense) (converted int) {
	if r, c := original.Dims(); r != 1 || c != digitCount {
		panic(errors.New("prediction is not 1 x 10"))
	}

	var maxChance float64
	original.Apply(func(_, j int, v float64) float64 {
		if v > maxChance {
			converted = j
			maxChance = v
		}
		return v
	}, original)

	return converted
}

// Returns gradient of the loss function, i.e. derivatives of all the weights and biases.
// Dimensions(rows x columns): x - N x 784, labels - N x 1, predictions - N x 10, dw - 784 x 10, db - 10 x 1
func dCost(x, labels, predictions *mat.Dense) (dw, db *mat.Dense) {
	imageCount, pixelCount := x.Dims()
	// Verifying input dimensions
	if lN, l1 := labels.Dims(); lN != imageCount || l1 != 1 {
		panic(errors.New("incorrect dimenions of labels"))
	} else if pN, p10 := predictions.Dims(); pN != imageCount || p10 != digitCount {
		panic(errors.New("incorrect dimenions of predictions"))
	}

	diff := mat.NewDense(imageCount, digitCount, nil) // N x 10
	diff.Sub(predictions, convertLabels(labels))

	dw = mat.NewDense(pixelCount, digitCount, nil)
	dw.Mul(x.T(), diff) // dw = Xt * diff -- (784xN) * (Nx10) = 784 x 10
	dw.Scale(2/float64(imageCount), dw)

	tmpdb := make([]float64, digitCount)
	for i := range tmpdb {
		tmpdb[i] = mat.Sum(diff.ColView(i))
	}
	db = mat.NewDense(digitCount, 1, tmpdb)
	db.Scale(2/float64(imageCount), db)

	return dw, db
}

// x - N x 784, labels - N x 1, w = 784 x 10, b - 1 x 10
func accuracy(x, labels, w, b *mat.Dense) float64 {
	predictions := inference(x, w, b)        // N x 10
	convertedLabels := convertLabels(labels) // N x 10

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

// Returns vector of the full probability destribution for the input vector elements.
// Sum of all elements in probabilities = 1.
// Input vector size - N x 1, probabilities size - N x 1.
// Row input vector will be transposed.
func softmax(vector *mat.Dense) (probabilities *mat.Dense) {
	r, c := vector.Dims()
	if r != 1 {
		// Working only with column vectors
		vector = mat.DenseCopyOf(vector.T())
		r, c = c, r
	}
	if r != 1 {
		// Throwing error instead of string for compatibility
		// with the error-handling code that uses errors package.
		panic(errors.New("softmax argument is not a vector"))
	}

	var denominator float64
	for i := 0; i < c; i++ {
		denominator += math.Exp(vector.At(0, i))
	}

	probabilities = mat.NewDense(1, c, nil)
	probabilities.Apply(
		func(_, i int, v float64) float64 {
			return math.Exp(vector.At(0, i)) / denominator
		}, probabilities)

	return probabilities
}
