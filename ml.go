package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// x = {n, size}
// w = {size, digits}
// b = {size, 1}
// p = {n, digits}

const (
	epochs = 10000
	lrw    = 0.000001
	lrb    = 5
)

//####################################################################################################

// Training for the ml to guess digits from drawings.
// Input: x matrix as training data of drawings, y matrix as the right answers for training data.
// Output: trained weight and bias coefficient matrices.
func regression(x, y *mat.Dense) (w, b *mat.Dense) {

	//Weight matrix
	w = mat.NewDense(size, digits, make([]float64, size*digits)) //not sure about length of slice

	//Bias matrix
	b = mat.NewDense(size, 1, make([]float64, size))
	b.Apply(func(i, j int, v float64) float64 { return 1 }, b) //making each element equal to 1

	//Training
	for epoch := 1; epoch < epochs; epoch++ { // for each epoch
		w, b = gradientDescent(x, y, w, b) // adjusting all coefficients
	}

	return w, b
}

// Adjusting coefficients
func gradientDescent(x, y, w, b *mat.Dense) (*mat.Dense, *mat.Dense) {

	p := inference(x, w, b) //getting current predictions

	dw, db := gradients(x, y, p)                                       //getting current gradients
	dw.Apply(func(i, j int, v float64) float64 { return v * lrw }, dw) //applying learning rate for w
	db.Apply(func(i, j int, v float64) float64 { return v * lrb }, db) //applying learning rate for b

	w.Sub(w, dw) //subtracting gradient from w
	b.Sub(b, db) //subtracting gradient from b

	return w, b //returning adjusted coefficients
}

func gradients(x, y, p *mat.Dense) (dw, db *mat.Dense) {

	var d *mat.Dense //deltas (errors)
	d.Sub(p, y)
	dw = d

	for image := 0; image < rows; image++ { //for each image
		for pixel := 0; pixel < columns; pixel++ { //for each pixel

			dw.Apply(func(i, j int, v float64) float64 { return v + (float64(1/n) * v * x.At(image, pixel)) }, dw)

		}

		db.Apply(func(i, j int, v float64) float64 { return v + (float64(1/n) * v) }, db)
	}

	return dw, db
}

//####################################################################################################

// Prediction of y from w*x + b for all images
func inference(x, w, b *mat.Dense) (p *mat.Dense) {
	p.Mul(x, w)                                                   // w*x
	p.Add(p, b)                                                   // + b
	p.Apply(func(i, j int, v float64) float64 { return g(v) }, p) //applying sigmoid to each element
	return p
}

// Sigmoid function
func g(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -1*z))
}

//####################################################################################################
