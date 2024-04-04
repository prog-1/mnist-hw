package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

//rows, columns
// x = {n, size}
// w = {size, outputs}
// b = {1, outputs}// matrix - column, but transplanated
// p = {n, outputs}
// y = {n, outputs} //labels

const (

	//Epochs
	epochs = 10

	//Learning rates
	lrw = 0.0001 //learning rate for weights
	lrb = 0.5    //learning rate for bias
)

//####################################################################################################

// Training for the ml to guess digits from drawings.
// Input: x matrix as training data of drawings, y matrix as the right answers for training data.
// Output: trained weight and bias coefficient matrices.
func regression(x, y *mat.Dense) (w, b *mat.Dense) {

	//Weight matrix
	w = mat.NewDense(size, outputs, make([]float64, size*outputs)) //declaring dense matrix {size,outputs} | not sure about length of slice
	w.Apply(func(i, j int, v float64) float64 { return 1 }, w)     //making each element equal to 1

	//Bias matrix
	b = mat.NewDense(1, outputs, make([]float64, outputs))     //declaring dense matrix {1,outputs}
	b.Apply(func(i, j int, v float64) float64 { return 1 }, b) //making each element equal to 1

	//Training
	for epoch := 1; epoch < epochs; epoch++ { // for each epoch
		w, b = gradientDescent(x, y, w, b) // adjusting all coefficients

		// if epoch % 100 = 1 {

		// }
	}

	return w, b //returning weight and bias matrices
}

// Adjusting coefficients
func gradientDescent(x, y, w, b *mat.Dense) (dw *mat.Dense, db *mat.Dense) {

	p := inference(x, w, b) //getting current predictions

	dw, db = gradients(x, y, p) //getting current gradients

	//Applying learning rates
	dw.Scale(lrw/float64(n), dw) //!!! need to divide lrw on batchsize or rows or on images or something !!!
	dw.Scale(lrb/float64(n), dw) //!!! need to divide lrb on batchsize or rows or something !!!

	//dw.Apply(func(i, j int, v float64) float64 { return v * lrw }, dw) //applying learning rate for w
	//db.Apply(func(i, j int, v float64) float64 { return v * lrb }, db) //applying learning rate for b

	w.Sub(w, dw) //subtracting gradient from w
	b.Sub(b, db) //subtracting gradient from b

	return w, b //returning adjusted coefficients
}

// Caclulating current gradients
func gradients(x, y, p *mat.Dense) (dw, db *mat.Dense) {

	//### Differences ###
	d := mat.NewDense(n, outputs, make([]float64, n*outputs)) //differences (deltas) (errors) matrix {n, outputs}
	d.Sub(p, y)                                               //subtracting labels from predictions to get differences

	//### DW ###
	dw = mat.NewDense(size, outputs, nil)
	dw.Mul(x.T(), d) //from theory: dw = 2/n * Tx * d, where 2/n can be omitted, because it is a constant that does not influence a thing

	//### DB ###
	db = mat.NewDense(1, outputs, nil)

	ones := mat.NewDense(1, n, nil)
	ones.Apply(func(i, j int, v float64) float64 { return 1 }, ones) //making each element equal to 1

	db.Mul(ones, d) //need to multiply matrix of differences (d) on ones to sum all that we have in d

	return dw, db //returning weight and bias coefficient matrices
}

//####################################################################################################

// Prediction of y from w*x + b for all images
// БЕРИ ДЛИННУ DIMS X
func inference(x, w, b *mat.Dense) (p *mat.Dense) {
	p = &mat.Dense{}                                                           //constructing prection matrix of {n, outputs}
	p.Mul(x, w)                                                                // w*x
	p.Apply(func(i, j int, v float64) float64 { return g(v + b.At(0, j)) }, p) //adding bias andapplying sigmoid to each element
	return p                                                                   //returning prediction matrix
}

// Sigmoid function
func g(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -1*z))
}

//####################################################################################################
