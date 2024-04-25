package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

/*
	x = {n, size}
	y = {n, outputs} //labels

	w1 = {size, outputs}
	b1 = {1, outputs}
	h = {n, outputs}

	w2 = {size, outputs}
	b2 = {1, size}
	z = {n, size}

	E = scalar
*/

const (

	//Epochs
	epochs = 10

	//Learning rates
	lrw = 1 //0.1     //learning rate for weights
	lrb = 1 //0.00001 //learning rate for bias
)

//####################################################################################################

// Training for the ml to guess digits from drawings.
// Input: x matrix as training data of drawings, y matrix as the right answers for training data.
// Output: trained weight and bias coefficient matrices.
func regression(x, y *mat.Dense, n int) (w1, b1, w2, b2 *mat.Dense) {

	//### Coefficient declaration ###

	source := rand.New(rand.NewSource(100)) //for parameter initial value randomization in range from 0 to 100

	//1st layer

	//Weight matrix
	w1 = mat.NewDense(size, outputs, nil)                                                  //w1 = {size,outputs} = {784, 10}
	w1.Apply(func(i, j int, v float64) float64 { return float64(source.Int31n(100)) }, w1) //setting random numbers from 1 to 100

	//Bias matrix
	b1 = mat.NewDense(1, outputs, nil)                                                     //b1 = {1,outputs} = {1,10}
	b1.Apply(func(i, j int, v float64) float64 { return float64(source.Int31n(100)) }, b1) //setting random numbers from 1 to 100

	//2nd layer

	//Weight matrix
	w2 = mat.NewDense(size, outputs, nil)                                                  //w2 = {size,outputs} = {784, 10}
	w2.Apply(func(i, j int, v float64) float64 { return float64(source.Int31n(100)) }, w2) //setting random numbers from 1 to 100

	//Bias matrix
	b2 = mat.NewDense(1, outputs, nil)                                                     //b2 = {1,outputs} = {1,10}
	b2.Apply(func(i, j int, v float64) float64 { return float64(source.Int31n(100)) }, b2) //setting random numbers from 1 to 100

	//### Training ###

	for epoch := 1; epoch < epochs; epoch++ { // for each epoch

		w1, b1, w2, b2 = gradientDescent(x, y, w1, b1, w2, b2) // adjusting all coefficients

		if epoch%10 == 0 {
			fmt.Println("Epoch:", epoch)
		}
	}

	return w1, b1, w2, b2 //returning weight and bias matrices
}

// Adjusting coefficients
func gradientDescent(x, y, w1, b1, w2, b2 *mat.Dense) (dw1, db1, dw2, db2 *mat.Dense) {

	h, z := inference(x, w1, b1, w2, b2) //getting current predictions (end result what to calculate E from)

	dw1, db1, dw2, db2 = gradients(x, y, z, h, w2) //calculating current gradients

	//fmt.Println("DB:", mat.Formatted(db))

	//Substraction
	//(all gradients are already scaled on learning rate in "gradients" function)

	//1st layer
	w1.Sub(w1, dw1) //w1 - dw1 | {784, 10}
	b1.Sub(b1, db1) //b1 - db1 | {1, 10}

	//2nd layer
	w2.Sub(w2, dw2) //w2 - dw2 | {784, 10}
	b2.Sub(b2, db2) //b2 - db2 | {1, 10}

	return dw1, db1, dw2, db2 //returning adjusted coefficients
}

// Caclulating current gradients (dCost)
func gradients(x, y, z, h, w2 *mat.Dense) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {

	//### Gradient Calculation ###

	//### dE_dt2 ###
	dE_dt2 := &mat.Dense{}
	dE_dt2.Sub(z, y) // dE_dt2 = {60000,10}

	//### dE_dw2 ###
	dE_dw2 := &mat.Dense{}
	dE_dw2.Mul(h.T(), dE_dt2) // dE_dw2 = h.T * dE_dt2 = {10,60000} * {60000,10} = {10, 10}  [??? But should be {784,10} ???]

	//### dE_db2 ###
	dE_db2 := dE_dt2 // dE_db2 = {60000,10} [??? But should be {1,10} ???]

	//### dE_dh ###
	dE_dh := &mat.Dense{}
	dE_dh.Mul(dE_dt2, w2.T()) //dE_dh = {60000,10} * {10,784} = {60000,784} [??? But should be {60000,10} ???]

	//### dE_dt1 ###
	dE_dt1 := &mat.Dense{}
	dE_dt1.MulElem(dE_dh, h) // dE_dt1 = {60000,784} * {60000,10} | [??? dE_dh should be {60000,10} ???]

	//### dE_dw1 ###
	dE_dw1 := &mat.Dense{}
	dE_dw1.Mul(x.T(), dE_dt1) // dE_dw1 = {784,60000} * {60000,784} | [??? dE_dh should be {60000,10} ???]

	//### dE_db1 ###
	dE_db1 := dE_dt1 // dE_db1 = {60000,10} | [??? But should be {1,10} ???]

	//### Applying learning rates ###
	//(previously there was diffision on n (image count))
	dE_dw1.Scale(lrw, dE_dw1)
	dE_db1.Scale(lrb, dE_db1)
	dE_dw2.Scale(lrw, dE_dw2)
	dE_db2.Scale(lrb, dE_db2)

	return dE_dw1, dE_db1, dE_dw2, dE_db2 //returning weight and bias coefficient matrices
}

//####################################################################################################

// Prediction of y from w*x + b for all images
func inference(x, w1, b1, w2, b2 *mat.Dense) (h, z *mat.Dense) {

	//x -> t1 -> h1
	h = &mat.Dense{}
	h.Mul(x, w1)                                                                      // t1 = w1*x | 60000x784 * 784x10 = 60000x10
	h.Apply(func(i, j int, v float64) float64 { return sigmoid(v + b1.At(0, j)) }, h) //h = g(w1*x + b1) | 60000x10

	//h1/x2 -> t2 -> z
	t2 := &mat.Dense{}
	t2.Mul(h, w2.T()) // t2 = w2.T*h | 60000x10 * 10x784 = 60000x784
	//### Converting 60000x784 into 60000x10 ###
	ones := mat.NewDense(size, outputs, nil) // 784x10 of ones
	ones.Apply(func(i, j int, v float64) float64 { return 1 }, ones)
	z = &mat.Dense{}
	z.Mul(t2, ones) //t2: 60000x784 * 784x10 = 60000x10
	//###
	z.Apply(func(i, j int, v float64) float64 { return (v + b2.At(0, j)) }, z) // t2 = w2*h + b2 | 60000x10
	z = softmax(z)                                                             // z = sm(w2*h + b2)

	return h, z
}

// Sigmoid function
func sigmoid(z float64) float64 {
	//return 1 / (1 + math.Pow(math.E, -1*z))
	return 1.0 / (1.0 + math.Exp(-z))
}

// Softmax function
func softmax(t *mat.Dense) *mat.Dense {

	//Calculating sum of e^ti to not do it every iteration
	var sum float64
	rows, columns := t.Dims()
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			sum += math.Exp(t.At(row, column))
		}
	}

	//Calculating softmax for each parameter
	t.Apply(func(i, j int, v float64) float64 { return (math.Exp(t.At(i, j))) / sum }, t)

	return t
}

//####################################################################################################

func accuracy(x, y, w1, b1, w2, b2 *mat.Dense) (a int) {
	_, p := inference(x, w1, b1, w2, w1)
	for i := 0; i < epochs; i++ {
		var i1, i2 int
		for j := 0; j < outputs; j++ {
			if p.At(i, j) > p.At(i, i1) {
				i1 = j
			}
		}
		for j := 0; j < outputs; j++ {
			if y.At(i, j) > y.At(i, i1) {
				i2 = j
			}
		}
		if i1 == i2 {
			a++
		}
	}

	//return float64(a) / 10000
	return a
}
