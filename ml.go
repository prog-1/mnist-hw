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
func regression(x, y *mat.Dense) (w1, b1, w2, b2 *mat.Dense) {

	n, _ := x.Dims() //getting image count

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
	w2 = mat.NewDense(outputs, outputs, nil)                                               //w2 = {outputs,outputs} = {10, 10}// {2nd layer input count, 2nd layer output count}
	w2.Apply(func(i, j int, v float64) float64 { return float64(source.Int31n(100)) }, w2) //setting random numbers from 1 to 100

	//Bias matrix
	b2 = mat.NewDense(1, outputs, nil)                                                     //b2 = {1,outputs} = {1,10}
	b2.Apply(func(i, j int, v float64) float64 { return float64(source.Int31n(100)) }, b2) //setting random numbers from 1 to 100

	//### Training ###

	for epoch := 1; epoch <= epochs; epoch++ { // for each epoch

		for image := 0; image < n; image++ { // for each image

			// Taking i'th row of x matrix
			xis := mat.Row(nil, image, x)    //saving slice
			xi := mat.NewDense(1, size, xis) //creating separate matrix

			// Taking i'th row of y matrix
			yis := mat.Row(nil, image, y)       //saving slice
			yi := mat.NewDense(1, outputs, yis) //creating separate matrix

			// Adjusting all coefficients
			w1, b1, w2, b2 = gradientDescent(xi, yi, w1, b1, w2, b2)
		}

		//if epoch%10 == 0 {
		fmt.Print("Epoch:", epoch, " | ")

		//Accuracy
		_, z := inference(x, w1, b1, w2, b2)         //getting answers on test dataset
		fmt.Print("Accuracy:", accuracy(z, y), "\n") //printing accuracy of the trained model

		//}

		inputShuffle(x, y, n, source) //Shuffling input and label rows
	}

	return w1, b1, w2, b2 //returning weight and bias matrices
}

// Adjusting coefficients
func gradientDescent(x, y, w1, b1, w2, b2 *mat.Dense) (dw1, db1, dw2, db2 *mat.Dense) {

	// ### ➡️ Forward Propagation ###

	h, z := inference(x, w1, b1, w2, b2) //getting current predictions (end result what to calculate E from)

	// ### ⬅️ Back Propagation ###

	dw1, db1, dw2, db2 = gradients(x, y, z, h, w2) //calculating current gradients

	//Gradient adjustment
	//(all gradients are already scaled on learning rate in "gradients" function)

	//1st layer
	w1.Sub(w1, dw1) //w1 - dw1 | {784, 10}
	b1.Sub(b1, db1) //b1 - db1 | {1, 10}

	//2nd layer
	w2.Sub(w2, dw2) //w2 - dw2 | {10, 10}
	b2.Sub(b2, db2) //b2 - db2 | {1, 10}

	//###

	return w1, b1, w2, b2 //returning adjusted coefficients
}

//####################################################################################################

// Prediction of y from w*x + b for all images
func inference(x, w1, b1, w2, b2 *mat.Dense) (h, z *mat.Dense) {

	//x = {1,784} | w1 = {784,10} | b1 = {1,10} | w2 = {10,10} | b2 = {1,10}

	//x -> t1 -> h1
	h = &mat.Dense{}
	h.Mul(x, w1)                                                                      // t1 = x*w1 | 1x784 * 784x10 = 1x10
	h.Apply(func(i, j int, v float64) float64 { return sigmoid(v + b1.At(0, j)) }, h) // h = g(w1*x + b1) | 1x10

	//h1/x2 -> t2 -> z
	t2 := &mat.Dense{}
	t2.Mul(h, w2)                                                                // t2 = h*w2 = 1x10 * 10x10 = 1x10
	t2.Apply(func(i, j int, v float64) float64 { return (v + b2.At(0, j)) }, t2) // t2 + b2 = 1x10 + 1x10 = 1x10
	z = softmax(t2)                                                              // z = sm(w2*h + b2) = 1x10

	return h, z
}

// Sigmoid function
func sigmoid(z float64) float64 {
	//return 1 / (1 + math.Pow(math.E, -1*z))
	return 1.0 / (1.0 + math.Exp(-z))
}

// Sigmoid function derivative (takes sigmoid result as input)
func dSigmoid(g float64) float64 {
	return g * (1 - g)
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

// Caclulating current gradients (dCost)
func gradients(x, y, z, h, w2 *mat.Dense) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {

	// x = {1, 784} | y = {1, 10} | z = {1, 10} | h = {1, 10} | w2 = {10, 10}

	//### Gradient Calculation ###

	//### dE_dt2 ###
	dE_dt2 := &mat.Dense{}
	dE_dt2.Sub(z, y) // dE_dt2 = {1,10} - {1,10} = {1,10}

	//### dE_dw2 ###
	dE_dw2 := &mat.Dense{}
	dE_dw2.Mul(h.T(), dE_dt2) // dE_dw2 = h.T * dE_dt2 = {10,1} * {1,10} = {10, 10}

	//### dE_db2 ###
	dE_db2 := dE_dt2 // dE_db2 = {1,10}

	//### dE_dh ###
	dE_dh := &mat.Dense{}
	dE_dh.Mul(dE_dt2, w2) // dE_dh = {1,10} * {10,10} = {1,10}

	//### dE_dt1 ###
	h.Apply(func(i, j int, v float64) float64 { return dSigmoid(v) }, h) //applying derivative to each h
	dE_dt1 := &mat.Dense{}
	dE_dt1.MulElem(dE_dh, h) // dE_dt1 = dE_dh o F'(h) = {1,10} o {1,10} = {1,10}

	//### dE_dw1 ###
	dE_dw1 := &mat.Dense{}
	dE_dw1.Mul(x.T(), dE_dt1) // dE_dw1 = {784,1} * {1,10} = {784, 10}

	//### dE_db1 ###
	dE_db1 := dE_dt1 // dE_db1 = {1,10}

	//### Applying learning rates ###
	//(previously there was diffision on n (image count))
	dE_dw1.Scale(lrw, dE_dw1)
	dE_db1.Scale(lrb, dE_db1)
	dE_dw2.Scale(lrw, dE_dw2)
	dE_db2.Scale(lrb, dE_db2)

	return dE_dw1, dE_db1, dE_dw2, dE_db2 //returning weight and bias coefficient matrices
}

//####################################################################################################

func accuracy(p, y *mat.Dense) (a float64) {
	n, _ := p.Dims() //getting image count

	for i := 0; i < n; i++ { //for each image

		ra := 0 //right answer
		for ; y.At(i, ra) != 1; ra++ {
		} //getting right answer index from label matrix

		if p.At(i, ra) < (0.5 / float64(n)) { //if the prediction of the gotten index is more than half
			a++
		}
	}
	return a / float64(n) // answer divided on image count
}

//####################################################################################################

// row shuffle for input and label matrices
func inputShuffle(x, y *mat.Dense, n int, source *rand.Rand) {

	perm := source.Perm(n) //permutation of rows

	Xcopy := mat.DenseCopyOf(x) //copy of x
	Ycopy := mat.DenseCopyOf(y) //copy of y

	for i, j := range perm { //for each permutation iteration

		//x matrix
		Xrow := Xcopy.RowView(j).(*mat.VecDense) // copy row
		x.SetRow(i, Xrow.RawVector().Data)       // set it to new place

		//y matrix
		Yrow := Ycopy.RowView(j).(*mat.VecDense) // copy row
		y.SetRow(i, Yrow.RawVector().Data)       // set it to new place
	}

}
