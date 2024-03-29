package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type model struct {
	weights *mat.Dense
	bias    *mat.Dense
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func NewModel(r, c int) *model {
	data := make([]float64, r*c*10)
	vector := mat.NewDense(r*c, 10, data)
	bias := mat.NewDense(1, 10, make([]float64, 10))
	return &model{vector, bias}
}

func (m *model) Train(images *mat.Dense, lables *mat.Dense) {
	for i := 0; i < 1000; i++ {
		res := m.Inference(images)
		m.dCost(images, res, lables)
		fmt.Printf("Epoch %v\n", i)
	}
}

func (m *model) dCost(x, p, l *mat.Dense) {
	var diff mat.Dense
	diff.Sub(p, l)

	var gradW mat.Dense
	gradW.Mul(x.T(), &diff)
	gradW.Scale(1e-5/float64(x.RawMatrix().Rows), &gradW)
	var gb [10]float64
	for i := range gb {
		gb[i] = mat.Sum(diff.ColView(i))
	}
	gradB := mat.NewDense(1, 10, gb[:])
	gradB.Scale(1e-5/float64(x.RawMatrix().Rows), gradB)

	// fmt.Println(mat.Formatted(&gradW))

	m.weights.Sub(m.weights, &gradW)
	m.bias.Sub(m.bias, gradB)
}

func (m *model) Inference(inputs *mat.Dense) *mat.Dense {
	var res mat.Dense
	res.Mul(inputs, m.weights)
	res.Apply(func(i, j int, v float64) float64 {
		return v + m.bias.At(0, j)
	}, &res)
	var b mat.Dense
	b.Apply(func(i, j int, v float64) float64 {
		return sigmoid(v)
	}, &res)

	// fmt.Println(res)
	return &b
}

func (m *model) Predict(a *mat.Dense) int {
	b := m.Inference(a)
	data := b.RawMatrix().Data
	maxx := max(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9])
	for i, v := range data {
		if maxx == v {
			return i
		}
	}
	panic(5)
}

func (m *model) Accuracy(testset, testlables *mat.Dense) {
	res := m.Inference(testset)
	good := 0.0
	for i := 0; i < res.RawMatrix().Rows; i++ {
		a := res.RowView(i)
		n := a.Len()
		data := make([]float64, n)
		for i := 0; i < n; i++ {
			data[i] = a.AtVec(i)
		}

		maxx := max(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9])
		for j, v := range data {
			if maxx == v {
				if testlables.At(i, j) == 1 {
					good += 1
				}
			}
		}
	}
	fmt.Print(good / float64(testset.RawMatrix().Rows))
}
