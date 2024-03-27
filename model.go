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
	for i := 0; i < 100; i++ {
		res := m.Inference(images)
		m.dCost(images, res, lables)
		fmt.Printf("Epoch %v", i)
	}
}

func (m *model) dCost(inputs, p, l *mat.Dense) {
	var diff mat.Dense
	diff.Sub(p, l)
	tmp := inputs.T()
	// fmt.Println(tmp.Dims())
	// fmt.Println(inputs.Dims())
	// fmt.Println(diff.Dims())
	var a mat.Dense
	a.Mul(tmp, &diff)
	var b mat.Dense
	b.Scale((2/float64(inputs.RawMatrix().Rows))*1e-5, &a)
	// fmt.Println(b)
	var c mat.Dense
	fmt.Println(m.weights.Dims())
	fmt.Println(diff.Dims())
	c.Sub(m.weights, &b)
	m.weights = &c
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
