package main

import (
	"errors"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

const epsilon = 1e-3

func TestInference(t *testing.T) {
	type Input struct {
		x, w, b *mat.Dense
	}
	for n, tc := range []struct {
		input Input
		want  *mat.Dense
	}{
		{
			input: Input{x: mat.NewDense(2, 3, []float64{-1, -5, 0, 0, 0, 0}), w: mat.NewDense(3, 2, []float64{1, 0, 0, 1, 0, 0}), b: mat.NewDense(2, 1, []float64{1, 2})},
			want:  mat.NewDense(2, 2, []float64{0.5, 0.0474, 0.7310, 0.8808}),
		},
	} {
		if got := inference(tc.input.x, tc.input.w, tc.input.b); !mat.EqualApprox(got, tc.want, epsilon) {
			t.Errorf("inference with input No. %v\n Got:\n%v\n\n Want:\n%v\n\n", n+1, mat.Formatted(got), mat.Formatted(tc.want))
		}
	}
}

func TestSigmoid(t *testing.T) {
	for _, tc := range []struct {
		input float64
		want  float64
	}{
		{0, 0.5},
		{1, 1 / (1 + math.Exp(-1))},
		{-1, 1 / (1 + math.Exp(1))},
		{100, 1 / (1 + math.Exp(-100))},
		{-100, 1 / (1 + math.Exp(100))},
	} {
		if got := sigmoid(0, 0, tc.input); math.Abs(got-tc.want) > epsilon {
			t.Errorf("sigmoid(%v) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

func TestConvertLabels(t *testing.T) {
	for n, tc := range []struct {
		input     *mat.Dense
		want      *mat.Dense
		mustPanic bool
	}{
		// Single label
		{
			input:     mat.NewDense(1, 1, []float64{0}),
			want:      mat.NewDense(1, 10, []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
			mustPanic: false,
		},
		// Multiple labels
		{
			input: mat.NewDense(3, 1, []float64{0, 1, 2}),
			want: mat.NewDense(3, 10, []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 1, 0, 0, 0, 0, 0, 0, 0}),
			mustPanic: false,
		},
		// Input dimension mismatch
		{
			input:     mat.NewDense(2, 2, nil),
			want:      nil,
			mustPanic: true,
		},
	} {
		if tc.mustPanic {
			defer func() {
				if r, ok := recover().(error); r != nil && !ok {
					panic(errors.New("panic is not an error"))
				} else if tc.mustPanic == true && r == nil {
					t.Errorf("convertLabels with input No. %v does not panic when it must", n+1)
				} else if tc.mustPanic == false && r != nil {
					t.Errorf("convertLabels with input No. %v panics when it must not", n+1)
				}
			}()
			if got := convertLabels(tc.input); !mat.EqualApprox(got, tc.want, epsilon) {
				t.Errorf("convertLabels with input No. %v\n\n Got:%v\n\n Want:\n%v\n\n", n+1, mat.Formatted(got), mat.Formatted(tc.want))
			}
		}
	}
}

func TestSoftmax(t *testing.T) {
	for n, tc := range []struct {
		input     *mat.Dense
		want      *mat.Dense
		mustPanic bool
	}{
		// Column vector with positive values
		{
			input: mat.NewDense(3, 1, []float64{1.0, 2.0, 3.0}),
			want: mat.NewDense(3, 1, []float64{
				math.Exp(1.0) / (math.Exp(1.0) + math.Exp(2.0) + math.Exp(3.0)),
				math.Exp(2.0) / (math.Exp(1.0) + math.Exp(2.0) + math.Exp(3.0)),
				math.Exp(3.0) / (math.Exp(1.0) + math.Exp(2.0) + math.Exp(3.0)),
			}),
			mustPanic: false,
		},
		// Column vector with negative values
		{
			input: mat.NewDense(3, 1, []float64{-1.0, 0.0, 1.0}),
			want: mat.NewDense(3, 1, []float64{
				math.Exp(-1.0) / (math.Exp(-1.0) + math.Exp(0.0) + math.Exp(1.0)),
				math.Exp(0.0) / (math.Exp(-1.0) + math.Exp(0.0) + math.Exp(1.0)),
				math.Exp(1.0) / (math.Exp(-1.0) + math.Exp(0.0) + math.Exp(1.0)),
			}),
			mustPanic: false,
		},
		// Column vector with all zeroes
		{
			input: mat.NewDense(3, 1, []float64{0, 0, 0}),
			want: mat.NewDense(3, 1, []float64{
				1.0 / 3.0,
				1.0 / 3.0,
				1.0 / 3.0,
			}),
			mustPanic: false,
		},
		// Row vector(should be transposed)
		{
			input: mat.NewDense(1, 3, []float64{1.0, 2.0, 3.0}),
			want: mat.NewDense(3, 1, []float64{
				math.Exp(1.0) / (math.Exp(1.0) + math.Exp(2.0) + math.Exp(3.0)),
				math.Exp(2.0) / (math.Exp(1.0) + math.Exp(2.0) + math.Exp(3.0)),
				math.Exp(3.0) / (math.Exp(1.0) + math.Exp(2.0) + math.Exp(3.0)),
			}),
			mustPanic: false,
		},
		//  Matrix with more than one column (should panic)
		{mat.NewDense(2, 2, nil), nil, true},
	} {
		if tc.mustPanic {
			defer func() {
				if r, ok := recover().(error); r != nil && !ok {
					panic(errors.New("panic is not an error"))
				} else if tc.mustPanic == true && r == nil {
					t.Errorf("softmax with input No. %v does not panic when it must", n+1)
				} else if tc.mustPanic == false && r != nil {
					t.Errorf("softmax with input No. %v panics when it must not", n+1)
				}
			}()
			_ = softmax(tc.input)
		} else {
			if got := softmax(tc.input); !mat.EqualApprox(got, tc.want, epsilon) {
				t.Errorf("softmax with input No. %v\n Got:\n%v\n\n Want:\n%v\n\n", n+1, mat.Formatted(got), mat.Formatted(tc.want))
			}
		}
	}
}
