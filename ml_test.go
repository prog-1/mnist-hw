package main

import (
	"errors"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

const epsilon = 1e-6

func TestSoftmax(t *testing.T) {
	for _, tc := range []struct {
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
					t.Errorf("softmax(%v) does not panic when it must", tc.input)
				} else if tc.mustPanic == false && r != nil {
					t.Errorf("softmax(%v) panics when it must not", tc.input)
				}
			}()
			_ = softmax(tc.input)
		} else {
			if got := softmax(tc.input); !mat.EqualApprox(got, tc.want, epsilon) {
				t.Errorf("softmax(%v) = %v, want %v", tc.input.RawMatrix().Data, got.RawMatrix().Data, tc.want.RawMatrix().Data)
			}
		}
	}
}
