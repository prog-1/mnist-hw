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
			input: mat.NewDense(1, 3, []float64{0, 1, 2}),
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
			_ = convertLabels(tc.input)
		} else {
			if got := convertLabels(tc.input); !mat.EqualApprox(got, tc.want, epsilon) {
				t.Errorf("convertLabels with input No. %v\n\n Got:\n%v\n\n Want:\n%v\n\n", n+1, mat.Formatted(got), mat.Formatted(tc.want))
			}
		}
	}
}

func TestConvertPrediction(t *testing.T) {
	for n, tc := range []struct {
		input     *mat.Dense
		want      int
		mustPanic bool
	}{
		// Single max
		{
			input:     mat.NewDense(1, 10, []float64{0.05, 0.95, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}),
			want:      1,
			mustPanic: false,
		},
		// Multiple max
		{
			input:     mat.NewDense(1, 10, []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}),
			want:      0,
			mustPanic: false,
		},
		// r != 10
		{
			input:     mat.NewDense(2, 2, nil),
			want:      -1,
			mustPanic: true,
		},
		// c != 1
		{
			input:     mat.NewDense(2, 10, nil),
			want:      -1,
			mustPanic: true,
		},
	} {
		if tc.mustPanic {
			defer func() {
				if r, ok := recover().(error); r != nil && !ok {
					panic(errors.New("panic is not an error"))
				} else if tc.mustPanic == true && r == nil {
					t.Errorf("convetPrediction with input No. %v does not panic when it must", n+1)
				} else if tc.mustPanic == false && r != nil {
					t.Errorf("convertPrediction with input No. %v panics when it must not", n+1)
				}
			}()
			_ = convertPrediction(tc.input)
		} else {
			if got := convertPrediction(tc.input); got != tc.want {
				t.Errorf("convertPrediction with input No. %v = %v, want %v", n+1, got, tc.want)
			}
		}
	}
}

func TestDCost(t *testing.T) {
	type Input struct {
		// x - N x M, labels - N x 1, predictions - N x 10
		x, labels, predictions *mat.Dense
	}
	type Result struct {
		// dw - N x 10, db - 10 x 1
		dw, db *mat.Dense
	}
	for n, tc := range []struct {
		input        Input
		want         Result
		panicMessage string // "" if no panic
	}{
		// 1. Single item. Fully correct prediction.
		{
			input: Input{
				x:           mat.NewDense(1, 1, []float64{1}),
				labels:      mat.NewDense(1, 1, []float64{1}),
				predictions: mat.NewDense(1, 10, []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0}),
			},
			want: Result{
				dw: mat.NewDense(1, 10, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
				db: mat.NewDense(10, 1, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
			},
		},
		// 2. Single item. Fully incorrect prediction.
		{
			input: Input{
				x:           mat.NewDense(1, 1, []float64{1}),
				labels:      mat.NewDense(1, 1, []float64{0}),
				predictions: mat.NewDense(1, 10, []float64{0, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
			},
			// diff(1 x 10) = predictions - convertLabels(labels) =
			// = (1 x 10) - (1 x 10) =
			// = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1} - {1, 0, 0, 0, 0, 0, 0, 0, 0, 0} =
			// = {-1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
			//
			// dw(1 x 10) = x.T() * diff = (1 x 1) * (1 x 10) = (1 x 10) =
			// = {1} * {-1, 1, 1, 1, 1, 1, 1, 1, 1, 1} =
			// = {-1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
			//
			// 2\1 * {-1, 1, 1, 1, 1, 1, 1, 1, 1, 1} =
			// = {-2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
			//
			//
			// db - 10 x 1
			// db[0] = diff[0] = 0
			// db(10 x 1) = {-1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, but vertical'
			// 2/1 * {-1, 1, 1, 1, 1, 1, 1, 1, 1, 1} =
			// = {-2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
			want: Result{
				dw: mat.NewDense(1, 10, []float64{-2, 2, 2, 2, 2, 2, 2, 2, 2, 2}),
				db: mat.NewDense(10, 1, []float64{-2, 2, 2, 2, 2, 2, 2, 2, 2, 2}),
			},
		},
		// 3. xN != labelsN -> panic
		{
			input: Input{
				x:           mat.NewDense(2, 1, nil),
				labels:      mat.NewDense(1, 1, nil),
				predictions: mat.NewDense(1, 10, nil),
			},
			panicMessage: "incorrect dimenions of labels",
		},
		// 4. labels1 !+ 1 -> panic
		{
			input: Input{
				x:      mat.NewDense(1, 1, nil),
				labels: mat.NewDense(1, 2, nil),
			},
			panicMessage: "incorrect dimenions of labels",
		},
		// 5. labels1 != 1 -> panic
		{
			input: Input{
				x:      mat.NewDense(1, 1, nil),
				labels: mat.NewDense(1, 2, nil),
			},
			panicMessage: "incorrect dimenions of labels",
		},
		// 6. predictionsN != N -> panic
		{
			input: Input{
				x:           mat.NewDense(1, 1, nil),
				labels:      mat.NewDense(1, 1, nil),
				predictions: mat.NewDense(1, 1, nil),
			},
			panicMessage: "incorrect dimenions of predictions",
		},
		// 7. predictions10 != 10 -> panic
		{
			input: Input{
				x:           mat.NewDense(1, 1, nil),
				labels:      mat.NewDense(1, 1, nil),
				predictions: mat.NewDense(1, 2, nil),
			},
			panicMessage: "incorrect dimenions of predictions",
		},
	} {
		if tc.panicMessage != "" { // TODO: Figure out why tests are only working under this if case
			defer func() {
				if err, ok := recover().(error); err != nil && !ok {
					panic(errors.New("panic is not an error"))
				} else if tc.panicMessage != "" && err == nil {
					t.Errorf("dCost(input %v) does not panic when it must panic with \"%v\"", n+1, tc.panicMessage)
				} else if tc.panicMessage == "" && err != nil {
					t.Errorf("dCost(input %v), when it must not panic, does so with \"%v\"", n+1, err)
				} else if err.Error() != tc.panicMessage {
					t.Errorf("dCost(input %v) panic message is \"%v\", want \"%v\"", n+1, err, tc.panicMessage)
				}
			}()
			_, _ = dCost(tc.input.x, tc.input.labels, tc.input.predictions)
		} else {
			var got Result
			got.dw, got.db = dCost(tc.input.x, tc.input.labels, tc.input.predictions)
			if !mat.EqualApprox(got.dw, tc.want.dw, epsilon) {
				t.Errorf("dCost(input %v).dw\n\n Got:\n%v\n\n Want:\n%v\n\n", n+1, mat.Formatted(got.dw), mat.Formatted(tc.want.dw))
			} else if !mat.EqualApprox(got.db, tc.want.db, epsilon) {
				t.Errorf("dCost(input %v).db\n\n Got:\n%v\n\n Want:\n%v\n\n", n+1, mat.Formatted(got.db), mat.Formatted(tc.want.db))
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
			input: mat.NewDense(1, 3, []float64{1.0, 2.0, 3.0}),
			want: mat.NewDense(1, 3, []float64{
				math.Exp(1.0) / (math.Exp(1.0) + math.Exp(2.0) + math.Exp(3.0)),
				math.Exp(2.0) / (math.Exp(1.0) + math.Exp(2.0) + math.Exp(3.0)),
				math.Exp(3.0) / (math.Exp(1.0) + math.Exp(2.0) + math.Exp(3.0)),
			}),
			mustPanic: false,
		},
		// Column vector with negative values
		{
			input: mat.NewDense(1, 3, []float64{-1.0, 0.0, 1.0}),
			want: mat.NewDense(1, 3, []float64{
				math.Exp(-1.0) / (math.Exp(-1.0) + math.Exp(0.0) + math.Exp(1.0)),
				math.Exp(0.0) / (math.Exp(-1.0) + math.Exp(0.0) + math.Exp(1.0)),
				math.Exp(1.0) / (math.Exp(-1.0) + math.Exp(0.0) + math.Exp(1.0)),
			}),
			mustPanic: false,
		},
		// Column vector with all zeroes
		{
			input: mat.NewDense(1, 3, []float64{0, 0, 0}),
			want: mat.NewDense(1, 3, []float64{
				1.0 / 3.0,
				1.0 / 3.0,
				1.0 / 3.0,
			}),
			mustPanic: false,
		},
		// Row vector(should be transposed)
		{
			input: mat.NewDense(1, 3, []float64{1.0, 2.0, 3.0}),
			want: mat.NewDense(1, 3, []float64{
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
