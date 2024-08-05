package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMnist(t *testing.T) {
	type Want struct {
		output *mat.Dense
		err    error
	}
	for n, tc := range []struct {
		input io.Reader
		want  Want
	}{
		// Valid image file
		{genMnistMockData(0x803, 1, 1, 1), Want{mat.NewDense(1, 1, []float64{5}), nil}},
		// Valid label file
		{genMnistMockData(0x801, 1, 1, 1), Want{mat.NewDense(1, 1, []float64{5}), nil}},
		// Empty file
		{bytes.NewReader([]byte{}), Want{nil, fmt.Errorf("failed to read magic number: EOF")}},
		// File with an incorrect magic number.
		{genMnistMockData(0x802, 1, 1, 1), Want{nil, fmt.Errorf("invalid magic number: 802")}},
		// Only magic number
		{bytes.NewReader([]byte{0x00, 0x00, 0x08, 0x03}), Want{nil, fmt.Errorf("failed to read element count: EOF")}},
		// Only magic number and count
		{bytes.NewReader([]byte{0x00, 0x00, 0x08, 0x03, 0x00, 0x00, 0x00, 0x01}), Want{nil, fmt.Errorf("failed to read row count: EOF")}},
		// Only magic number, count and rows
		{func() io.Reader {
			mockFile := new(bytes.Buffer)
			writeToFile(mockFile, 0x803)
			writeToFile(mockFile, 1)
			writeToFile(mockFile, 1)
			return mockFile
		}(), Want{nil, fmt.Errorf("failed to read column count: EOF")}},
		// Only no data
		{func() io.Reader {
			mockFile := new(bytes.Buffer)
			writeToFile(mockFile, 0x803)
			writeToFile(mockFile, 1)
			writeToFile(mockFile, 1)
			writeToFile(mockFile, 1)
			return mockFile
		}(), Want{nil, fmt.Errorf("failed to read data: EOF")}},
		// Incomplete data
		{func() io.Reader {
			mockFile := new(bytes.Buffer)
			writeToFile(mockFile, 0x801)
			writeToFile(mockFile, 2)
			if _, err := mockFile.Write([]byte{byte(1)}); err != nil {
				panic(err)
			}
			return mockFile
		}(), Want{nil, fmt.Errorf("failed to read data: unexpected EOF")}},
	} {
		if got, err := mnistDataFromReader(tc.input); err != nil {
			if tc.want.err == nil || err.Error() != tc.want.err.Error() {
				t.Errorf("mnistDataFromReader(tc%v.input) error = %v, want error = %v", n, err, tc.want.err)
			}
		} else if tc.want.err != nil {
			t.Errorf("mnistDataFromReader(tc%v.input) expected error, got none", n)
		} else if !mat.Equal(tc.want.output, got) {
			t.Errorf("mnistDataFromReader(tc%v.input) = %v, want %v", n, got, tc.want)
		}
	}
}

func writeToFile(file io.Writer, value uint32) {
	if err := binary.Write(file, binary.BigEndian, value); err != nil {
		panic(err)
	}
}

func genMnistMockData(magic, count, rows, cols uint32) io.Reader {
	mockFile := new(bytes.Buffer)

	writeToFile(mockFile, magic)
	writeToFile(mockFile, count)

	if magic == 0x803 /*images*/ {
		writeToFile(mockFile, rows)
		writeToFile(mockFile, cols)
	}

	data := make([]byte, count*rows*cols)
	for i := range data {
		// Any value will do, we only check for count
		data[i] = byte(5)
	}
	if _, err := mockFile.Write(data); err != nil {
		panic(err)
	}

	return mockFile
}
