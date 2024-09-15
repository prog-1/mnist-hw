package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
)

// MNIST Format
// 1. 4 bytes: magic number // 32 bits = size(uint)
// 1.2 2 bytes = 0
// 1.3 Data type 08
// 1.4 dimensions count(2)
// 2. 4 bytes: number of items
// 3. 4 bytes: number of rows
// 4. 4 bytes: number of columns
// 5. n bytes: data

// Returns matrix with all the images or labels from a MNIST file by path
func MnistDataByPath(path string) *mat.Dense {
	f, err1 := os.Open(path)
	if err1 != nil {
		panic(fmt.Errorf("failed to open file %q: %q", path, err1))
	}
	defer f.Close()
	data, err2 := mnistDataFromReader(f)
	if err2 != nil {
		panic(fmt.Errorf("failed to read file %q: %q", path, err2))
	}
	return data
}

// Returns matrix with all the images or labels from a reader with a MNIST file
func mnistDataFromReader(r io.Reader) (*mat.Dense, error) {
	var magic, count uint32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		// Returning error instead of panic, because it is an exported function
		return nil, fmt.Errorf("failed to read magic number: %v", err)
	}
	if magic != 0x803 /*images*/ && magic != 0x801 /*labels*/ {
		return nil, fmt.Errorf("invalid magic number: %x", magic)
	}
	if err := binary.Read(r, binary.BigEndian, &count); err != nil {
		return nil, fmt.Errorf("failed to read element count: %v", err)
	}
	var cols, rows uint32 = 1, 1
	if magic == 0x803 {
		if err := binary.Read(r, binary.BigEndian, &rows); err != nil {
			return nil, fmt.Errorf("failed to read row count: %v", err)
		}
		if err := binary.Read(r, binary.BigEndian, &cols); err != nil {
			return nil, fmt.Errorf("failed to read column count: %v", err)
		}
	}
	data := make([]byte, count*rows*cols)
	_, err := io.ReadFull(r, data)
	if err != nil {
		return nil, fmt.Errorf("failed to read data: %v", err)
	}

	switch magic {
	case 0x803:
		return bytesToMat(count, rows*cols, data), nil
	case 0x801:
		return bytesToMat(1, count, data), nil
	default:
		return nil, fmt.Errorf("invalid magic number: %x", magic)
	}
}

// Returns matrix of the dimensions specified filled with bytes casted to float64
func bytesToMat(rows, cols uint32, input []byte) *mat.Dense {
	output := make([]float64, len(input))
	for i, x := range input {
		output[i] = float64(x)
	}
	return mat.NewDense(int(rows), int(cols), output)
}

// Draws in the console the i'th image from the images matrix.
// Assumes images being stored as rows of pixels.
func PrintMnistImage(i int, images *mat.Dense) {
	cols := images.RawMatrix().Cols
	elementCount := int(math.Sqrt(float64(cols)))

	for j := 0; j < cols; j++ {
		if pixel := images.At(i, j); pixel == 0 {
			fmt.Print(" ")
		} else if pixel < 128 {
			fmt.Print(".")
		} else {
			fmt.Print("#")
		}
		if j%elementCount == 0 {
			fmt.Println()
		}
	}
	fmt.Println()
}
