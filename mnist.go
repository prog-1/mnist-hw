package main

import (
	"encoding/binary"
	"fmt"
	"io"

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

func MnistDataFromReader(r io.Reader) (*mat.Dense, error) {
	var magic, count uint32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
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
		count *= rows * cols
	}
	data := make([]byte, count)
	n, err := io.ReadFull(r, data)
	if err != nil {
		return nil, fmt.Errorf("failed to read data: %v", err)
	}
	if n != len(data) {
		return nil, fmt.Errorf("read %d bytes, want %d", n, count)
	}

	return bytesToMat(count, rows*cols, data), nil
}

func bytesToMat(rowCount, colCount uint32, input []byte) *mat.Dense {
	output := make([]float64, len(input))
	for i, x := range input {
		output[i] = float64(x)
	}
	return mat.NewDense(int(rowCount), int(colCount), output)
}

func PrintMnistImage(rowCount, colCount uint32, pixels []byte) {
	for r := uint32(0); r < rowCount; r++ {
		for c := uint32(0); c < colCount; c++ {
			if pixel := pixels[r*colCount+c]; pixel == 0 {
				fmt.Print(" ")
			} else if pixel < 128 {
				fmt.Print(".")
			} else {
				fmt.Print("#")
			}
		}
		fmt.Println()
	}
}
