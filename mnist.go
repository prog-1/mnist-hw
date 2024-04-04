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

// 3 uint:
// 2. 4 bytes: number of items
// 3. 4 bytes: number of rows
// 4. 4 bytes: number of columns
// 5. n bytes: data

type mnistImages struct {
	Count      uint32
	Rows, Cols uint32
	Pixels     *mat.Dense // Rows x Cols
}

type mnistLabels struct {
	Count  uint32
	Labels *mat.Dense
}

// TODO: Combine 2 Read functions into 1
func ReadMnistImages(r io.Reader) (*mnistImages, error) {
	var magic, count, rows, cols uint32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic number: %v", err)
	}
	if magic != 0x803 { // images
		return nil, fmt.Errorf("magic number is not for images: %x", magic)
	}
	if err := binary.Read(r, binary.BigEndian, &count); err != nil {
		return nil, fmt.Errorf("failed to read image count: %v", err)
	}
	if err := binary.Read(r, binary.BigEndian, &rows); err != nil {
		return nil, fmt.Errorf("failed to read row count: %v", err)
	}
	if err := binary.Read(r, binary.BigEndian, &cols); err != nil {
		return nil, fmt.Errorf("failed to read column count: %v", err)
	}

	// pixels := mat.NewDense(imageCount, rows*cols, nil)
	pixels := make([]byte, count*rows*cols)
	n, err := io.ReadFull(r, pixels) // n - number of bytes
	if err != nil {
		panic(fmt.Sprintf("failed to read images: %v", err))
	}
	if n != len(pixels) {
		panic(fmt.Sprintf("read %d bytes; want %d", n, len(pixels)))
	}

	return &mnistImages{count, rows, cols, bytesToMat(count, rows*cols, pixels)}, nil
}

func ReadMnistLabels(r io.Reader) (*mnistLabels, error) {
	var magic, count uint32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic number: %v", err)
	}
	if magic != 0x801 { // labels
		return nil, fmt.Errorf("magic number is not for labels: %x", magic)
	}
	if err := binary.Read(r, binary.BigEndian, &count); err != nil {
		return nil, fmt.Errorf("failed to read image count: %v", err)
	}

	labels := make([]byte, count)
	n, err := io.ReadFull(r, labels)
	if err != nil {
		panic(fmt.Sprintf("failed to read labels: %v", err))
	}
	if n != len(labels) {
		panic(fmt.Sprintf("read %d bytes; want %d", n, len(labels)))
	}
	return &mnistLabels{count, bytesToMat(1, count, labels)}, nil
}

func bytesToMat(rowCount, colCount uint32, input []byte) *mat.Dense {
	output := make([]float64, len(input))
	for i, x := range input {
		output[i] = float64(x)
	}
	return mat.NewDense(int(rowCount), int(colCount), output)
}

func printImage(rowCount, colCount uint32, pixels []byte) {
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
