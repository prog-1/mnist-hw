package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
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
	Pixels     []byte
}

func ReadMnistImages(r io.Reader) mnistImages {
	var magic, imgCount, rows, cols uint32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		fmt.Errorf("failed to read magic number: %v", err)
	}
	// text 0081 // label // last bit is dimesnion = 1
	// image 0083 // image
	if magic != 0x803 { // image
		fmt.Errorf("Magic number is not for images: %x", magic)
	}
	if err := binary.Read(r, binary.BigEndian, &imgCount); err != nil {
		fmt.Errorf("failed to read image count: %v", err)
	}
	if err := binary.Read(r, binary.BigEndian, &rows); err != nil {
		fmt.Errorf("failed to read row count: %v", err)
	}
	if err := binary.Read(r, binary.BigEndian, &cols); err != nil {
		fmt.Errorf("failed to read column count: %v", err)
	}

	pixels := make([]byte, imgCount*rows*cols)
	n, err := io.ReadFull(r, pixels) // n - number of bytes
	if err != nil {
		fmt.Errorf("failed to read images: %v", err)
	}
	if n != len(pixels) {
		fmt.Errorf("read %d bytes; want %d", n, len(pixels))
	}

	return mnistImages{imgCount, rows, cols, pixels}
}

// Returns bytes resposnible for the image with the index given
func (m mnistImages) getImageBytes(i uint32) []byte {
	return m.Pixels[i*m.Rows*m.Cols : (i+1)*m.Rows*m.Cols]
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

func printImageFromMnistDatabase(dbFileName string, imgIndex uint) {
	f, err := os.Open(dbFileName)
	if err != nil {
		panic("failed to open file")
	}
	defer f.Close()
	m := ReadMnistImages(f)
	printImage(m.Rows, m.Cols, m.getImageBytes(uint32(imgIndex)))
}
