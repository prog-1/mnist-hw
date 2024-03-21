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

func (m mnistImages) printImage(pixels []byte) {
	for i := uint32(0); i < m.Rows; i++ {
		for j := uint32(0); j < m.Cols; j++ {
			if pixel := pixels[i*m.Cols+j]; pixel == 0 {
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
	m.printImage(m.getImageBytes(uint32(imgIndex)))
}

func main() {
	printImageFromMnistDatabase("data/t10k-images.idx3-ubyte", 3)
}

// How to implement on-screen drawing:
// 1. Output the pixels on the screen
// 2. On LNB click, get the coord of the mouse
// 3. Color the pixels
// 3.1. Color the pixel under the cursor white
// 3.2. Color the adjacent pixels gray
// 3.2.1. Add coef. that is responsible for how many adjacent pixels will be coloured grey(uint gp - gray pixels), e.g. 2
// 3.2.2. Setting the adjacent(optional)
/*
	gpc := 2 // gray pixel count
	wp := getCursorPos() // white pixel
	setGrayScale(wp, 100)

	// Recursively colours the pixel and its adjuscent ones while index <= gpc
	colourise := func(pixel, index) {
		for i := index; i <= gp; i++ {
			gs := 100 / (i+1) // gray scale value from 0 to 100
			setGrayScale({cp.x-1, cp.y-1}, gs)
			setGrayScale({cp.x+1, cp.y-1}, gs)
			setGrayScale({cp.x-1, cp.y+1}, gs)
			setGrayScale({cp.x+1, cp.y+1}, gs)
		}
	}
	// TODO: Finish it
*/
// 4. Store pixels in the byte array in the row-major order
