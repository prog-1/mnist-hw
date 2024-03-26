package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"

	"gonum.org/v1/gonum/mat"
)

const (

	//Proper format of magic number and labels
	wantMagic = 0x00000803
	wantLabel = 0x00000801
)

// Reads MNIST image data from a .gz file and returns x (input) matrix of {n, size}
func readGZ(filename string) (x *mat.Dense) {

	//Opening .gz file
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	//Initializing reader
	gzReader, err := gzip.NewReader(file)
	if err != nil {
		log.Fatal(err)
	}
	defer gzReader.Close()

	//################################################

	//Parsing magic number
	var magic uint32
	if err := binary.Read(gzReader, binary.BigEndian, &magic); err != nil {
		log.Fatal(err)
	} else if magic != wantMagic {
		fmt.Printf("invalid magic number %v", magic)
		return
	}

	//Parsing image count
	var un uint32
	if err := binary.Read(gzReader, binary.BigEndian, &un); err != nil {
		log.Fatal(err)
	}

	//Parsing image rows & columns
	var urows, ucolumns uint32
	if err := binary.Read(gzReader, binary.BigEndian, &urows); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(gzReader, binary.BigEndian, &ucolumns); err != nil {
		log.Fatal(err)
	}

	//################################################

	rows, columns = int(urows), int(ucolumns)

	size = rows * columns // 28*28 = 784 - area of each image in pixels

	n = int(un) //image count

	//################################################

	// Parsing all image data into single slice of pixels
	pixels := make([]byte, n*size)

	if _, err := io.ReadFull(gzReader, pixels); err != nil {
		log.Fatal(err)
	}

	//fmt.Println("n:", n)
	//fmt.Println("size:", size)
	//fmt.Println("len pixels:", len(pixels))

	x = mat.NewDense(n, size, byteToFloat(pixels))

	//################################################

	return x
}

// Printing one data set digit to the console
func printDigit(pixels []float64) {

	var pb []byte //print buffer

	for i, p := range pixels { //p - pixel
		if i%columns == 0 {
			pb = append(pb, '\n')
		}
		if p == 0 {
			pb = append(pb, ' ', ' ')
		} else if p < 128 {
			pb = append(pb, '.', '.')
		} else {
			pb = append(pb, '#', '#')
		}
	}

	fmt.Println(string(pb))
}

// Conversion of []byte into []float64
func byteToFloat(bytes []byte) (floats []float64) {
	floats = make([]float64, len(bytes))
	for i, b := range bytes {
		floats[i] = float64(b)
	}
	return floats
}
