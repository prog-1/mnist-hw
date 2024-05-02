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

// Reads MNIST image data from a .gz file and returns x (input) matrix of {n, size} and n - image count
func readImages(filename string) (x *mat.Dense, n int) {

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

	//################################################

	x = mat.NewDense(n, size, byteToFloat(pixels))

	return x, n
}

// Reads label data from mnist .gz file and returns y (label) matrix of {n, digits}
func readLabels(n int, filename string) (y *mat.Dense) {

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
	} else if magic != wantLabel {
		fmt.Printf("invalid magic number %v", magic)
		return
	}

	//Parsing answer count
	var ac uint32
	if err := binary.Read(gzReader, binary.BigEndian, &ac); err != nil {
		log.Fatal(err)
	}

	//################################################

	// Parsing answers
	answers := make([]byte, ac)                               //slice of answers what digit is illustrated on each image
	if _, err := io.ReadFull(gzReader, answers); err != nil { //reading all answers
		log.Fatal(err)
	}

	//################################################

	y = mat.NewDense(n, outputs, nil) //creating y (label) matrix {n, outputs}

	for i, a := range answers { //for each answer (or image count)
		y.Set(i, int(a), 1) //setting value in row of our current image in column of the right answer to 1
	}

	//fmt.Println(mat.Formatted(y))
	return y
}

//// Printing one data set digit to the console
// func printDigit(pixels []float64) {

// 	var pb []byte //print buffer

// 	for i, p := range pixels { //p - pixel
// 		if i%columns == 0 {
// 			pb = append(pb, '\n')
// 		}
// 		if p == 0 {
// 			pb = append(pb, ' ', ' ')
// 		} else if p < 128 {
// 			pb = append(pb, '.', '.')
// 		} else {
// 			pb = append(pb, '#', '#')
// 		}
// 	}

// 	fmt.Println(string(pb))
// }

// Conversion of []byte into []float64 (works fine 👌)
func byteToFloat(bytes []byte) (floats []float64) {
	floats = make([]float64, len(bytes))
	for i, b := range bytes {
		floats[i] = float64(b)
	}
	return floats
}
