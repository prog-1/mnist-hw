package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"

	"gonum.org/v1/gonum/mat"
)

func readMnistImageSet(r io.Reader) []byte {
	var magicNumber uint32
	err := binary.Read(r, binary.BigEndian, &magicNumber)
	if err != nil {
		log.Fatal("Cannot parse binary file: ", err)
	}
	if magicNumber != 0x803 {
		log.Fatal("Wrong magic number")
	}

	var imageNum, cols, rows uint32
	err = binary.Read(r, binary.BigEndian, &imageNum)
	if err != nil {
		log.Fatal("Cannot parse binary file: ", err)
	}
	err = binary.Read(r, binary.BigEndian, &cols)
	if err != nil {
		log.Fatal("Cannot parse binary file: ", err)
	}
	err = binary.Read(r, binary.BigEndian, &rows)
	if err != nil {
		log.Fatal("Cannot parse binary file: ", err)
	}
	fmt.Println(imageNum, rows, cols)
	res := make([]byte, imageNum*rows*cols)
	n, err := io.ReadFull(r, res)
	if n != len(res) {
		log.Fatal("Length of data is not correct")
	}
	return res
}

func readMnistLabels(r io.Reader) []byte {
	var magicNumber uint32
	err := binary.Read(r, binary.BigEndian, &magicNumber)
	if err != nil {
		log.Fatal("Cannot parse binary file: ", err)
	}
	if magicNumber != 0x801 {
		log.Fatal("Wrong magic number")
	}
	var lablesNum uint32
	err = binary.Read(r, binary.BigEndian, &lablesNum)
	if err != nil {
		log.Fatal("Cannot parse binary file: ", err)
	}
	res := make([]byte, lablesNum)
	n, err := io.ReadFull(r, res)
	if n != len(res) {
		log.Fatal("Length of data is not correct")
	}
	a := make([]byte, 0)
	for _, v := range res {
		tmp := make([]byte, 10)
		tmp[v] = 1
		a = append(a, tmp...)
	}
	return a
}

func DebugPrint(cols int, image []byte) {
	for i, v := range image {
		if i%cols == 0 {
			fmt.Println()
		}
		if v == 0 {
			fmt.Print(" ")
		} else if v < 128 {
			fmt.Print(".")
		} else {
			fmt.Print("#")
		}
	}
}

func GetTestAndTrainingSet() (TestSet *mat.Dense, TestLablesSet *mat.Dense, TrainSet *mat.Dense, TrainLablesSet *mat.Dense) {
	file, err := os.Open("dataset/t10k-images.idx3-ubyte")
	if err != nil {
		log.Fatal(err)
	}
	TestImages := readMnistImageSet(file)
	file.Close()
	file, err = os.Open("dataset/t10k-labels.idx1-ubyte")
	if err != nil {
		log.Fatal(err)
	}
	// fmt.Println(len(TestImages)/784, len(bytesToFloat64Slice(TestImages)))
	TestSet = mat.NewDense(len(TestImages)/784, 784, bytesToFloat64Slice(TestImages))
	TestLables := readMnistLabels(file)
	TestLablesSet = mat.NewDense(len(TestLables)/10, 10, bytesToFloat64Slice(TestLables))
	file.Close()

	file, err = os.Open("dataset/train-images.idx3-ubyte")
	if err != nil {
		log.Fatal(err)
	}

	TrainImages := readMnistImageSet(file)
	TrainSet = mat.NewDense(len(TrainImages)/784, 784, bytesToFloat64Slice(TrainImages))
	file.Close()

	file, err = os.Open("dataset/train-labels.idx1-ubyte")
	if err != nil {
		log.Fatal(err)
	}
	TrainLables := readMnistLabels(file)
	TrainLablesSet = mat.NewDense(len(TrainLables)/10, 10, bytesToFloat64Slice(TrainLables))
	file.Close()
	return
	// DebugPrint(28, images[28*28*9999:28*28*10000])
	// fmt.Println(lables[9999])
}

func bytesToFloat64Slice(bytes []byte) []float64 {
	floats := make([]float64, len(bytes))
	for i, b := range bytes {
		floats[i] = float64(b)
	}
	return floats
}
