package main

import (
	"fmt"
)

const (
	epochCount = 100
	lrw, lrb   = 1e-3, 0.5
)

func main() {
	xTrain, yTrain, xTest, yTest := MnistDataByPath("data/t10k-images.idx3-ubyte"), MnistDataByPath("data/t10k-labels.idx1-ubyte"), MnistDataByPath("data/train-images.idx3-ubyte"), MnistDataByPath("data/train-labels.idx1-ubyte")
	w, b, err := Train(epochCount, xTrain, yTrain, lrw, lrb)
	if err != nil {
		panic(err)
	}
	fmt.Println(Accuracy(xTest, yTest, w, b))
	RunDrawing(500, w, b)
}
