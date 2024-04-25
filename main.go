package main

import (
	"fmt"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
)

var rows, columns int //global variable of rows and columns of the canvas and the digit images
var size int          //global variable of area of each image in pixels (28*28 = 784)
// var n int             //global variable of mnist image count (60000)
const outputs = 10 //output count of the digits from 0 to 9

func main() {

	//################### MNIST data read ######################

	xTrain, n := readImages("data/train-images-idx3-ubyte.gz") //getting image matrix
	yTrain := readLabels(n, "data/train-labels-idx1-ubyte.gz") //getting label (right answer) matrix

	xTest, m := readImages("data/t10k-images-idx3-ubyte.gz") //getting image matrix
	yTest := readLabels(m, "data/t10k-labels-idx1-ubyte.gz") //getting label (right answer) matrix

	//// ### Debug ###
	//Printing one digit from data set
	//i := 59999 //digit index (min - 0 | max - 59999)
	//printMnist(pixels.RawMatrix().Data[i*size : (i+1)*size])

	//################## Machine Learning ####################

	w1, b1, w2, b2 := regression(xTrain, yTrain, n)                  //going through training process
	fmt.Println("Accuracy:", accuracy(xTest, yTest, w1, b1, w2, b2)) //printing accuracy of the trained model

	//####################### Ebiten #########################

	//Window
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Digitopia")

	//App instance
	a := NewApp(w1, b1, w2, b2)

	//Running app
	if err := ebiten.RunGame(a); err != nil {
		log.Fatal(err)
	}

	//###########################################################

}
