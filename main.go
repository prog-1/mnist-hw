package main

import (
	"log"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	//rows and cols of the canvas and the digit images
	rows    = 28
	columns = 28
)

func main() {

	//######################## MNIST data read #########################

	//reading MNIST data set
	//pixels := readGZ("data/train-images-idx3-ubyte.gz")

	//Printing one digit from data set
	//n := 43 //digit index (max 75)
	//fmt.Println(printMnist(pixels[n*rows*columns : (n+1)*rows*columns]))

	//####################### Ebiten #########################

	//Window
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("MNIST")

	//App instance
	a := NewApp()

	//Running app
	if err := ebiten.RunGame(a); err != nil {
		log.Fatal(err)
	}

	//####################### Print #########################

	//print the stuff here
}
