package main

import (
	"fmt"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
)

func main() {

	//######################## Canvas #########################

	//Calculating canvas height & width
	canvasHeight := screenHeight / rows
	canvasWidth := screenWidth / columns

	//Initializing canvas
	canvas := make([][]bool, canvasHeight)
	for i := range canvas {
		canvas[i] = make([]bool, canvasWidth)
	}

	//######################## MNIST data read #########################

	//reading MNIST data set
	pixels := readGZ("data/train-images-idx3-ubyte.gz")

	//Printing one digit from data set
	n := 43 //digit index (max 75)
	fmt.Println(printMnist(pixels[n*rows*columns : (n+1)*rows*columns]))

	//####################### Ebiten #########################

	//Window
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("MNIST stuff")

	//App instance
	a := NewApp(screenWidth, screenHeight, canvas)

	//Running app
	if err := ebiten.RunGame(a); err != nil {
		log.Fatal(err)
	}

	//####################### Print #########################

	//Canvas print in console after app closure
	printCanvas(a.canvas)
}
