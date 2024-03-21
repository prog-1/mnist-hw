package main

import (
	"log"

	"github.com/hajimehoshi/ebiten/v2"
)

func main() {

	//######################## Canvas #########################

	//Calculating canvas height & width
	canvasHeight := screenHeight / canvasRows
	canvasWidth := screenWidth / canvasColumns

	//Initializing canvas
	canvas := make([][]bool, canvasHeight)
	for i := range canvas {
		canvas[i] = make([]bool, canvasWidth)
	}

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
