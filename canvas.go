package main

import (
	"bytes"
	"fmt"
)

const (
	//rows and cols of the canvas and the digit images
	rows    = 28
	columns = 28
)

// func (a *App) DrawCanvas(screen *ebiten.Image) {

// 	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {

// 		//Getting current mouse position
// 		x, y := ebiten.CursorPosition()

// 		//Checking if mouse position is in canvas boundaries
// 		if 0 <= x && x < screenWidth && 0 <= y && y < screenHeight {
// 			//Converting mouse position to canvas coordinates
// 			canvasX := x / columns
// 			canvasY := y / rows

// 			//Setting the corresponding pixel on true (making white)
// 			//a.canvas[canvasY][canvasX] = true
// 		}
// 	}

// 	//Clearing logic
// 	if ebiten.IsKeyPressed(ebiten.KeyD) {
// 		//Clearing canvas by resetting all elements to false
// 		for y := range a.canvas {
// 			for x := range a.canvas[y] {
// 				a.canvas[y][x] = false
// 			}
// 		}
// 	}
// 	if ebiten.IsKeyPressed(ebiten.KeyA) {
// 		//Clearing canvas by resetting all elements to false
// 		for y := range a.canvas {
// 			for x := range a.canvas[y] {
// 				a.canvas[y][x] = true
// 			}
// 		}
// 	}

// 	//Drawing the canvas
// 	for y, row := range a.canvas {
// 		for x, column := range row {
// 			if column {
// 				//Drawing a white pixel where canvas value is true
// 				for i := 1; i < rows; i++ {
// 					for j := 1; j < columns; j++ {
// 						screen.Set(x*columns+i, y*rows+j, color.White)
// 					}
// 				}
// 			}
// 		}
// 	}
// }

// Function to print the content to the console
func printCanvas(canvas [][]bool) {

	//Initializing printing buffer
	var buffer bytes.Buffer

	for _, row := range canvas {
		for _, column := range row {
			if !column { //if current pixel is false (black)
				buffer.WriteString(" ")
			} else { //if current pixel is true (white)
				buffer.WriteString("#")
			}
		}
		buffer.WriteString("\n")
	}

	fmt.Println(buffer.String())
}
