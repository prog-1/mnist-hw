package main

import (
	"fmt"
	"image/color"
	"os"
	"os/exec"
	"runtime"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"gonum.org/v1/gonum/mat"
)

const (
	//size for canvas to equaly cover the window
	screenWidth  = 560
	screenHeight = 560
)

type App struct {
	//For drawing
	screen                         *ebiten.Image //screen buffer
	w, b                           *mat.Dense    //trained weight and bias matrices to run our drawing through the model
	prevCursorPosX, prevCursorPosY int           //previous mouse position to draw the line with anti-aliasing
}

func (a *App) Update() error {

	//Drawing handling

	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) { //if the mouse is pressed first time

		//Getting current mouse position
		x, y := ebiten.CursorPosition()

		//Drawing just a pixel if nothing else is done else
		a.screen.Set(x, y, color.White)

		//Saving current position as previous mouse position
		a.prevCursorPosX = x
		a.prevCursorPosY = y

	} else if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {

		//Getting current mouse position
		x, y := ebiten.CursorPosition()

		//Drawing the line with anti-aliasing
		vector.StrokeLine(a.screen, float32(a.prevCursorPosX), float32(a.prevCursorPosY), float32(x), float32(y), 2, color.White, true)

		//Saving new mouse position
		a.prevCursorPosX = x
		a.prevCursorPosY = y

	} else if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		//a.PrintScreen()
		fmt.Println(inference(a.ScreenToMatrix(), a.w, a.b))
	}

	if ebiten.IsKeyPressed(ebiten.KeyD) {

		//Clearing the screen (recreating it)
		a.screen = ebiten.NewImage(rows, columns)

		if runtime.GOOS == "windows" { //if OS is windows
			//Clearing console
			cmd := exec.Command("cmd", "/c", "cls")
			cmd.Stdout = os.Stdout
			cmd.Run()
		}

	}

	return nil
}

func (a *App) Draw(screen *ebiten.Image) {
	screen.DrawImage(a.screen, nil)
}

func (a *App) Layout(inWidth, inHeight int) (int, int) {
	return rows, columns //returning 28x28 to set bigger pixel size
}

func NewApp(w, b *mat.Dense) *App {
	return &App{ebiten.NewImage(rows, columns), w, b, 0, 0}
}

// Prints content drawed on the screen to the console
func (a *App) PrintScreen() {

	var pb []byte //print buffer

	for r := 0; r < rows; r++ { //for each row (r - row)
		for c := 0; c < columns; c++ { //for each column (c - column)

			_, _, _, alpha := a.screen.At(c, r).RGBA() //taking alpha channel from current pixel

			if alpha == 0 {
				pb = append(pb, ' ', ' ')
			} else if alpha < 32767 { //some magic number that is equal to 128 in byte or something
				pb = append(pb, '.', '.')
			} else {
				pb = append(pb, '#', '#')
			}
		}
		pb = append(pb, '\n')
	}

	fmt.Println(string(pb)) //converting and printing the print buffer

}

// Converts screen pixels into input matrix {n, size}
func (a *App) ScreenToMatrix() *mat.Dense {

	var pixels []float64 //single slice of all screen pixels

	for r := 0; r < rows; r++ { //for each row (r - row)
		for c := 0; c < columns; c++ { //for each column (c - column)

			_, _, _, alpha := a.screen.At(c, r).RGBA() //taking alpha from current pixel

			pixels = append(pixels, float64(alpha)) //saving into slice of pixels
		}
	}

	return mat.NewDense(1, size, pixels)
}
