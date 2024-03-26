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
)

const (
	//size for canvas to equaly cover the window
	screenWidth  = 560
	screenHeight = 560
)

type App struct {
	//For drawing
	screen                         *ebiten.Image //screen buffer
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
		fmt.Println(a.PrintScreen())
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

func NewApp() *App {
	return &App{ebiten.NewImage(rows, columns), 0, 0}
}

// Prints content drawed on the screen
func (a *App) PrintScreen() string {

	var print []byte

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			_, _, _, a := a.screen.At(j, i).RGBA()
			if a == 0 {
				print = append(print, ' ', ' ')
			} else if a < 32767 {
				print = append(print, '.', '.')
			} else {
				print = append(print, '#', '#')
			}
		}
		print = append(print, '\n')
	}
	return string(print)
}
