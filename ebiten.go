package main

import (
	"fmt"
	"image/color"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const (
	screenWidth, screenHeight = 500, 500
	rowCount, colCount        = 28, 28
	title                     = "mnist"
	strokeWidth               = 2
)

type App struct {
	mousePrevX, mousePrevY int
	screenBuffer           *ebiten.Image
}

func NewGame() *App {
	return &App{
		-1, -1,
		ebiten.NewImage(screenWidth, screenHeight),
	}
}

func (a *App) Layout(outWidth, outHeight int) (w, h int) { return colCount, rowCount }
func (a *App) Update() error {
	if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		ClearConsole()
		a.drawScreenInConsole()
	} else if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		a.handleDrawing()
	} else if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight) {
		a.screenBuffer.Clear()
		a.resetMousePrevCoord()
		ClearConsole()
	}
	a.updateMousePrevCoord()
	return nil
}

func (a *App) Draw(screen *ebiten.Image) {
	screen.DrawImage(a.screenBuffer, &ebiten.DrawImageOptions{})
}

func (a *App) updateMousePrevCoord() {
	a.mousePrevX, a.mousePrevY = ebiten.CursorPosition()
}

func (a *App) resetMousePrevCoord() {
	a.mousePrevX, a.mousePrevY = -1, -1
}

// Colours pixels under the cursor with shades of gray when LMB is pressed
// using cursor position from the previous frame(pass -1, -1 if it's the first frame).
func (a *App) handleDrawing() {
	x, y := ebiten.CursorPosition()
	if a.mousePrevX == -1 && a.mousePrevY == -1 {
		a.screenBuffer.Set(x, y, color.White)
	} else {
		vector.StrokeLine(a.screenBuffer, float32(a.mousePrevX), float32(a.mousePrevY), float32(x), float32(y), strokeWidth, color.White, true)
	}
}

// Draws screen buffer's content with ASCII characters in console.
func (a *App) drawScreenInConsole() {
	for r := 0; r < rowCount; r++ {
		for c := 0; c < colCount; c++ {
			// Type assertions - https://go.dev/tour/methods/15
			alpha := a.screenBuffer.At(c, r).(color.RGBA).A
			if alpha == 0 {
				fmt.Print(" ")
			} else if alpha < 128 { // 255/2 = 127.5 ~ 128
				fmt.Print(".")
			} else {
				fmt.Print("#")
			}
		}
		fmt.Println()
	}
}

// Initialises app and runs main loop that handles drawing on screen.
func RunDrawing() {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle(title)
	if err := ebiten.RunGame(NewGame()); err != nil {
		log.Fatal(err)
	}
}
