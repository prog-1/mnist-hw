package main

import (
	"image/color"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const (
	screenWidth, screenHeight = 500, 500
	rows, cols                = 28, 28
	title                     = "mnist"
	strokeWidth               = 1
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

func (a *App) Layout(outWidth, outHeight int) (w, h int) { return cols, rows }
func (a *App) Update() error {
	a.handleDrawing()
	a.updateMousePrevCoord()
	return nil
}
func (a *App) Draw(screen *ebiten.Image) {
	screen.DrawImage(a.screenBuffer, &ebiten.DrawImageOptions{})
}

// Initialises app and runs main loop that handles drawing on screen.
func RunDrawing() {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle(title)
	if err := ebiten.RunGame(NewGame()); err != nil {
		log.Fatal(err)
	}
}

// Colours pixels under the cursor with shades of gray when LMB is pressed
// using cursor position from the previous frame(pass -1, -1 if it's the first frame).
func (a *App) handleDrawing() {
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		x, y := ebiten.CursorPosition()
		if a.mousePrevX == -1 && a.mousePrevY == -1 {
			a.screenBuffer.Set(x, y, color.White)
		} else {
			vector.StrokeLine(a.screenBuffer, float32(a.mousePrevX), float32(a.mousePrevY), float32(x), float32(y), strokeWidth, color.White, true)
		}
	}
}

func (a *App) updateMousePrevCoord() {
	a.mousePrevX, a.mousePrevY = ebiten.CursorPosition()
}
