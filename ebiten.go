package main

import (
	"image/color"

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
	screen                         *ebiten.Image
	prevCursorPosX, prevCursorPosY int
}

func (a *App) Update() error {

	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {

		x, y := ebiten.CursorPosition()

		a.screen.Set(x, y, color.White)

		a.prevCursorPosX = x
		a.prevCursorPosY = y

	} else if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {

		x, y := ebiten.CursorPosition()

		vector.StrokeLine(a.screen, float32(a.prevCursorPosX), float32(a.prevCursorPosY), float32(x), float32(y), 2, color.White, true)

		a.prevCursorPosX = x
		a.prevCursorPosY = y
	}
	return nil
}

func (a *App) Draw(screen *ebiten.Image) {
	screen.DrawImage(a.screen, nil)
}

func (a *App) Layout(inWidth, inHeight int) (int, int) {
	return rows, columns
}

func NewApp() *App {
	return &App{ebiten.NewImage(rows, columns), 0, 0}
}
