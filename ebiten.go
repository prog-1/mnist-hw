package main

import (
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	//size for canvas to equaly cover the window
	screenWidth  = 560
	screenHeight = 560
)

type App struct {
	screen *ebiten.Image
}

func (a *App) Update() error {
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {

		x, y := ebiten.CursorPosition()

		a.screen.Set(x, y, color.White)
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
	return &App{ebiten.NewImage(rows, columns)}
}
