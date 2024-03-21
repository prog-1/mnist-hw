package main

import (
	"github.com/hajimehoshi/ebiten/v2"
)

const (
	//size for canvas to equaly cover the window
	screenWidth  = 560
	screenHeight = 560
)

type App struct {
	width, height int      //screen width & height
	canvas        [][]bool //2d bool slice (of alya pixels)
}

func (a *App) Update() error {
	return nil
}

func (a *App) Draw(screen *ebiten.Image) {
	a.DrawCanvas(screen)
}

func (a *App) Layout(inWidth, inHeight int) (int, int) {
	return a.width, a.height
}

func NewApp(width, height int, canvas [][]bool) *App {
	return &App{width: width, height: height, canvas: canvas}
}
