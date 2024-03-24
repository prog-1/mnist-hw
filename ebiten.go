package main

import (
	"log"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	screenWidth, screenHeight = 500, 500
	rows, cols                = 28, 28
	title                     = "mnist"
)

type App struct {
	screenBuffer *ebiten.Image
}

func NewGame() *App {
	return &App{
		ebiten.NewImage(screenWidth, screenHeight),
	}
}

func (a *App) Layout(outWidth, outHeight int) (w, h int) { return cols, rows }
func (a *App) Update() error {
	return nil
}
func (a *App) Draw(screen *ebiten.Image) {
	screen.DrawImage(a.screenBuffer, &ebiten.DrawImageOptions{})
}

func runDrawing() /*[]byte*/ {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle(title)
	if err := ebiten.RunGame(NewGame()); err != nil {
		log.Fatal(err)
	}
}
