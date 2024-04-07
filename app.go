package main

import (
	"fmt"
	"image/color"
	"os"

	"github.com/hajimehoshi/ebiten/inpututil"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"gonum.org/v1/gonum/mat"
)

type Game struct {
	screen       *ebiten.Image
	prevX, prevY int
	w, b         *mat.Dense
}

func (g *Game) Update() error {
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		x, y := ebiten.CursorPosition()
		g.screen.Set(x, y, color.White)
		g.prevX, g.prevY = x, y
	} else if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		x, y := ebiten.CursorPosition()
		vector.StrokeLine(g.screen, float32(x), float32(y), float32(g.prevX), float32(g.prevY), 2, color.White, true)
		g.prevX, g.prevY = x, y
	} else if ebiten.IsMouseButtonPressed(ebiten.MouseButtonRight) {
		g.screen.Clear()
	}
	if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		inputs := make([]float64, 0, 28*28)
		g.prevX, g.prevY = -1, -1
		for i := 0; i < 28; i++ {
			for j := 0; j < 28; j++ {
				_, _, _, a := g.screen.At(j, i).RGBA()
				inputs = append(inputs, float64(a)/255)
				if a > 0 {
					fmt.Print("#")
				} else {
					fmt.Print(" ")
				}
			}
			fmt.Println()
		}
		fmt.Println(predict(inputs, g.w, g.b))
	}
	if ebiten.IsKeyPressed(ebiten.KeyEscape) {
		os.Exit(0)
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.DrawImage(g.screen, &ebiten.DrawImageOptions{})
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 28, 28
}
