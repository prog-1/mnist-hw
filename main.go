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
	screenWidth  = 300
	screenHeight = 300
)

type game struct {
	newDimension *ebiten.Image
	x, y         int
}

func (g *game) Layout(outWidth, outHeight int) (w, h int) { return 28, 28 }
func (g *game) Update() error {
	x, y := ebiten.CursorPosition()
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		g.newDimension.Set(x, y, color.White)
		g.x, g.y = x, y

	} else if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		vector.StrokeLine(g.newDimension, float32(x), float32(y), float32(g.x), float32(g.y), 2, color.White, true)
		g.x, g.y = x, y
	}
	if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		for i := 0; i < 28; i++ {
			for j := 0; j < 28; j++ {
				a, _, _, _ := g.newDimension.At(j, i).RGBA()
				if a > 0 {
					fmt.Print("#")
				} else {
					fmt.Print(" ")
				}
			}
			fmt.Println()
		}
	}
	return nil
}
func (g *game) Draw(screen *ebiten.Image) {
	screen.DrawImage(g.newDimension, nil)
}

func main() {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	g := &game{ebiten.NewImage(28, 28), 0, 0}
	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
	readData()
}

func readData() {

}
