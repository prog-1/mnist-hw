package main

import (
	"fmt"
	"image/color"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"gonum.org/v1/gonum/mat"
)

type Game struct {
	lastX, lastY int
	background   *ebiten.Image
	m            *model
}

func (g *Game) Update() error {
	if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		inputs := make([]float64, 0, 28*28)
		g.lastX, g.lastY = -1, -1
		for i := 0; i < 28; i++ {
			for j := 0; j < 28; j++ {
				color := g.background.At(j, i)
				_, _, _, a := color.RGBA()
				inputs = append(inputs, float64(a)/255)

			}
		}
		VecInputs := mat.NewDense(1, 28*28, inputs)
		fmt.Println(VecInputs)
		fmt.Println(g.m.Predict(VecInputs))
	}
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		if g.lastX == -1 || g.lastY == -1 {
			x, y := ebiten.CursorPosition()
			g.background.Set(x, y, color.White)
			g.lastX, g.lastY = x, y
		} else {
			x, y := ebiten.CursorPosition()
			vector.StrokeLine(g.background, float32(g.lastX), float32(g.lastY), float32(x), float32(y), 2, color.White, true)
			g.lastX, g.lastY = x, y

		}
	}
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight) {
		g.background = ebiten.NewImage(28, 28)
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.DrawImage(g.background, &ebiten.DrawImageOptions{})
	// tmp := ebiten.NewImage(28, 28)
	// for i, v := range g.trainSet[28*28*150 : 28*28*151] {
	// 	// fmt.Println(i%28, i/28, v)
	// 	tmp.Set(i%28, i/28, color.RGBA{v, v, v, v})
	// }
	// screen.DrawImage(tmp, &ebiten.DrawImageOptions{})

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 28, 28
}

func main() {
	// m := NewModel()
	// fmt.Println(m)
	a, b, c, d := TestAndTrainingSet()
	// fmt.Println(d)
	m := NewModel(28, 28)
	m.Train(c, d)
	m.Accuracy(a,b)
	ebiten.SetWindowSize(640, 480)
	ebiten.SetWindowTitle("Hello, World!")
	if err := ebiten.RunGame(&Game{-1, -1, ebiten.NewImage(28, 28), m}); err != nil {
		log.Fatal(err)
	}
}
