package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"os"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
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

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func inference(inputs, w, b *mat.Dense) *mat.Dense {
	var res mat.Dense
	res.Mul(inputs, w)
	res.Apply(func(_, j int, v float64) float64 {
		return sigmoid(v + b.At(0, j))
	}, &res)
	return &res
}

func dCost(inputs, y, p *mat.Dense, lrW, lrB float64) (dw, db *mat.Dense) {
	r, c := p.Dims()
	data := make([]float64, r*10)
	for i := 0; i < r; i++ {
		data[i*10+int(y.At(i, 0))] = 1
	}
	l := mat.NewDense(r, c, data)
	dw = mat.NewDense(784, 10, nil)
	var diff mat.Dense
	diff.Sub(p, l)
	dw.Mul(inputs.T(), &diff)
	dw.Scale(lrW/float64(inputs.RawMatrix().Rows), dw)
	var b [10]float64
	for i := range b {
		b[i] = mat.Sum(diff.ColView(i))
	}
	db = mat.NewDense(1, 10, b[:])
	db.Scale(lrB/float64(inputs.RawMatrix().Rows), db)
	return
}

func accuracy(inputs, y []float64, w, b *mat.Dense) float64 {
	var res float64
	for i := 0; i < len(y); i++ {
		n := predict(inputs[i*784:(i+1)*784], w, b)
		if int(y[i]) == n {
			res++
		}
	}
	return res / float64(len(y)) * 100
}

func predict(inputs []float64, w, b *mat.Dense) int {
	x := mat.NewDense(1, 784, inputs)
	pred := inference(x, w, b)
	var num int
	var numP float64
	for i := 0; i < 10; i++ {
		if pred.At(0, i) > numP {
			numP = pred.At(0, i)
			num = i
		}
	}
	return num
}

func main() {
	trainImages, testImages, trainLabels, testLabels := ReadImages("train-images.idx3-ubyte"), ReadImages("t10k-images.idx3-ubyte"), ReadLabels("train-labels.idx1-ubyte"), ReadLabels("t10k-labels.idx1-ubyte")
	var xTrain, yTrain, xTest, yTest []float64
	for i := range trainImages {
		xTrain = append(xTrain, float64(trainImages[i]))
	}
	for i := range trainLabels {
		yTrain = append(yTrain, float64(trainLabels[i]))
	}
	for i := range testImages {
		xTest = append(xTest, float64(testImages[i]))
	}
	for i := range testLabels {
		yTest = append(yTest, float64(testLabels[i]))
	}
	inputsTrain, labelsTrain := mat.NewDense(60000, 784, xTrain), mat.NewDense(len(yTrain), 1, yTrain)
	w, b := mat.NewDense(784, 10, nil), mat.NewDense(1, 10, nil)
	go func() {
		epochs := int(1e2)
		lrW, lrB := 1e-5, 1e0
		for i := 0; i <= epochs; i++ {
			y := inference(inputsTrain, w, b)
			dw, db := dCost(inputsTrain, labelsTrain, y, lrW, lrB)
			w.Sub(w, dw)
			b.Sub(b, db)
			fmt.Printf(`Epoch: %d
			Accuracy: %.2f
			`, i, accuracy(xTest, yTest, w, b))
		}
	}()
	ebiten.SetWindowSize(960, 720)
	if err := ebiten.RunGame(&Game{ebiten.NewImage(28, 28), -1, -1, w, b}); err != nil {
		log.Fatal(err)
	}
}
