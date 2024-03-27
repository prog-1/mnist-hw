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

func p(x, w, b *mat.Dense) *mat.Dense {
	var y mat.Dense
	y.Mul(x, w)
	y.Apply(func(_, j int, v float64) float64 {
		return sigmoid(v + b.At(0, j))
	}, &y)
	return &y
}

func inference(inputs []float64, w, b *mat.Dense) *mat.Dense {
	res := mat.NewDense(len(inputs)/784, 10, nil)
	for i := 0; i < len(inputs)/784; i++ {
		x := mat.NewDense(1, 784, inputs[i*784:(i+1)*784])
		tmp := p(x, w, b)
		for j := 0; j < 10; j++ {
			res.Set(i, j, tmp.At(0, j))
		}
	}
	return res
}

func dCost(inputs, y, p *mat.Dense) (dw, db *mat.Dense) {
	dw = mat.NewDense(784, 10, nil)
	db = mat.NewDense(1, 10, nil)
	r, c := p.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			diff := y.At(i, 0) - p.At(i, j)
			for k := 0; k < 784; k++ {
				dw.Set(k, j, dw.At(k, j)+inputs.At(i, k)*diff/float64(c))
			}
			db.Set(0, j, db.At(0, j)+diff/float64(c))
		}
	}
	return
}

func accuracy(inputs, y []float64, w, b *mat.Dense) float64 {
	var res, max, maxN float64
	_, c := b.Dims()
	for i := 0; i < len(inputs)/784; i++ {
		x := mat.NewDense(1, 784, inputs[i*784:(i+1)*784])
		pred := p(x, w, b)
		for j := 0; j < c; j++ {
			tmp := pred.At(0, j)
			if max < tmp {
				max = tmp
				maxN = float64(j)
			}
		}
		if y[i] == maxN {
			res++
		}
	}
	return res / float64(len(y)) * 100
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
	go func() {
		epochs := int(1e5)
		printEveryNthEpochs := int(1e4)
		learningRate := 1e-3
		w, squaredGradW := mat.NewDense(784, 10, nil), mat.NewDense(784, 10, nil)
		b, squaredGradB := mat.NewDense(1, 10, nil), mat.NewDense(1, 10, nil)
		epsilon := 1e-8
		for i := 0; i <= epochs; i++ {
			y := inference(xTrain, w, b)
			dw, db := dCost(inputsTrain, labelsTrain, y)
			r, c := dw.Dims()
			for j := 0; j < r; j++ {
				for k := 0; k < c; k++ {
					squaredGradW.Set(j, k, squaredGradW.At(j, k)+dw.At(j, k)*dw.At(j, k))
					w.Set(j, k, w.At(j, k)-(learningRate/math.Sqrt(squaredGradW.At(j, k)+epsilon))*dw.At(j, k))
				}
			}
			for j := 0; j < c; j++ {
				squaredGradB.Set(0, j, squaredGradB.At(0, j)+db.At(0, j)*db.At(0, j))
				b.Set(0, j, b.At(0, j)-(learningRate/math.Sqrt(squaredGradB.At(0, j)+epsilon))*db.At(0, j))
			}
			if i%printEveryNthEpochs == 0 {
				fmt.Printf(`accuracy: %.2f`, accuracy(xTest, yTest, w, b))
			}
		}
	}()
	ebiten.SetWindowSize(960, 720)
	if err := ebiten.RunGame(&Game{ebiten.NewImage(28, 28), -1, -1}); err != nil {
		log.Fatal(err)
	}
}
