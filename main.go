package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand/v2"
	"os"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"gonum.org/v1/gonum/mat"
)

type Game struct {
	screen       *ebiten.Image
	prevX, prevY int
	w, b, w2, b2 *mat.Dense
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
		fmt.Println(predict(inputs, g.w, g.b, g.w2, g.b2))
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

func softmax(z, s float64) float64 {
	return math.Exp(z) / s
}

func inference(inputs, w, b, w2, b2 *mat.Dense) (*mat.Dense, *mat.Dense) {
	var y mat.Dense
	y.Mul(inputs, w)
	y.Apply(func(_, j int, v float64) float64 {
		return sigmoid(v + b.At(0, j))
	}, &y)
	var z mat.Dense
	z.Mul(&y, w2)
	z.Apply(func(_, j int, v float64) float64 {
		return v + b2.At(0, j)
	}, &z)
	_, c := z.Dims()
	z.Apply(func(i, j int, v float64) float64 {
		var s float64
		for k := 0; k < c; k++ {
			s += math.Exp(z.At(i, k))
		}
		return softmax(v, s)
	}, &z)
	return &z, &y
}

func dCost(inputs, y, p, h1, w2 *mat.Dense, lrW, lrB float64) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	var dw, dw2 mat.Dense
	r, c := p.Dims()
	data := make([]float64, r*10)
	for i := 0; i < r; i++ {
		data[i*10+int(y.At(i, 0))] = 1
	}
	l := mat.NewDense(r, c, data)
	var dt2, dh1, dert1, dt1 mat.Dense
	dt2.Sub(p, l)
	dw2.Mul(h1.T(), &dt2)
	dw2.Scale(lrW/float64(inputs.RawMatrix().Rows), &dw2)
	db2 := mat.DenseCopyOf(&dt2)
	db2.Scale(lrB/float64(inputs.RawMatrix().Rows), db2)
	dh1.Mul(&dt2, w2.T())
	derSigmoid := func(i, j int, v float64) float64 {
		return v * (1 - v)
	}
	dert1.Apply(derSigmoid, &dh1)
	dt1.MulElem(&dt2, &dert1)
	dw.Mul(inputs.T(), &dt1)
	dw.Scale(lrW/float64(inputs.RawMatrix().Rows), &dw)
	db := mat.DenseCopyOf(&dt1)
	db.Scale(lrB/float64(inputs.RawMatrix().Rows), db)
	return &dw, db, &dw2, db2
}

func accuracy(inputs, y []float64, w, b, w2, b2 *mat.Dense) float64 {
	var res float64
	for i := 0; i < len(y); i++ {
		n := predict(inputs[i*784:(i+1)*784], w, b, w2, b2)
		if int(y[i]) == n {
			res++
		}
	}
	return res / float64(len(y)) * 100
}

func predict(inputs []float64, w, b, w2, b2 *mat.Dense) int {
	x := mat.NewDense(1, 784, inputs)
	pred, _ := inference(x, w, b, w2, b2)
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

func random(m *mat.Dense, r *rand.Rand) *mat.Dense {
	m.Apply(func(i int, j int, v float64) float64 {
		return 10 - 20*r.Float64()
	}, m)
	return m
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
	rand := rand.New(rand.NewPCG(1, 2))
	w, b := mat.NewDense(784, 10, nil), mat.NewDense(1, 10, nil)
	w2, b2 := mat.NewDense(10, 10, nil), mat.NewDense(1, 10, nil)
	w, b, w2, b2 = random(w, rand), random(b, rand), random(w2, rand), random(b2, rand)
	go func() {
		epochs := int(1e3)
		lrW, lrB := 1e0, 1e1
		for i := 0; i <= epochs; i++ {
			y, h1 := inference(inputsTrain, w, b, w2, b2)
			dw, db, dw2, db2 := dCost(inputsTrain, labelsTrain, y, h1, w2, lrW, lrB)
			dbData, db2Data := make([]float64, 10), make([]float64, 10)
			for j := range dbData {
				dbData[j] = mat.Sum(db.ColView(j))
				db2Data[j] = mat.Sum(db2.ColView(j))
			}
			dbt, db2t := mat.NewDense(1, 10, dbData[:]), mat.NewDense(1, 10, db2Data[:])
			w.Sub(w, dw)
			b.Sub(b, dbt)
			w2.Sub(w2, dw2)
			b2.Sub(b2, db2t)
			fmt.Printf(`Epoch: %d
            Accuracy: %.2f
            `, i, accuracy(xTest, yTest, w, b, w2, b2))
		}
	}()
	ebiten.SetWindowSize(960, 720)
	if err := ebiten.RunGame(&Game{ebiten.NewImage(28, 28), -1, -1, w, b, w2, b2}); err != nil {
		log.Fatal(err)
	}
}
