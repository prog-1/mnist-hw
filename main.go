package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	imageSize                 = 28 * 28
	screenWidth, screenHeight = 1280, 900
	epochs                    = 1000
	printEveryXEpoch          = 100
	learningRateB             = 1
	learningRateW             = 1
	nOutputs                  = 10
	trainAmount               = 60000
)

func readImages() (*mat.Dense, error) {
	var wantMagic uint32 = 0x00000803
	var mnistMagic uint32
	var xFloat []float64

	//Open file
	file, err := os.Open("train-images.idx3-ubyte")
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := bufio.NewReader(file)

	//Check magic
	if err := binary.Read(reader, binary.BigEndian, &mnistMagic); err != nil {
		log.Fatal(err)
	} else if wantMagic != mnistMagic {
		log.Fatal(fmt.Errorf("magic = %v, wantMagic  = %v", wantMagic, mnistMagic))
	}

	//Read images
	var nImages, nRows, nCols uint32
	if err := binary.Read(reader, binary.BigEndian, &nImages); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(reader, binary.BigEndian, &nRows); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(reader, binary.BigEndian, &nCols); err != nil {
		log.Fatal(err)
	}
	pixels := make([]byte, nImages*nRows*nCols)
	if _, err := io.ReadFull(reader, pixels); err != nil {
		log.Fatal(err)
	}

	for _, j := range pixels {
		xFloat = append(xFloat, float64(j))
	}
	// draw(pixels)
	x := mat.NewDense(trainAmount, imageSize, xFloat)

	return x, nil
}

func readLabels() (*mat.Dense, error) {
	var wantMagic uint32 = 0x00000801
	var labelMagic uint32
	var yFloat []float64

	//Open file
	file, err := os.Open("train-labels.idx3-ubyte")
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := bufio.NewReader(file)

	//Chechk magic
	if err := binary.Read(reader, binary.BigEndian, &labelMagic); err != nil {
		log.Fatal(err)
	} else if wantMagic != labelMagic {
		log.Fatal(fmt.Errorf("magic = %v, wantMagic  = %v", wantMagic, labelMagic))
	}

	//Read labels
	var nLabels uint32
	if err := binary.Read(reader, binary.BigEndian, &nLabels); err != nil {
		log.Fatal(err)
	}
	labels := make([]byte, nLabels)
	if _, err := io.ReadFull(reader, labels); err != nil {
		log.Fatal(err)
	}

	for _, j := range labels {
		yFloat = append(yFloat, float64(j))
	}
	y := mat.NewDense(trainAmount, 1, yFloat)

	return y, nil
}

func draw(img []byte) {
	var b []byte
	n := 3
	pixels := img[n*imageSize : (n+1)*imageSize]
	for i, p := range pixels {
		if uint32(i)%28 == 0 {
			b = append(b, '\n')
		}
		if p == 0 {
			b = append(b, ' ', ' ')
		} else if p < 128 {
			b = append(b, '.', '.')
		} else {
			b = append(b, '#', '#')
		}
	}
	fmt.Print(string(b))
}

func main() {
	readImages()
	b := mat.NewDense(1, nOutputs, nil)
	w := mat.NewDense(imageSize, nOutputs, nil)
	x, _ := readImages()
	y, _ := readLabels()

	//Main loop
	for i := 0; i <= epochs; i++ {
		p := inference(x, w, b)
		dw, db := deratives(x, y, p)
		w.Sub(w, dw)
		b.Sub(b, db)
		if i%printEveryXEpoch == 0 {
			fmt.Printf("Epoch number: %d\ndw: %f\ndb: %f\n", i, dw, db)
		}
	}
	ebiten.SetWindowSize(960, 720)
	if err := ebiten.RunGame(&Game{ebiten.NewImage(28, 28), -1, -1, w, b}); err != nil {
		log.Fatal(err)
	}
}

func deratives(x, p, y *mat.Dense) (*mat.Dense, *mat.Dense) {
	//p[n,10] y[n,1] d[n,1] dw[784,1] db[1,10]
	var d, dw, gradW, gradB *mat.Dense
	n, b := p.Dims()
	subtractSet := mat.NewDense(n, b, nil)
	for i := 0; i < n; i++ {
		subtractSet.Set(i, int(y.At(i, 1)), 1)
	}
	d.Sub(p, subtractSet)
	dw.Mul(x.T(), d)
	gradW.Scale(learningRateW/float64(n), dw)

	var a [10]float64
	for i := range a {
		a[i] = mat.Sum(d.ColView(i))
	}
	db := mat.NewDense(1, 10, a[:])
	gradB.Scale(learningRateW/float64(n), db)

	return gradW, gradB
}

func inference(x, w, b *mat.Dense) *mat.Dense {
	var p mat.Dense
	p.Mul(x, w)
	p.Apply(func(_, j int, v float64) float64 {
		return sigmoid(v + b.At(0, j))
	}, &p)
	return &p
}

func sigmoid(z float64) float64 { return 1 / (1 + math.Exp(-z)) }

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
