package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"image/color"
	"io"
	"log"
	"math"
	"os"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"gonum.org/v1/gonum/mat"
)

const (
	screenWidth  = 300
	screenHeight = 300
)

type game struct {
	newDimension *ebiten.Image
	x, y         int
	w, b         *mat.Dense
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
		var drawnImage []float64
		for i := 0; i < 28; i++ {
			for j := 0; j < 28; j++ {
				a, _, _, _ := g.newDimension.At(j, i).RGBA()
				if a > 0 {
					fmt.Print("#")
					drawnImage = append(drawnImage, 1)
				} else {
					fmt.Print(" ")
					drawnImage = append(drawnImage, 0)
				}
			}
			fmt.Println()
		}
		drawnImageMatrix := mat.NewDense(1, 784, drawnImage)
		p := Inference(drawnImageMatrix, g.w, g.b)
		fmt.Println(mat.Formatted(&p))
		var ans []int
		for i := 0; i < 10; i++ {
			if p.At(0, i) > 0.5 {
				ans = append(ans, i)
			}
		}
		fmt.Println(ans)
	}
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight) {
		g.newDimension.Clear()
	}
	return nil
}
func (g *game) Draw(screen *ebiten.Image) {
	screen.DrawImage(g.newDimension, nil)
}

func main() { // softmax
	ebiten.SetWindowSize(screenWidth, screenHeight)
	images := ReadImages("t10k-images.idx3-ubyte")
	labels := ReadLabels("t10k-labels.idx1-ubyte")
	trainImages := ReadImages("train-images.idx3-ubyte")
	trainLabels := ReadLabels("train-labels.idx1-ubyte")
	Mimages := toMatrix(images, 10000, 784)
	Mlabels := toMatrix(labels, 10000, 1)
	MtrainImages := toMatrix(trainImages, 60000, 784)
	MtrainLabels := toMatrix(trainLabels, 60000, 1)
	w := mat.NewDense(784, 10, nil) // random weights todo
	b := mat.NewDense(1, 10, nil)
	alphaw, alphab := 1e-3, 0.01
	betterYtrain, betterYtest := convert(MtrainLabels), convert(Mlabels)
	//fmt.Println(mat.Formatted(&betterYtrain))
	w, b = gradientDescent(MtrainImages, &betterYtrain, w, b, alphaw, alphab, 100) // two learning
	//fmt.Println(dw, db)
	fmt.Println(accuracy(Mimages, &betterYtest, w, b))

	// var drawnImage []float64
	// for i := 0; i < 28; i++ {
	// 	for j := 0; j < 28; j++ {
	// 		_, _, _, a := g.newDimension.At(j, i).RGBA()
	// 		if a > 0 {
	// 			drawnImage = append(drawnImage, 1)
	// 		} else {
	// 			drawnImage = append(drawnImage, 0)
	// 		}
	// 	}
	// }
	// drawnImageMatrix := mat.NewDense(1, 784, drawnImage)
	// p := Inference(drawnImageMatrix, w, b)
	// //fmt.Println(mat.Formatted(drawnImageMatrix))
	// fmt.Println(mat.Formatted(&p))
	g := &game{ebiten.NewImage(28, 28), 0, 0, w, b}
	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
}
func convert(y *mat.Dense) mat.Dense {
	rows, _ := y.Dims()
	b := make([]float64, rows*10)
	for i := 0; i < rows; i++ {
		b[i*10+int(y.At(i, 0))] = 1
	}
	return *mat.NewDense(rows, 10, b)
}
func toMatrix(images []byte, rows, columns int) *mat.Dense {
	data := make([]float64, len(images))
	for i, v := range images {
		data[i] = float64(v)
	}
	return mat.NewDense(rows, columns, data)
}
func ReadImages(filePath string) []byte {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	var magic uint32
	const wantMagic = 0x00000803
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		log.Fatal(err)
	} else if magic != wantMagic {
		log.Fatal(fmt.Errorf("magic = %v, wantMagic  = %v", magic, wantMagic))
	}
	var nImages, rows, columns uint32
	if err := binary.Read(reader, binary.BigEndian, &nImages); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(reader, binary.BigEndian, &rows); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(reader, binary.BigEndian, &columns); err != nil {
		log.Fatal(err)
	}
	images := make([]byte, nImages*rows*columns)
	if _, err := io.ReadFull(reader, images); err != nil {
		log.Fatal(err)
	}
	return images
}
func ReadLabels(filePath string) []byte {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	var magic uint32
	const wantMagic = 0x00000801
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		log.Fatal(err)
	} else if magic != wantMagic {
		log.Fatal(fmt.Errorf("magic = %v, wantMagic  = %v", magic, wantMagic))
	}
	var nImages uint32
	if err := binary.Read(reader, binary.BigEndian, &nImages); err != nil {
		log.Fatal(err)
	}
	images := make([]byte, nImages)
	if _, err := io.ReadFull(reader, images); err != nil {
		log.Fatal(err)
	}
	return images
}
func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
func Inference(inputs *mat.Dense, w *mat.Dense, b *mat.Dense) (res mat.Dense) {
	res = mat.Dense{}
	res.Mul(inputs, w)
	res.Apply(func(i, j int, v float64) float64 {
		return Sigmoid(v + b.At(0, j))
	}, &res)
	return res
}
func dCost(x, y *mat.Dense, p mat.Dense, alphaw, alphab float64) (dw, db mat.Dense) {
	dw = *mat.NewDense(784, 10, nil)
	db = *mat.NewDense(1, 10, nil)
	sub := mat.NewDense(x.RawMatrix().Rows, 10, nil)
	sub.Sub(&p, y)

	dw.Mul(x.T(), sub) // 784 x 10
	dw.Scale(alphaw/float64(x.RawMatrix().Rows), &dw)
	b := make([]float64, x.RawMatrix().Rows) // to global
	for i := 0; i < x.RawMatrix().Rows; i++ {
		b[i] = 1
	}
	a := mat.NewDense(1, x.RawMatrix().Rows, b)
	db.Mul(a, sub) //db 1 x 10
	db.Scale(alphab/float64(x.RawMatrix().Rows), &db)

	return dw, db
} // gradient shows direction to max
func gradientDescent(inputs, y *mat.Dense, w, b *mat.Dense, alphaw, alphab float64, epochs int) (*mat.Dense, *mat.Dense) {
	for i := 0; i < epochs; i++ {
		p := Inference(inputs, w, b)
		dw, db := dCost(inputs, y, p, alphaw, alphab)
		w.Sub(w, &dw)
		b.Sub(b, &db)
	}
	return w, b
}
func accuracy(inputs, y *mat.Dense, w, b *mat.Dense) float64 {
	p := Inference(inputs, w, b)
	var correct int
	for i := 0; i < 10000; i++ {
		var index1, index2 int
		for j := 0; j < 10; j++ {
			if p.At(i, j) > p.At(i, index1) {
				index1 = j
			}
		}
		for j := 0; j < 10; j++ {
			if y.At(i, j) > y.At(i, index2) {
				index2 = j
			}
		}
		if index1 == index2 {
			correct++
		}
	}
	return float64(correct) / 10000
}
