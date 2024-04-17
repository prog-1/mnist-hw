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
		_, p := Inference(drawnImageMatrix, g.w, g.b)
		_, p2 := Inference2(&p, g.w, g.b)
		fmt.Println(mat.Formatted(&p))
		var ans []int
		for i := 0; i < 10; i++ {
			if p2.At(0, i) > 0.5 {
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
	w2 := mat.NewDense(10, 10, nil)
	b2 := mat.NewDense(1, 10, nil)
	alphaw, alphab := 1e-3, 0.01
	betterYtrain, betterYtest := convert(MtrainLabels), convert(Mlabels)
	//fmt.Println(mat.Formatted(&betterYtrain))
	w, b = gradientDescent(MtrainImages, &betterYtrain, w, b, w2, b2, alphaw, alphab, 100) // two learning
	//fmt.Println(dw, db)
	fmt.Println(accuracy(Mimages, &betterYtest, w, b))

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
func Inference(inputs *mat.Dense, w *mat.Dense, b *mat.Dense) (t1, h1 mat.Dense) {
	t1 = mat.Dense{}
	t1.Mul(inputs, w)
	t1.Apply(func(i, j int, v float64) float64 {
		return v + b.At(0, j)
	}, &t1)
	h1 = t1
	h1.Apply(func(i, j int, v float64) float64 {
		return Sigmoid(v)
	}, &h1)
	return t1, h1
}
func Inference2(inputs *mat.Dense, w *mat.Dense, b *mat.Dense) (t2, z mat.Dense) {
	t2 = mat.Dense{}
	t2.Mul(inputs, w)
	t2.Apply(func(i, j int, v float64) float64 {
		return v + b.At(0, j)
	}, &t2)
	z = t2
	z.Apply(func(i, j int, v float64) float64 {
		return Softmax(v)
	}, &z)
	return t2, z
}
func Softmax(z float64) float64 {
	a := 0.0
	for i := 0; i < 10; i++ {
		a += math.Exp(z)
	}
	return math.Exp(z) / a
}

func dCost(x, y, w2 *mat.Dense, h1, z mat.Dense, alphaw, alphab float64) (err, derrdw2, derrdb2, derrdw1, derrdb1 mat.Dense) {
	// dw = *mat.NewDense(784, 10, nil)
	// db = *mat.NewDense(1, 10, nil)
	// sub := mat.NewDense(x.RawMatrix().Rows, 10, nil)
	// sub.Sub(&p, y)

	// dw.Mul(x.T(), sub) // 784 x 10
	// dw.Scale(alphaw/float64(x.RawMatrix().Rows), &dw)
	// b := make([]float64, x.RawMatrix().Rows) // to global
	// for i := 0; i < x.RawMatrix().Rows; i++ {
	// 	b[i] = 1
	// }
	// a := mat.NewDense(1, x.RawMatrix().Rows, b)
	// db.Mul(a, sub) //db 1 x 10
	// db.Scale(alphab/float64(x.RawMatrix().Rows), &db)

	err = CrossEntropy(z, y)
	var derrdt2 mat.Dense
	derrdt2.Sub(&z, y)
	derrdw2.Mul(h1.T(), &derrdt2)
	derrdw2.Scale(alphaw/float64(x.RawMatrix().Rows), &derrdw2)
	derrdb2 = derrdt2
	derrdb2.Scale(alphab/float64(x.RawMatrix().Rows), &derrdb2)

	var derrdh1 mat.Dense
	derrdh1.Mul(&derrdt2, w2.T())
	var derrdt1 mat.Dense
	var derivativeSigmoid = func(i, j int, v float64) float64 {
		return v * (1 - v)
	}
	var derivative1 mat.Dense
	derivative1.Apply(derivativeSigmoid, &h1)
	derrdt1.MulElem(&derrdh1, &derivative1) // ?????

	derrdw1.Mul(x.T(), &derrdt1)
	derrdw1.Scale(alphaw/float64(x.RawMatrix().Rows), &derrdw1)
	derrdb1 = derrdt1
	derrdb1.Scale(alphab/float64(x.RawMatrix().Rows), &derrdb1)

	return err, derrdw2, derrdb2, derrdw1, derrdb1
} // gradient shows direction to max
func CrossEntropy(z mat.Dense, y *mat.Dense) mat.Dense {
	var err mat.Dense
	err.Apply(func(i, j int, v float64) float64 {
		return -y.At(i, j) * math.Log(z.At(i, j))
	}, &z)
	return err
}

func gradientDescent(inputs, y *mat.Dense, w, b, w2, b2 *mat.Dense, alphaw, alphab float64, epochs int) (*mat.Dense, *mat.Dense) {
	for i := 0; i < epochs; i++ {
		_, h1 := Inference(inputs, w, b)
		_, z := Inference2(&h1, w2, b2)
		_, derrdw2, derrdb2, derrdw1, derrdb1 := dCost(inputs, y, w2, h1, z, alphaw, alphab)
		w2.Sub(w2, &derrdw2) //????????????????????????????
		b2.Sub(b2, &derrdb2)
		w.Sub(w, &derrdw1)
		b.Sub(b, &derrdb1)
		// w.Sub(w, &dw)
		// b.Sub(b, &db)
	}
	return w, b
}
func accuracy(inputs, y *mat.Dense, w, b *mat.Dense) float64 {
	_, p := Inference(inputs, w, b)
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
