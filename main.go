package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"image/color"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"gonum.org/v1/gonum/mat"
)

const (
	screenWidth, screenHeight = 500, 500
	wantMagic                 = 0x00000803
	wantLabels                = 0x00000801
	rows, cols                = 28, 28
	epochs                    = 100
	learningRateW             = 1e-1
	learningRateB             = 1
)

type Num struct {
	screen     *ebiten.Image
	image      []float64
	w, b       *mat.Dense
	posX, posY int
}

func readImages(data string) ([]float64, error) {
	file, err := os.Open(data)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	var magicNumber uint32
	err = binary.Read(gz, binary.BigEndian, &magicNumber)
	if err != nil {
		return nil, err
	}
	if magicNumber != wantMagic {
		return nil, fmt.Errorf("invalid magic number: %d", magicNumber)
	}

	var nImages, nRows, nCols uint32
	err = binary.Read(gz, binary.BigEndian, &nImages)
	if err != nil {
		return nil, err
	}
	err = binary.Read(gz, binary.BigEndian, &nRows)
	if err != nil {
		return nil, err
	}
	err = binary.Read(gz, binary.BigEndian, &nCols)
	if err != nil {
		return nil, err
	}
	images := make([]float64, nImages*rows*cols)
	bImages := make([]byte, nImages*rows*cols)
	_, err = io.ReadFull(gz, bImages)
	if err != nil {
		return nil, err
	}
	for i := 0; i < int(nImages*rows*cols); i++ {
		images[i] = float64(bImages[i])
	}
	return images, nil
}

func readLabels(data string) (*mat.Dense, error) {
	file, err := os.Open(data)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	var magicNumber uint32
	err = binary.Read(gz, binary.BigEndian, &magicNumber)
	if err != nil {
		return nil, err
	}
	if magicNumber != wantLabels {
		return nil, fmt.Errorf("invalid magic number: %d", magicNumber)
	}
	var nLabels uint32
	err = binary.Read(gz, binary.BigEndian, &nLabels)
	if err != nil {
		return nil, err
	}
	labels := make([]byte, nLabels)
	_, err = io.ReadFull(gz, labels)
	if err != nil {
		return nil, err
	}
	a := make([]float64, nLabels*10)
	for i, v := range labels {
		a[i*10+int(v)] = 1
	}
	return mat.NewDense(int(nLabels), 10, a), nil
}

func (n *Num) UpdateWeights(w, b *mat.Dense) {
	n.w, n.b = w, b
}

func (n *Num) Update() error {
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		x, y := ebiten.CursorPosition()
		n.screen.Set(x, y, color.White)
		n.posX, n.posY = x, y
	} else if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		x, y := ebiten.CursorPosition()
		if x != n.posX || y != n.posY {
			vector.StrokeLine(n.screen, float32(n.posX), float32(n.posY), float32(x), float32(y), 2, color.White, true)
			n.posX = x
			n.posY = y
		}
	} else if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		fmt.Println(n.PrintScreen(n.w, n.b))
	}
	if ebiten.IsKeyPressed(ebiten.KeyC) {
		n.ClearScreen()
	}
	if ebiten.IsKeyPressed(ebiten.KeyQ) {
		os.Exit(0)
	}

	return nil
}

func (n *Num) PrintScreen(w, b *mat.Dense) (string, int) {
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	var builder strings.Builder
	n.image = make([]float64, 0, 784)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			colorr := n.screen.At(x, y).(color.RGBA)
			n.image = append(n.image, float64(colorr.A)/255)
			if colorr == white {
				builder.WriteRune('#')
			} else {
				builder.WriteRune(' ')
			}
		}
		builder.WriteRune('\n')
	}
	return builder.String(), predict(mat.NewDense(1, 784, n.image), w, b)
}

func (n *Num) ClearScreen() {
	n.screen = ebiten.NewImage(rows, cols)
	if runtime.GOOS == "windows" {
		cmd := exec.Command("cmd", "/c", "cls")
		cmd.Stdout = os.Stdout
		cmd.Run()
	}
}

func (n *Num) Draw(screen *ebiten.Image) {
	screen.DrawImage(n.screen, nil)
}

func (n *Num) Layout(width, height int) (int, int) {
	return rows, cols
}

func main() {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	timages, err := readImages("data/train-images-idx3-ubyte.gz")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	matrixTImages := mat.NewDense(len(timages)/784, 784, timages)
	images, err := readImages("data/t10k-images-idx3-ubyte.gz")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	matrixImages := mat.NewDense(len(images)/784, 784, images)
	tlabels, err := readLabels("data/train-labels-idx1-ubyte.gz")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	labels, err := readLabels("data/t10k-labels-idx1-ubyte.gz")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	w := mat.NewDense(784, 10, make([]float64, 7840))
	b := mat.NewDense(1, 10, make([]float64, 10))
	for i := 0; i < epochs; i++ {
		p := inference(matrixTImages, w, b)
		dw, db := dCost(matrixTImages, tlabels, p)
		w.Sub(w, dw)
		b.Sub(b, db)
	}
	fmt.Println(accuracy(matrixImages, labels, w, b))
	n := &Num{ebiten.NewImage(rows, cols), []float64{}, w, b, 0, 0}
	if err := ebiten.RunGame(n); err != nil {
		log.Fatal(err)
	}
	n.UpdateWeights(w, b)
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func predict(image, w, b *mat.Dense) int {
	p := inference(image, w, b)
	_, col := p.Dims()
	maxi, maxv := 0, p.At(0, 0)
	for i := 1; i < col; i++ {
		v := p.At(0, i)
		if v > maxv {
			maxi, maxv = i, v
		}
	}
	return maxi
}

func inference(x, w, b *mat.Dense) *mat.Dense {
	var res mat.Dense
	res.Mul(x, w)
	res.Apply(func(_, j int, v float64) float64 {
		return sigmoid(v + b.At(0, j))
	}, &res)
	return &res
}

func dCost(images, labels, p *mat.Dense) (dw, db *mat.Dense) {
	row, _ := p.Dims()
	diff := mat.NewDense(row, 10, nil)
	diff.Sub(p, labels)
	var gradW mat.Dense
	gradW.Mul(images.T(), diff)
	gradW.Scale(learningRateW/float64(row), &gradW)
	b := make([]float64, 10)
	for i := range b {
		b[i] = mat.Sum(diff.ColView(i))
	}
	db = mat.NewDense(1, 10, b)
	db.Scale(learningRateB/float64(row), db)
	return &gradW, db
}

func accuracy(images, labels *mat.Dense, w, b *mat.Dense) float64 {
	correct, all := 0, 0
	nImages, _ := images.Dims()
	for i := 0; i < nImages; i++ {
		image, label := mat.Row(nil, i, images), mat.Row(nil, i, labels)
		p := predict(mat.NewDense(1, len(image), image), w, b)
		if p == maxInd(label) {
			correct++
		}
		all++
	}
	return float64(correct) / float64(all)
}

func maxInd(s []float64) int {
	maxi, maxv := 0, s[0]
	for i, v := range s {
		if v > maxv {
			maxi, maxv = i, v
		}
	}
	return maxi
}
