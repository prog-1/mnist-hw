package main

import (
	"bufio"
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

	"github.com/hajimehoshi/ebiten/inpututil"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"gonum.org/v1/gonum/mat"
)

const (
	screenWidth  = 640
	screenHeight = 480
	epochs       = 100000
	lrW, lrB     = 0.01, 0.8
	rows, cols   = 28, 28
)

type Game struct {
	screen *ebiten.Image
	image  []float64
	x, y   int
	w, b   *mat.Dense
}

// func ReadLabels(r io.Reader) ([]byte, error) {
// 	const wantMagic = 0x00000801 // 08 01 means ubyte 1 dimension, it's mandatory
// 	var magic uint32
// 	err := binary.Read(r, binary.BigEndian, &magic)
// 	if err != nil {
// 		log.Fatal("Failed to read magic number:", err)
// 	} else if magic != wantMagic {
// 		log.Fatal("Invalid magic number %x, want %x", magic, wantMagic)
// 	}
// 	var labelsNum uint32
// 	err = binary.Read(r, binary.BigEndian, &labelsNum)
// 	if err != nil {
// 		log.Fatal("Cannot parse binary file:", err)
// 	}
// 	res := make([]byte, labelsNum)
// 	n, err := io.ReadFull(r, res)
// 	if n != len(res) {
// 		log.Fatal("Length of data is not correct")
// 	}
// 	a := make([]float64, labelsNum*10)
// 	for i, v := range res {
// 		a[i*10+int(v)] = 1
// 	}
// 	return nil, nil mat.NewDense(int(labelsNum), 10, a)
// }

func DebugPrint(columns uint32, pixels []byte) string {
	var b []byte
	for i, p := range pixels {
		if uint32(i)%columns == 0 {
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
	return string(b)
}

// func ReadLabels(filename string) ([]byte, error) {
// 	file, err := os.Open(filename)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	defer file.Close()
// 	reader := bufio.NewReader(file)
// 	const wantMagic = 0x00000801 // 08 01 means ubyte 1 dimension, it's mandatory
// 	var magic uint32
// 	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
// 		log.Fatal(err)
// 	} else if magic != wantMagic {
// 		log.Fatal(fmt.Printf("Invalid magic number %x, want %x", magic, wantMagic))
// 	}
// 	var nImages, rows, columns uint32
// 	if err := binary.Read(reader, binary.BigEndian, &nImages); err != nil {
// 		log.Fatal(err)
// 	}
// 	if err := binary.Read(reader, binary.BigEndian, &rows); err != nil {
// 		log.Fatal(err)
// 	}
// 	if err := binary.Read(reader, binary.BigEndian, &columns); err != nil {
// 		log.Fatal(err)
// 	}
// 	images := make([]byte, nImages*rows*columns)
// 	if _, err := io.ReadFull(reader, images); err != nil {
// 		log.Fatal(err)
// 	}
// 	return images, nil
// }

func ReadLabels(filename string) (*mat.Dense, error) {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		log.Fatal(err)
	}
	defer gz.Close()

	const wantMagic = 0x00000801 // 08 01 means ubyte 1 dimension, it's mandatory
	var magic uint32
	err = binary.Read(gz, binary.BigEndian, &magic)
	if err != nil {
		log.Fatal(err)
	}
	if magic != wantMagic {
		log.Fatal(fmt.Printf("Invalid magic number %x, want %x", magic, wantMagic))
	}
	var nLabels uint32
	err = binary.Read(gz, binary.BigEndian, &nLabels)
	if err != nil {
		log.Fatal(err)
	}
	labels := make([]byte, nLabels)
	_, err = io.ReadFull(gz, labels)
	if err != nil {
		log.Fatal(err)
	}
	a := make([]float64, nLabels*10)
	for i, v := range labels {
		a[i*10+int(v)] = 1
	}
	return mat.NewDense(int(nLabels), 10, a), nil
}

func ReadImages(filename string) ([]byte, error) {
	file, err := os.Open(filename)
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
		log.Fatal(fmt.Printf("Invalid magic number %x, want %x", magic, wantMagic))
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
	return images, nil
}

func (g *Game) UpdateWeights(w, b *mat.Dense) {
	g.w, g.b = w, b
}

func (g *Game) Update() error {
	if inpututil.IsMouseButtonJustPressed(0) { // left mouse button
		x, y := ebiten.CursorPosition()
		g.screen.Set(x, y, color.White)
		g.x, g.y = x, y
	} else if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		x, y := ebiten.CursorPosition()
		if x != g.x || y != g.y {
			vector.StrokeLine(g.screen, float32(g.x), float32(g.y), float32(x), float32(y), 2, color.White, true)
			g.x = x
			g.y = y
		}
	} else if inpututil.IsMouseButtonJustReleased(0) { // left mouse button
		fmt.Println(g.printScreen(g.w, g.b))
	}
	if ebiten.IsKeyPressed(ebiten.KeyC) {
		g.clearScreen()
	}
	if ebiten.IsKeyPressed(ebiten.KeyQ) {
		os.Exit(0)
	}
	return nil
}

func (g *Game) printScreen(w, b *mat.Dense) (string, int) {
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	var builder strings.Builder
	g.image = make([]float64, 0, 784)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			colorr := g.screen.At(x, y).(color.RGBA)
			g.image = append(g.image, float64(colorr.A)/255)
			if colorr == white {
				builder.WriteRune('#')
			} else {
				builder.WriteRune(' ')
			}
		}
		builder.WriteRune('\n')
	}
	return builder.String(), predict(mat.NewDense(1, 784, g.image), w, b)
}

func (g *Game) clearScreen() {
	g.screen = ebiten.NewImage(rows, cols)
	if runtime.GOOS == "windows" {
		cmd := exec.Command("cmd", "/c", "cls")
		cmd.Stdout = os.Stdout
		cmd.Run()
	}
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.DrawImage(g.screen, nil)
}

func (g *Game) Layout(width, height int) (int, int) {
	return rows, cols
}

func main() {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	timages, err := ReadImages("data/train-images-idx3-ubyte.gz")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	timagesF := make([]float64, len(timages))
	for i := range timages {
		timagesF[i] = float64(timages[i])
	}
	matrixTImages := mat.NewDense(len(timages)/784, 784, timagesF)
	images, err := ReadImages("data/t10k-images-idx3-ubyte.gz")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	imagesF := make([]float64, len(images))
	for i := 0; i < len(images); i++ {
		imagesF[i] = float64(images[i])
	}
	matrixImages := mat.NewDense(len(images)/784, 784, imagesF)
	tlabels, err := ReadLabels("data/train-labels-idx1-ubyte.gz")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	labels, err := ReadLabels("data/t10k-labels-idx1-ubyte.gz")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	w := mat.NewDense(784, 10, make([]float64, 7840))
	b := mat.NewDense(1, 10, make([]float64, 10))
	for i := 0; i < epochs; i++ {
		p := inference(matrixTImages, w, b)
		dw, db := cost(matrixTImages, tlabels, p)
		w.Sub(w, dw)
		b.Sub(b, db)
	}
	fmt.Println(accuracy(matrixImages, labels, w, b))
	g := &Game{ebiten.NewImage(rows, cols), []float64{}, 0, 0, w, b}
	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
	g.UpdateWeights(w, b)
}

func Sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
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
		return Sigmoid(v + b.At(0, j))
	}, &res)
	return &res
}

func cost(images, labels, p *mat.Dense) (dw, db *mat.Dense) {
	row, _ := p.Dims()
	diff := mat.NewDense(row, 10, nil)
	diff.Sub(p, labels)
	var gradW mat.Dense
	gradW.Mul(images.T(), diff)
	gradW.Scale(lrW/float64(row), &gradW)
	b := make([]float64, 10)
	for i := range b {
		b[i] = mat.Sum(diff.ColView(i))
	}
	db = mat.NewDense(1, 10, b)
	db.Scale(lrB/float64(row), db)
	return &gradW, db
}

func accuracy(images, labels *mat.Dense, w, b *mat.Dense) float64 {
	var correct, all int
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
	maxI, maxV := 0, s[0]
	for i, v := range s {
		if v > maxV {
			maxI, maxV = i, v
		}
	}
	return maxI
}
