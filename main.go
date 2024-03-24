package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"image/color"
	"io"
	"log"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const (
	screenWidth, screenHeight = 500, 500
	wantMagic                 = 0x00000803
	rows, cols                = 28, 28
)

type Num struct {
	screen     *ebiten.Image
	posX, posY int
}

func readImages(data string) ([][]byte, error) {
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
	images := make([][]byte, nImages)
	for i := uint32(0); i < nImages; i++ {
		image := make([]byte, nRows*nCols)
		_, err := io.ReadFull(gz, image)
		if err != nil {
			return nil, err
		}
		images[i] = image
	}

	return images, nil
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
		fmt.Println(n.PrintScreen())
	}
	if ebiten.IsKeyPressed(ebiten.KeyC) {
		n.ClearScreen()
	}
	if ebiten.IsKeyPressed(ebiten.KeyQ) {
		os.Exit(0)
	}

	return nil
}

func (n *Num) PrintScreen() string {
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	var builder strings.Builder
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			colorr := n.screen.At(x, y).(color.RGBA)
			if colorr == white {
				builder.WriteRune('#')
			} else {
				builder.WriteRune(' ')
			}
		}
		builder.WriteRune('\n')
	}

	return builder.String()
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

func NewNum() *Num {
	return &Num{ebiten.NewImage(rows, cols), 0, 0}
}

func printNumber(image []byte) {
	for i, pixel := range image {
		if i%cols == 0 {
			fmt.Println()
		}
		if pixel > 128 {
			fmt.Print("#")
		} else {
			fmt.Print(" ")
		}
	}
	fmt.Println()
}

func main() {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	data := "data/train-images-idx3-ubyte.gz"
	images, err := readImages(data)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	ind := 0
	printNumber(images[ind])
	if err := ebiten.RunGame(NewNum()); err != nil {
		log.Fatal(err)
	}
}
