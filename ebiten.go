package main

import (
	"fmt"
	"image/color"
	"log"
	"os"
	"os/exec"
	"runtime"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"gonum.org/v1/gonum/mat"
)

const (
	rowCount, colCount = 28, 28 // TODO: Make it dynamic
	title              = "digit-recognition"
	strokeWidth        = 2
)

type App struct {
	W, B                   *mat.Dense // Weights and biases
	guess                  string     // Digit that model thinks is currently drawn
	mousePrevX, mousePrevY int
	screenBuffer           *ebiten.Image
}

func NewGame(physicalSide int, w, b *mat.Dense) *App {
	return &App{
		w, b,
		" ",
		-1, -1,
		ebiten.NewImage(physicalSide, physicalSide),
	}
}

func (a *App) Layout(outWidth, outHeight int) (logicalWidth, logicalHeight int) {
	return colCount, rowCount
}

func (a *App) Update() error {
	if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		alphas := a.pixelAlphas()
		fmt.Println(alphas.Dims())
		a.guess = fmt.Sprint(convertPredictions(inference(alphas, a.W, a.B)).At(0, 0))
	} else if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		a.handleDrawing()
	} else if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight) {
		a.screenBuffer.Clear()
		a.resetMousePrevCoord()
		clearConsole()
	}
	a.updateMousePrevCoord()
	return nil
}

func (a *App) Draw(screen *ebiten.Image) {
	screen.DrawImage(a.screenBuffer, &ebiten.DrawImageOptions{})
	ebitenutil.DebugPrint(screen, a.guess)
}

func clearConsole() {
	// Source: https://stackoverflow.com/questions/22891644/how-can-i-clear-the-terminal-screen-in-go
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "linux":
		cmd = exec.Command("clear") // WARNING: Untested by me
	case "windows":
		cmd = exec.Command("cmd", "/c", "cls")
	default:
		return
	}
	cmd.Stdout = os.Stdout
	cmd.Run()
}

func (a *App) updateMousePrevCoord() {
	a.mousePrevX, a.mousePrevY = ebiten.CursorPosition()
}

func (a *App) resetMousePrevCoord() {
	a.mousePrevX, a.mousePrevY = -1, -1
}

// Colours pixels under the cursor with shades of gray when LMB is pressed
// using cursor position from the previous frame(pass -1, -1 if it's the first frame).
func (a *App) handleDrawing() {
	x, y := ebiten.CursorPosition()
	if a.mousePrevX == -1 && a.mousePrevY == -1 {
		a.screenBuffer.Set(x, y, color.White)
	} else {
		vector.StrokeLine(a.screenBuffer, float32(a.mousePrevX), float32(a.mousePrevY), float32(x), float32(y), strokeWidth, color.White, true)
	}
}

// Returns alpha values of logical pixels on the screen. Dims: 1 x 784
func (a *App) pixelAlphas() *mat.Dense {
	alphas := make([]byte, rowCount*colCount)
	for r := 0; r < rowCount; r++ {
		for c := 0; c < colCount; c++ {
			// Type assertions - https://go.dev/tour/methods/15
			alphas[r*colCount+c] = a.screenBuffer.At(c, r).(color.RGBA).A
		}
	}
	return bytesToMat(1, rowCount*colCount, alphas)
}

// Initialises app and runs main loop that handles drawing on screen.
// Panics if image resolution is not square.
// The variable physicalSide is either screen width or height in physical pixels(width = height).
func RunDrawing(physicalSide int, w, b *mat.Dense) {
	ebiten.SetWindowSize(physicalSide, physicalSide)
	ebiten.SetWindowTitle(title)
	if err := ebiten.RunGame(NewGame(physicalSide, w, b)); err != nil {
		log.Fatal(err)
	}
}
