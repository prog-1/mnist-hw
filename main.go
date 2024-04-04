package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"

	"gonum.org/v1/gonum/mat"
)

const (
	epochCount = 100
	lrw, lrb   = 1e-3, 0.5
)

func main() {
	xTrain, yTrain, xTest, yTest := MnistImages("data/t10k-images.idx3-ubyte"), MnistLabels("data/t10k-labels.idx1-ubyte"), MnistImages("data/train-images.idx3-ubyte"), MnistLabels("data/train-labels.idx1-ubyte")
	w, b, err := train(epochCount, xTrain, yTrain, lrw, lrb, nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(accuracy(xTest, yTest, w, b))
}

func ClearConsole() {
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

func MnistImages(path string) *mat.Dense {
	f, err1 := os.Open(path)
	if err1 != nil {
		panic(fmt.Sprintf("failed to open file %q: %q", path, err1))
	}
	defer f.Close()
	images, err2 := ReadMnistImages(f)
	if err2 != nil {
		panic(fmt.Sprintf("failed to read file %q: %q", path, err2))
	}
	return images.Pixels
}

func MnistLabels(path string) *mat.Dense {
	f, err1 := os.Open(path)
	if err1 != nil {
		panic(fmt.Sprintf("failed to open file %q: %q", path, err1))
	}
	defer f.Close()
	images, err2 := ReadMnistLabels(f)
	if err2 != nil {
		panic(fmt.Sprintf("failed to read file %q: %q", path, err2))
	}
	return images.Labels
}
