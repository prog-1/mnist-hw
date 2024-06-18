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
	xTrain, yTrain, xTest, yTest := MnistDataFromFile("data/t10k-images.idx3-ubyte"), MnistDataFromFile("data/t10k-labels.idx1-ubyte"), MnistDataFromFile("data/train-images.idx3-ubyte"), MnistDataFromFile("data/train-labels.idx1-ubyte")
	w, b, err := train(epochCount, xTrain, yTrain, lrw, lrb, nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(accuracy(xTest, yTest, w, b))
	RunDrawing(w, b)
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

func MnistDataFromFile(path string) *mat.Dense {
	f, err1 := os.Open(path)
	if err1 != nil {
		panic(fmt.Errorf("failed to open file %q: %q", path, err1))
	}
	defer f.Close()
	data, err2 := MnistDataFromReader(f)
	if err2 != nil {
		panic(fmt.Errorf("failed to read file %q: %q", path, err2))
	}
	return data
}
