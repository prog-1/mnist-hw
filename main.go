package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
)

const (
	epochCount = 100
	lrw, lrb   = 1e-3, 0.5
)

func main() {
	xTrain, yTrain, xTest, yTest := MnistDataByPath("data/t10k-images.idx3-ubyte"), MnistDataByPath("data/t10k-labels.idx1-ubyte"), MnistDataByPath("data/train-images.idx3-ubyte"), MnistDataByPath("data/train-labels.idx1-ubyte")
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
