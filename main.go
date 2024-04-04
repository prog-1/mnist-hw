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
	xTrain, yTrain := ReadMnistDB("data/t10k-images.idx3-ubyte"), ReadMnistDB("data/t10k-labels.idx1-ubyte")
	sink := func(epoch int, w, dw, b, db *mat.Dense) {
		// TODO: Print accuracy
		if epoch%10 == 0 {
			fmt.Printf("Epoch: %v\n\n", epoch)
			fmt.Printf("Derivatives:\nws = %v\nb = %v\n\n", dw, db)
			fmt.Println()
		}
	}
	_, _, err := train(epochCount, xTrain.Pixels, yTrain.Pixels, lrw, lrb, sink)
	if err != nil {
		panic(err)
	}
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

func ReadMnistDB(path string) *mnistImages {
	f, err1 := os.Open(path)
	if err1 != nil {
		panic(fmt.Sprintf("failed to open file %q: %q", path, err1))
	}
	defer f.Close()
	m, err2 := ReadMnistImages(f)
	if err2 != nil {
		panic(fmt.Sprintf("failed to read file %q: %q", path, err2))
	}
	return m
}
