package main

import (
	"os"
	"os/exec"
	"runtime"
)

func main() {
	m, err := ReadMnistDB("data/t10k-images.idx3-ubyte")
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
