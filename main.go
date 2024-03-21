package main

import "fmt"

func main() {
	// printImageFromMnistDatabase("data/t10k-images.idx3-ubyte", 3)
	const (
		rows, cols = 28, 28
	)
	pixels := make([]byte, rows, cols)
	fmt.Println(pixels)
}
