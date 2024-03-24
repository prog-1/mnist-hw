package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
)

type mnistImage struct {
	Len, Rows, Cols uint32
	Pixels          []byte
}

const (
	filename  = "C:/Users/vovau/Desktop/Programming/mnist-hw/archive/t10k-images.idx3-ubyte"
	imageSize = 28 * 28
)

func readMnist() error {
	var magic uint32 = 0x00000803
	var mnistmagic uint32
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	reader := bufio.NewReader(file)

	if err := binary.Read(reader, binary.BigEndian, &mnistmagic); err != nil {
		log.Fatal(err)
	} else if magic != mnistmagic {
		log.Fatal(fmt.Errorf("magic = %v, wantMagic  = %v", magic, mnistmagic))
	}
	var nImages, nRows, nCols uint32
	if err := binary.Read(reader, binary.BigEndian, &nImages); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(reader, binary.BigEndian, &nRows); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(reader, binary.BigEndian, &nCols); err != nil {
		log.Fatal(err)
	}
	pixels := make([]byte, nImages*nRows*nCols)
	if _, err := io.ReadFull(reader, pixels); err != nil {
		log.Fatal(err)
	}
	draw(mnistImage{nImages, nRows, nCols, pixels})
	return nil
}

func draw(img mnistImage) {
	var b []byte
	n := 0
	pixels := img.Pixels[n*imageSize : (n+1)*imageSize]
	for i, p := range pixels {
		if uint32(i)%28 == 0 {
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
	fmt.Print(string(b))
}

func main() {
	readMnist()
}
