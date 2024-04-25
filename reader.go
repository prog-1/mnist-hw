package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
)

func ReadImages(filePath string) []byte {
	file, err := os.Open(filePath)
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
		log.Fatal(fmt.Errorf("magic = %v, wantMagic  = %v", magic, wantMagic))
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
	return images
}

func ReadLabels(filePath string) []byte {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	var magic uint32
	const wantMagic = 0x00000801
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		log.Fatal(err)
	} else if magic != wantMagic {
		log.Fatal(fmt.Errorf("magic = %v, wantMagic  = %v", magic, wantMagic))
	}
	var nImages uint32
	if err := binary.Read(reader, binary.BigEndian, &nImages); err != nil {
		log.Fatal(err)
	}
	images := make([]byte, nImages)
	if _, err := io.ReadFull(reader, images); err != nil {
		log.Fatal(err)
	}
	return images
}

func DebugPrint(pixels []byte, size uint32) {
	var b []byte
	for i, p := range pixels {
		if uint32(i)%size == 0 {
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
