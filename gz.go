package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
)

const (

	//proper format of magic number and labels
	wantMagic = 0x00000803
	wantLabel = 0x00000801
)

// Reads MNIST image data from a .gz file
func readGZ(filename string) []byte {

	//Opening .gz file
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	//Initializing reader
	gzReader, err := gzip.NewReader(file)
	if err != nil {
		log.Fatal(err)
	}
	defer gzReader.Close()

	//Parsing magic number
	var magic uint32
	if err := binary.Read(gzReader, binary.BigEndian, &magic); err != nil {
		log.Fatal(err)
	} else if magic != wantMagic {
		fmt.Printf("invalid magic number %v", magic)
		return nil
	}

	//Parsing image count
	var pixelCount uint32
	if err := binary.Read(gzReader, binary.BigEndian, &pixelCount); err != nil {
		log.Fatal(err)
	}

	//Parsing image rows & columns
	var rows, columns uint32
	if err := binary.Read(gzReader, binary.BigEndian, &rows); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(gzReader, binary.BigEndian, &columns); err != nil {
		log.Fatal(err)
	}

	// Parsing all image data into single array of pixels
	pixels := make([]byte, pixelCount)

	if _, err := io.ReadFull(gzReader, pixels); err != nil {
		log.Fatal(err)
	}

	return pixels
}

// printing one data set digit in the console
func printMnist(pixels []byte) string {
	var b []byte
	for i, p := range pixels {
		if uint32(i)%columns == 0 {
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
	return string(b)
}
