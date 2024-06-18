package main

import (
	"bytes"
	"encoding/binary"
	"io"
	"testing"
)

// What's the point of this stuff?
// I want to test whether my MnistDataFromReader
// reads data from a reader properly
// How am I going to do this?
// Errors:
// 1. Catch error with invalid magic number
// 2. Catch error if
// I should check whether the matrix we return
// How can I test this?

func TestMnist(t *testing.T) {
	for _, tc := range []struct {
	}{
		// Valid image file.
		// Valid label file.
		// File with an incorrect magic number.
		// File with incomplete content, i.e.:
		// 1. count * rows * cols != len(data)
		// 2. no magic number
		// 3. Image file data with abscent rows or cols
		// 4. not enough data
		// Empty file
	} {

	}
}

func GenMnistMockData(magic, count, rows, cols uint32) io.Reader {
	mockFile := new(bytes.Buffer)

	writeToMockFile := func(value uint32) {
		if err := binary.Write(mockFile, binary.BigEndian, value); err != nil {
			panic(err)
		}
	}

	writeToMockFile(magic)
	writeToMockFile(count)

	if magic == 0x803 /*images*/ {
		writeToMockFile(rows)
		writeToMockFile(cols)
	}

	data := make([]byte, count*rows*cols)
	for i := range data {
		// Any value will do, we only check for count
		data[i] = byte(5)
	}
	if _, err := mockFile.Write(data); err != nil {
		panic(err)
	}

	return mockFile
}
