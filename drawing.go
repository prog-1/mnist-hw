
// How to implement on-screen drawing:
// 1. Output the pixels on the screen
// 2. On LNB click, get the coord of the mouse
// 3. Color the pixels
// 3.1. Color the pixel under the cursor white
// 3.2. Color the adjacent pixels gray
// 3.2.1. Add coef. that is responsible for how many adjacent pixels will be coloured grey(uint gp - gray pixels), e.g. 2
// 3.2.2. Setting the adjacent(optional)
/*
	gpc := 2 // gray pixel count
	wp := getCursorPos() // white pixel
	setGrayScale(wp, 100)

	// Recursively colours the pixel and its adjuscent ones while index <= gpc
	colourise := func(pixel, index) {
		for i := index; i <= gp; i++ {
			gs := 255 / (i+1) // gray scale value from 0 to 255
			setGrayScale({cp.x-1, cp.y-1}, gs)
			setGrayScale({cp.x+1, cp.y-1}, gs)
			setGrayScale({cp.x-1, cp.y+1}, gs)
			setGrayScale({cp.x+1, cp.y+1}, gs)
		}
	}
	// TODO: Finish it
*/
// 4. Store pixels in the byte array in the row-major order
