package main

import (
	"bufio"
	"encoding/gob"
	"flag"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg" // Register JPEG decoder
	"image/png"    // Register PNG decoder
	"io"
	"log"
	"math"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	neighborhoodSizeLR = 3
	inputKeySize       = neighborhoodSizeLR * neighborhoodSizeLR
	outputBlockSizeHR  = 3
	outputValueSize    = outputBlockSizeHR * outputBlockSizeHR
	treshold           = 50

	modelFileNameDefault = "unblur_denoice_shared_model_x3_20.gob"
	serverAbsModelPath   = "/home/unblur_denoice_shared_model_x3_20.gob"
	VALUE_DIVIDER        = 20
)

type ChannelModel map[[inputKeySize]uint8][outputValueSize + 2]uint32

type FullModel struct {
	SharedChannelData  ChannelModel
	SharedChannelData2 ChannelModel
}

func newFullModel() *FullModel {
	return &FullModel{
		SharedChannelData:  make(ChannelModel),
		SharedChannelData2: make(ChannelModel),
	}
}

// Helper to load image and convert to NRGBA with origin at (0,0)
func loadImageAsNRGBA(path string) (*image.NRGBA, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open image %s: %w", path, err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image %s: %w", path, err)
	}
	// log.Printf("Loaded image %s (format: %s, original bounds: %v)\n", path, format, img.Bounds())

	bounds := img.Bounds()
	nrgbaImg := image.NewNRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	draw.Draw(nrgbaImg, nrgbaImg.Bounds(), img, bounds.Min, draw.Src)
	return nrgbaImg, nil
}

// saveImageOptimized allows specifying a compression level.
// For maximum speed, use png.NoCompression or png.BestSpeed.
func saveImageOptimized(img image.Image, path string, compressionLevel png.CompressionLevel) error {
	outFile, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create output file %s: %w", path, err)
	}
	defer outFile.Close()

	bufferedWriter := bufio.NewWriter(outFile)

	encoder := png.Encoder{CompressionLevel: compressionLevel}
	err = encoder.Encode(bufferedWriter, img)
	if err != nil {
		return fmt.Errorf("failed to encode image to %s: %w", path, err)
	}

	err = bufferedWriter.Flush()
	if err != nil {
		return fmt.Errorf("failed to flush buffer for %s: %w", path, err)
	}

	log.Printf("Saved image to %s (compression: %v)\n", path, compressionLevel)
	return nil
}

func applyDivider(a uint8) uint8 {
	if a < 20 {
		return 0
	}
	return a / 20
}

func extractLRNeighborhood(img *image.NRGBA, centerX, centerY, channelOffset int) [inputKeySize]uint8 {
	var neighborhood [inputKeySize]uint8
	idx := 0
	bounds := img.Bounds()

	var centerVal uint8 = img.Pix[img.PixOffset(centerX, centerY)+channelOffset]
	for yOffset := -1; yOffset <= 1; yOffset++ {
		for xOffset := -1; xOffset <= 1; xOffset++ {
			currentX, currentY := centerX+xOffset, centerY+yOffset
			var v uint8 = 0
			if currentX >= 0 && currentX < bounds.Dx() &&
				currentY >= 0 && currentY < bounds.Dy() {
				v = applyDivider(img.Pix[img.PixOffset(currentX, currentY)+channelOffset])
			} else {
				v = applyDivider(centerVal)
			}
			neighborhood[idx] = v
			idx++
		}
	}

	return neighborhood
}

func extractLRNeighborhoodOriginal(img *image.NRGBA, centerX, centerY, channelOffset int) [inputKeySize]uint8 {
	var neighborhood [inputKeySize]uint8
	idx := 0
	bounds := img.Bounds()

	var centerVal uint8 = img.Pix[img.PixOffset(centerX, centerY)+channelOffset]
	for yOffset := -1; yOffset <= 1; yOffset++ {
		for xOffset := -1; xOffset <= 1; xOffset++ {
			currentX, currentY := centerX+xOffset, centerY+yOffset
			var v uint8 = 0
			if currentX >= 0 && currentX < bounds.Dx() &&
				currentY >= 0 && currentY < bounds.Dy() {
				v = img.Pix[img.PixOffset(currentX, currentY)+channelOffset]
			} else {
				v = centerVal
			}
			neighborhood[idx] = v
			idx++
		}
	}

	return neighborhood
}

var globalTrainedModel *FullModel

func train(lrImgPath, hrImgPath, modelPath string, saveModelAfter bool) error {
	// not awailable in free version, contact me for full version and more models
	// training models for any task: unblur, denoise, upscale 2x, 3x, 4x, chromatic aberation fix etc
	return nil
}

func stepDown(lrNeighborhood [inputKeySize]uint8, step float64) [inputKeySize]uint8 {
	var new [inputKeySize]uint8
	for key, val := range lrNeighborhood {
		new[key] = uint8(math.Round((float64(val) / step)))
	}
	return new
}

func stepUpHR(lrNeighborhood [outputValueSize + 2]uint32, step float64) ([outputValueSize + 2]uint32, bool) {
	newVal := [outputValueSize + 2]uint32{}
	for i := 0; i <= 8; i++ {
		val := float64(lrNeighborhood[i]) / float64(lrNeighborhood[9]) * step
		if val > 255 {
			val = 255
			return [outputValueSize + 2]uint32{}, false
		}
		if val < 0 {
			val = 0
			return [outputValueSize + 2]uint32{}, false
		}
		newVal[i] = uint32(math.Round(float64(lrNeighborhood[i]) * step))
	}
	newVal[9] = lrNeighborhood[9]
	newVal[10] = uint32(math.Round(float64(lrNeighborhood[10]) * step))
	return newVal, true
}

var countNF = 0

func findOtherPath2(lrNeighborhood [inputKeySize]uint8, lrNeighborhood0 [inputKeySize]uint8) ([outputValueSize + 2]uint32, bool) {
	var step float64 = 0.005
	var out [outputValueSize + 2]uint32 = [outputValueSize + 2]uint32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	for i := 1; i <= 800; i++ {
		lrNeighborhood2 := stepDown(lrNeighborhood, 1+step*float64(i))
		hrBlock, found, _, _ := findPrimaryWithRotation(lrNeighborhood2)
		if found {
			new, valid := stepUpHR(hrBlock, 1+step*float64(i))
			if valid && new[9] > 0 {
				out = rAdd(out, new)
			}
		}
		if out[9] > treshold {
			return out, true
		}
	}

	var i uint8
	for i = 1; i < 6; i++ {
		lrNeighborhood2 := [inputKeySize]uint8{
			max(i, min(10-i, lrNeighborhood[0])),
			max(i, min(10-i, lrNeighborhood[1])),
			max(i, min(10-i, lrNeighborhood[2])),
			max(i, min(10-i, lrNeighborhood[3])),
			max(i, min(10-i, lrNeighborhood[4])),
			max(i, min(10-i, lrNeighborhood[5])),
			max(i, min(10-i, lrNeighborhood[6])),
			max(i, min(10-i, lrNeighborhood[7])),
			max(i, min(10-i, lrNeighborhood[8])),
		}
		hrBlock, found, _, _ := findPrimaryWithRotation(lrNeighborhood2)
		if found {
			out = rAdd(out, hrBlock)
		}
		if out[9] > treshold {
			return out, true
		}
	}

	if out[9] == 0 {
		countNF += 1
	}

	return out, false
}

func findPrimaryWithRotation(lrNeighborhood [inputKeySize]uint8) ([outputValueSize + 2]uint32, bool, int, int) {
	a, b, c := _findPrimaryWithRotation(lrNeighborhood)
	if b {
		return a, b, c, 0
	}

	// flip horizontaly
	var lrNeighborhoodFH [inputKeySize]uint8 = [inputKeySize]uint8{
		lrNeighborhood[2], lrNeighborhood[1], lrNeighborhood[0],
		lrNeighborhood[5], lrNeighborhood[4], lrNeighborhood[3],
		lrNeighborhood[8], lrNeighborhood[7], lrNeighborhood[6],
	}

	a, b, c = _findPrimaryWithRotation(lrNeighborhoodFH)
	if b {
		return [outputValueSize + 2]uint32{
			a[2], a[1], a[0],
			a[5], a[4], a[3],
			a[8], a[7], a[6],
			a[9], a[10],
		}, b, c, 1
	}
	// flip verticaly
	var lrNeighborhoodFV [inputKeySize]uint8 = [inputKeySize]uint8{
		lrNeighborhood[6], lrNeighborhood[7], lrNeighborhood[8],
		lrNeighborhood[3], lrNeighborhood[4], lrNeighborhood[5],
		lrNeighborhood[0], lrNeighborhood[1], lrNeighborhood[2],
	}
	a, b, c = _findPrimaryWithRotation(lrNeighborhoodFV)
	if b {
		return [outputValueSize + 2]uint32{
			a[6], a[7], a[8],
			a[3], a[4], a[5],
			a[0], a[1], a[2],
			a[9], a[10],
		}, b, c, 2
	}
	return [outputValueSize + 2]uint32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1000, 1000}, false, 0, 0
}

func _findPrimaryWithRotation(lrNeighborhood [inputKeySize]uint8) ([outputValueSize + 2]uint32, bool, int) {
	modelToUse := globalTrainedModel

	/*
	 1 2 3
	 4 5 6
	 7 8 9

	 3 6 9
	 2 5 8
	 1 4 7

	 9 8 7
	 6 5 4
	 3 2 1

	 7 4 1
	 8 5 2
	 9 6 3
	*/

	hrBlock1, found1 := modelToUse.SharedChannelData[lrNeighborhood]
	if found1 {
		return hrBlock1, true, 0
	}

	lrNeighborhood2 := [inputKeySize]uint8{
		lrNeighborhood[2],
		lrNeighborhood[5],
		lrNeighborhood[8],
		lrNeighborhood[1],
		lrNeighborhood[4],
		lrNeighborhood[7],
		lrNeighborhood[0],
		lrNeighborhood[3],
		lrNeighborhood[6],
	}
	hrBlock2, found2 := modelToUse.SharedChannelData[lrNeighborhood2]
	if found2 {
		return [outputValueSize + 2]uint32{
			hrBlock2[6],
			hrBlock2[3],
			hrBlock2[0],
			hrBlock2[7],
			hrBlock2[4],
			hrBlock2[1],
			hrBlock2[8],
			hrBlock2[5],
			hrBlock2[2],

			hrBlock2[9],
			hrBlock2[10],
		}, true, 1
	}

	lrNeighborhood3 := [inputKeySize]uint8{
		lrNeighborhood[8],
		lrNeighborhood[7],
		lrNeighborhood[6],
		lrNeighborhood[5],
		lrNeighborhood[4],
		lrNeighborhood[3],
		lrNeighborhood[2],
		lrNeighborhood[1],
		lrNeighborhood[0],
	}
	hrBlock3, found3 := modelToUse.SharedChannelData[lrNeighborhood3]
	if found3 {
		return [outputValueSize + 2]uint32{
			hrBlock3[8],
			hrBlock3[7],
			hrBlock3[6],
			hrBlock3[5],
			hrBlock3[4],
			hrBlock3[3],
			hrBlock3[2],
			hrBlock3[1],
			hrBlock3[0],

			hrBlock3[9],
			hrBlock3[10],
		}, true, 2
	}

	lrNeighborhood4 := [inputKeySize]uint8{
		lrNeighborhood[6],
		lrNeighborhood[3],
		lrNeighborhood[0],
		lrNeighborhood[7],
		lrNeighborhood[4],
		lrNeighborhood[1],
		lrNeighborhood[8],
		lrNeighborhood[5],
		lrNeighborhood[2],
	}
	hrBlock4, found4 := modelToUse.SharedChannelData[lrNeighborhood4]
	if found4 {
		return [outputValueSize + 2]uint32{
			hrBlock4[2],
			hrBlock4[5],
			hrBlock4[8],
			hrBlock4[1],
			hrBlock4[4],
			hrBlock4[7],
			hrBlock4[0],
			hrBlock4[3],
			hrBlock4[6],

			hrBlock4[9],
			hrBlock4[10],
		}, true, 3
	}

	return [outputValueSize + 2]uint32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, false, -1
}

func rAdd(a [outputValueSize + 2]uint32, b [outputValueSize + 2]uint32) [outputValueSize + 2]uint32 {
	a[0] = a[0] + b[0]
	a[1] = a[1] + b[1]
	a[2] = a[2] + b[2]
	a[3] = a[3] + b[3]
	a[4] = a[4] + b[4]
	a[5] = a[5] + b[5]
	a[6] = a[6] + b[6]
	a[7] = a[7] + b[7]
	a[8] = a[8] + b[8]
	a[9] = a[9] + b[9]
	a[10] = a[10] + b[10]
	return a
}

func isFlat(lr [inputKeySize]uint8) bool {
	min, max := lr[0], lr[0]
	for _, v := range lr {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	if max-min < 2 {
		return true
	}

	return lr[0] == lr[1] &&
		lr[1] == lr[2] &&
		lr[2] == lr[3] &&
		lr[3] == lr[4] &&
		lr[4] == lr[5] &&
		lr[5] == lr[6] &&
		lr[6] == lr[7] &&
		lr[7] == lr[8]
}

func processColorCashe(lrImg *image.NRGBA, xLR int, yLR int, index int) [outputValueSize + 2]uint32 {
	return processColor(lrImg, xLR, yLR, index)
}

func processColor(lrImg *image.NRGBA, xLR int, yLR int, index int) [outputValueSize + 2]uint32 {
	lrNeighborhood := extractLRNeighborhood(lrImg, xLR, yLR, index)

	if isFlat(lrNeighborhood) {
		lrNeighborhood0 := extractLRNeighborhoodOriginal(lrImg, xLR, yLR, index)
		return simpleUpscaleCenterPixel(lrNeighborhood0)
	}

	hrBlock, found, _, _ := findPrimaryWithRotation(lrNeighborhood)
	if found && hrBlock[9] > treshold {
		return hrBlock
	}

	//hits++
	lrNeighborhood0 := extractLRNeighborhoodOriginal(lrImg, xLR, yLR, index)

	hrBlocknew2, found2 := findOtherPath2(lrNeighborhood, lrNeighborhood0)
	if found2 && hrBlocknew2[9] > treshold {
		return hrBlocknew2
	}

	return [outputValueSize + 2]uint32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
}

func upscale(lrImgPath, outputPath, modelPath string) error {
	start := time.Now()

	lrImg, err := loadImageAsNRGBA(lrImgPath)
	if err != nil {
		return fmt.Errorf("upscale: failed to load image: %w", err)
	}

	lrBounds := lrImg.Bounds()
	lrWidth := lrBounds.Dx()
	lrHeight := lrBounds.Dy()

	hrWidth := lrWidth * outputBlockSizeHR
	hrHeight := lrHeight * outputBlockSizeHR

	outputImg := image.NewNRGBA(image.Rect(0, 0, hrWidth, hrHeight))
	outputPix := outputImg.Pix
	outputStride := outputImg.Stride

	lrPix := lrImg.Pix
	lrStride := lrImg.Stride

	numWorkers := runtime.NumCPU()
	//numWorkers := 1
	var wg sync.WaitGroup

	rowsPerWorker := (lrHeight + numWorkers - 1) / numWorkers

	for i := 0; i < numWorkers; i++ {
		startYLR := i * rowsPerWorker
		endYLR := (i + 1) * rowsPerWorker
		if endYLR > lrHeight {
			endYLR = lrHeight
		}
		if startYLR >= endYLR { // No work for this worker
			continue
		}

		wg.Add(1)
		go func(startYLR, endYLR int) {
			defer wg.Done()

			var hrBlockR, hrBlockG, hrBlockB [11]uint32

			for yLR := startYLR; yLR < endYLR; yLR++ {
				for xLR := 0; xLR < lrWidth; xLR++ {
					hrBlockR = processColorCashe(lrImg, xLR, yLR, 0)
					hrBlockG = processColorCashe(lrImg, xLR, yLR, 1)
					hrBlockB = processColorCashe(lrImg, xLR, yLR, 2)

					lrPixOffset := yLR*lrStride + xLR*4
					centerLR_R := lrPix[lrPixOffset]
					centerLR_G := lrPix[lrPixOffset+1]
					centerLR_B := lrPix[lrPixOffset+2]
					// centerLR_A := lrPix[lrPixOffset+3] // Alpha, if needed

					outBaseX := xLR * outputBlockSizeHR
					outBaseY := yLR * outputBlockSizeHR

					for yOffset := 0; yOffset < outputBlockSizeHR; yOffset++ {
						for xOffset := 0; xOffset < outputBlockSizeHR; xOffset++ {
							valIdx := yOffset*outputBlockSizeHR + xOffset

							divisorR := float32(hrBlockR[outputBlockSizeHR*outputBlockSizeHR+1])
							divisorG := float32(hrBlockG[outputBlockSizeHR*outputBlockSizeHR+1])
							divisorB := float32(hrBlockB[outputBlockSizeHR*outputBlockSizeHR+1])

							// Avoid division by zero or very small numbers if +1 isn't enough
							if divisorR == 0 {
								divisorR = 1
							}
							if divisorG == 0 {
								divisorG = 1
							}
							if divisorB == 0 {
								divisorB = 1
							}
							var rValF float32 = 0
							var gValF float32 = 0
							var bValF float32 = 0

							rValF = float32(centerLR_R) * (float32(hrBlockR[valIdx]) / divisorR)
							gValF = float32(centerLR_G) * (float32(hrBlockG[valIdx]) / divisorG)
							bValF = float32(centerLR_B) * (float32(hrBlockB[valIdx]) / divisorB)

							var rOut, gOut, bOut uint8
							if rValF > 255 {
								rOut = 255
							} else if rValF < 0 {
								rOut = 0
							} else {
								rOut = uint8(rValF)
							}
							if gValF > 255 {
								gOut = 255
							} else if gValF < 0 {
								gOut = 0
							} else {
								gOut = uint8(gValF)
							}
							if bValF > 255 {
								bOut = 255
							} else if bValF < 0 {
								bOut = 0
							} else {
								bOut = uint8(bValF)
							}

							targetX := outBaseX + xOffset
							targetY := outBaseY + yOffset

							pixOffset := targetY*outputStride + targetX*4
							outputPix[pixOffset] = rOut
							outputPix[pixOffset+1] = gOut
							outputPix[pixOffset+2] = bOut
							outputPix[pixOffset+3] = 255 // Alpha
						}
					}
				}
			}
		}(startYLR, endYLR)
	}
	wg.Wait()

	duration := time.Since(start)
	fmt.Printf("Upscale function took %v\n", duration)

	return saveImageOptimized(outputImg, outputPath, png.NoCompression)
}

func simpleUpscaleCenterPixel(rn [inputKeySize]uint8) [outputValueSize + 2]uint32 {
	var block [outputValueSize + 2]uint32
	/*
		0 1 2
		3 4 5
		6 7 8
	*/
	block[0] = uint32((float32(rn[0]) + float32(rn[4]) + float32(rn[4])) / 3)
	block[2] = uint32((float32(rn[2]) + float32(rn[4]) + float32(rn[4])) / 3)
	block[6] = uint32((float32(rn[6]) + float32(rn[4]) + float32(rn[4])) / 3)
	block[8] = uint32((float32(rn[8]) + float32(rn[4]) + float32(rn[4])) / 3)
	block[7] = uint32((float32(rn[7]) + float32(rn[4]) + float32(rn[4])) / 3)
	block[5] = uint32((float32(rn[5]) + float32(rn[4]) + float32(rn[4])) / 3)
	block[1] = uint32((float32(rn[1]) + float32(rn[4]) + float32(rn[4])) / 3)
	block[3] = uint32((float32(rn[3]) + float32(rn[4]) + float32(rn[4])) / 3)
	block[4] = uint32(rn[4])
	block[9] = 1
	block[10] = uint32(rn[4])
	return block
}

func loadModelFromFile(path string) (*FullModel, bool, error) {
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("failed to open model file %s: %w", path, err)
	}
	defer file.Close()

	loadedModel := newFullModel()
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(loadedModel); err != nil {
		fileInfo, statErr := file.Stat()
		if statErr == nil && fileInfo.Size() == 0 {
			log.Printf("Model file %s is empty. Treating as new model.", path)
			return newFullModel(), false, nil
		}
		return nil, false, fmt.Errorf("failed to decode model from %s: %w", path, err)
	}
	return loadedModel, true, nil
}

// --- FFmpeg related utility functions ---
var ffmpegPath string
var ffprobePath string

func findExecutable(name string) (string, error) {
	if path, err := exec.LookPath(name); err == nil {
		return path, nil
	}

	return "", fmt.Errorf("%s not found in PATH", name)
}

func initFFmpegPaths() error {
	var errFFmpeg, errFFprobe error
	ffmpegPath, errFFmpeg = findExecutable("ffmpeg")
	if errFFmpeg != nil {
		return fmt.Errorf("ffmpeg not found: %w. Please ensure ffmpeg is installed and in your system PATH", errFFmpeg)
	}
	ffprobePath, errFFprobe = findExecutable("ffprobe")
	if errFFprobe != nil {
		return fmt.Errorf("ffprobe not found: %w. Please ensure ffprobe (part of ffmpeg) is installed and in your system PATH", errFFprobe)
	}
	return nil
}

type upscaleJob struct {
	lrFramePath    string
	hrFramePath    string
	modelPathToUse string
}

func upscaleWorker(id int, wg *sync.WaitGroup, jobs <-chan upscaleJob, errors chan<- error) {
	defer wg.Done()
	for job := range jobs {
		log.Printf("Worker %d: Starting upscale for %s\n", id, job.lrFramePath)
		if err := upscale(job.lrFramePath, job.hrFramePath, job.modelPathToUse); err != nil {
			log.Printf("Worker %d: Warning: failed to upscale frame %s: %v.\n", id, job.lrFramePath, err)
		} else {
			log.Printf("Worker %d: Finished upscale for %s\n", id, job.lrFramePath)
		}
	}
	log.Printf("Worker %d: Exiting\n", id)
}

func processFramesInParallel(lrFrameFiles []string, hrFramesDir string, modelPathToUse string) {
	numJobs := len(lrFrameFiles)
	if numJobs == 0 {
		log.Println("No frames to process.")
		return
	}

	numWorkers := runtime.NumCPU()
	//numWorkers := 1
	if numJobs < numWorkers { // Don't spawn more workers than jobs
		numWorkers = numJobs
	}

	log.Printf("Starting parallel upscale with %d workers for %d frames.\n", numWorkers, numJobs)

	jobs := make(chan upscaleJob, numJobs)
	var wg sync.WaitGroup

	// Start workers
	for w := 1; w <= numWorkers; w++ {
		wg.Add(1)
		go upscaleWorker(w, &wg, jobs, nil)
	}

	// Send jobs to the workers
	for _, lrFramePath := range lrFrameFiles {
		baseName := filepath.Base(lrFramePath)
		hrFramePath := filepath.Join(hrFramesDir, baseName)
		jobs <- upscaleJob{
			lrFramePath:    lrFramePath,
			hrFramePath:    hrFramePath,
			modelPathToUse: modelPathToUse,
		}
	}
	close(jobs)

	wg.Wait() // Wait for all workers to finish

	log.Println("All frames processed.")
}

func upscaleVideo(inputVideoPath, outputVideoPath, modelPathToUse string) error {
	start := time.Now()
	if err := initFFmpegPaths(); err != nil {
		return err
	}

	tempDir, err := os.MkdirTemp("", "video_upscale_")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer func() {
		log.Printf("Cleaning up temp directory: %s", tempDir)
		if err := os.RemoveAll(tempDir); err != nil {
			log.Printf("Error cleaning up temp directory %s: %v", tempDir, err)
		}
	}()

	lrFramesDir := filepath.Join(tempDir, "lr_frames")
	hrFramesDir := filepath.Join(tempDir, "hr_frames")
	if err := os.Mkdir(lrFramesDir, 0755); err != nil {
		return fmt.Errorf("failed to create lr_frames_dir: %w", err)
	}
	if err := os.Mkdir(hrFramesDir, 0755); err != nil {
		return fmt.Errorf("failed to create hr_frames_dir: %w", err)
	}

	log.Printf("Determining FPS for %s...", inputVideoPath)
	//fps, err := getVideoFPS(inputVideoPath)
	/*if err != nil {
		log.Printf("Warning: could not reliably determine FPS for %s: %v. Defaulting to 25fps.", inputVideoPath, err)
		fps = "25" // Default FPS
	}*/
	fps := "24"
	log.Printf("Using FPS: %s for video processing.", fps)

	log.Println("Extracting frames from video...")
	// -vsync vfr helps preserve original frame timings if the source is VFR.
	// -qscale:v 2 is good quality for PNG.
	extractCmdArgs := []string{"-i", inputVideoPath, "-vsync", "vfr", "-qscale:v", "2", filepath.Join(lrFramesDir, "frame_%07d.png")}
	extractCmd := exec.Command(ffmpegPath, extractCmdArgs...)
	log.Printf("Executing ffmpeg: %s %s", ffmpegPath, strings.Join(extractCmdArgs, " "))
	outputBytes, err := extractCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to extract frames: %w\nffmpeg output:\n%s", err, string(outputBytes))
	}
	log.Println("Frames extracted.")

	lrFrameFiles, err := filepath.Glob(filepath.Join(lrFramesDir, "*.png"))
	if err != nil {
		return fmt.Errorf("error listing extracted frames: %w", err)
	}
	if len(lrFrameFiles) == 0 {
		log.Printf("ffmpeg output during extraction:\n%s", string(outputBytes))
		return fmt.Errorf("no frames were extracted from the video. Check video format and ffmpeg output")
	}
	sort.Strings(lrFrameFiles) // Ensure frames are processed in order

	log.Printf("Upscaling %d frames...", len(lrFrameFiles))

	processFramesInParallel(lrFrameFiles, hrFramesDir, modelPathToUse)

	log.Println("All frames processed.")

	tempOutputVideoPath := filepath.Join(tempDir, "temp_video_no_audio.mp4")
	log.Println("Reassembling video from upscaled frames (no audio yet)...")
	// -c:v libx264, -pix_fmt yuv420p for wide compatibility, -crf 18 for high quality.
	// -an to explicitly disable audio for this intermediate step.
	assembleCmdArgs := []string{
		"-framerate", fps, "-i", filepath.Join(hrFramesDir, "frame_%07d.png"),
		"-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-an", //"-crf", "18" high quality, "-crf", "0", RAW
		"-y", // Overwrite if temp_video_no_audio.mp4 exists (shouldn't)
		tempOutputVideoPath,
	}
	assembleCmd := exec.Command(ffmpegPath, assembleCmdArgs...)
	log.Printf("Executing ffmpeg: %s %s", ffmpegPath, strings.Join(assembleCmdArgs, " "))
	outputBytes, err = assembleCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to reassemble video (no audio step): %w\nffmpeg output:\n%s", err, string(outputBytes))
	}

	log.Println("Attempting to merge audio from original video...")
	// -c:v copy (copy video stream as is), -c:a copy (copy audio stream as is)
	// -map 0:v:0 (video from first input), -map 1:a:0? (audio from second, '?' makes it optional)
	// -shortest (finish when shortest input ends)
	audioCopyCmdArgs := []string{
		"-i", tempOutputVideoPath, "-i", inputVideoPath,
		"-c:v", "copy", "-c:a", "copy",
		"-map", "0:v:0", "-map", "1:a:0?",
		"-shortest", "-y", // Overwrite outputVideoPath if exists
		outputVideoPath,
	}
	audioCopyCmd := exec.Command(ffmpegPath, audioCopyCmdArgs...)
	log.Printf("Executing ffmpeg: %s %s", ffmpegPath, strings.Join(audioCopyCmdArgs, " "))
	outputBytes, err = audioCopyCmd.CombinedOutput()
	if err != nil {
		ffmpegOutputStr := string(outputBytes)
		// Check if failure was due to no audio stream in the original
		if strings.Contains(ffmpegOutputStr, "Stream map '1:a:0?' matches no streams") ||
			strings.Contains(ffmpegOutputStr, "Could not find stream") ||
			strings.Contains(ffmpegOutputStr, "Audio stream not found") {
			log.Printf("Original video likely has no audio stream, or audio codec incompatible with copy. Saving video without audio. FFmpeg output:\n%s", ffmpegOutputStr)
			// Rename the video-only file to the final output path
			if errRename := os.Rename(tempOutputVideoPath, outputVideoPath); errRename != nil {
				return fmt.Errorf("failed to rename temp video to final output %s after audio merge attempt: %w (original ffmpeg audio error: %s)", outputVideoPath, errRename, ffmpegOutputStr)
			}
		} else {
			// A more serious error occurred during audio merge
			return fmt.Errorf("failed to merge audio: %w\nffmpeg output:\n%s", err, ffmpegOutputStr)
		}
	}

	log.Printf("Video upscaling complete. Output: %s", outputVideoPath)
	duration := time.Since(start)
	fmt.Printf("Upscale video function took %v\n", duration)
	return nil
}

/* server */
func saveUploadedFile(file multipart.File, handler *multipart.FileHeader) (string, error) {
	tempFile, err := os.CreateTemp("", "upload-*"+filepath.Ext(handler.Filename))
	if err != nil {
		return "", fmt.Errorf("creating temp file: %w", err)
	}
	defer file.Close()

	_, err = io.Copy(tempFile, file)
	if err != nil {
		tempFile.Close()
		os.Remove(tempFile.Name())
		return "", fmt.Errorf("copying uploaded file: %w", err)
	}

	if err := tempFile.Close(); err != nil {
		os.Remove(tempFile.Name())
		return "", fmt.Errorf("closing temp file: %w", err)
	}
	return tempFile.Name(), nil
}

func upscaleImageHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	// Max 30MB for image upload
	if err := r.ParseMultipartForm(30 << 20); err != nil {
		http.Error(w, fmt.Sprintf("Could not parse multipart form: %v", err), http.StatusBadRequest)
		return
	}

	lrFile, lrHeader, err := r.FormFile("lrImage")
	if err != nil {
		http.Error(w, "LR image 'lrImage' not found in request: "+err.Error(), http.StatusBadRequest)
		return
	}

	tempLRPath, err := saveUploadedFile(lrFile, lrHeader)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to save uploaded LR image: %v", err), http.StatusInternalServerError)
		return
	}
	defer os.Remove(tempLRPath)

	tempOutputPath := filepath.Join(os.TempDir(), fmt.Sprintf("upscaled-%d-%s", time.Now().UnixNano(), lrHeader.Filename))
	defer os.Remove(tempOutputPath)

	log.Printf("Mode: Image Upscaling. LR (temp): %s, Output (temp): %s, Model: %s", tempLRPath, tempOutputPath, serverAbsModelPath)
	if err := upscale(tempLRPath, tempOutputPath, serverAbsModelPath); err != nil {
		log.Printf("Image upscaling failed: %v", err)
		http.Error(w, fmt.Sprintf("Image upscaling failed: %v", err), http.StatusInternalServerError)
		return
	}

	log.Println("Image upscaling finished successfully. Serving file.")
	w.Header().Set("Content-Disposition", "attachment; filename="+filepath.Base(tempOutputPath))
	w.Header().Set("Content-Type", "image/png")
	http.ServeFile(w, r, tempOutputPath)
}

func main() {
	port := flag.String("port", "8080", "Port for the server to listen on")

	mode := flag.String("mode", "upscale", "Mode: 'train' or 'upscale'")
	modelPath := flag.String("model", modelFileNameDefault, "Path for model weights.")

	lrImgInputPath := flag.String("lr", "", "Path to LR image for upscaling or single-pair training.")
	outputPath := flag.String("out", "output.png", "Path for the output upscaled image.")

	videoInPath := flag.String("videoin", "", "Path to input video for upscaling.")
	videoOutPath := flag.String("videoout", "output_video.mp4", "Path for the output upscaled video.")

	flag.Parse()

	absModelPath, err := filepath.Abs(*modelPath)
	if err != nil {
		log.Fatalf("Failed to get absolute path for model: %v", err)
	}

	// Initialize or load the global model ONCE
	loadedModel, modelExists, loadErr := loadModelFromFile(absModelPath)
	if loadErr != nil {
		log.Printf("Critical error loading model from %s: %v. Exiting.", absModelPath, loadErr)
		if !os.IsNotExist(loadErr) && !(err != nil && err.Error() == "EOF" && func() bool { fi, _ := os.Stat(absModelPath); return fi.Size() == 0 }()) {
			globalTrainedModel = newFullModel() // Fallback to new if critical error
		}
	}

	if modelExists {
		globalTrainedModel = loadedModel
		log.Printf("Successfully loaded existing model from %s. Entries: %d", absModelPath, len(globalTrainedModel.SharedChannelData))
	} else {
		globalTrainedModel = newFullModel()
		log.Printf("No existing model at %s or model was empty/corrupt. Starting with a new model.", absModelPath)
	}

	switch *mode {
	//case "autotrain":
	//	autotrainConfigurable(absModelPath)
	//case "train":
	case "upscale":
		if *videoInPath != "" { // Video upscaling mode
			if *lrImgInputPath != "" {
				log.Fatal("Cannot specify both -lr (image input) and -videoin (video input) for upscaling.")
			}
			// use output video name
			actualVideoOutPath := *videoOutPath

			// or save as input name + "_u.mp4"
			//actualVideoOutPath := *videoInPath
			//actualVideoOutPath = actualVideoOutPath[:len(actualVideoOutPath)-4] + "_u.mp4"

			log.Printf("Mode: Video Upscaling. Input: %s, Output: %s, Model: %s", *videoInPath, actualVideoOutPath, absModelPath)
			if err := upscaleVideo(*videoInPath, actualVideoOutPath, absModelPath); err != nil {
				log.Fatalf("Video upscaling failed: %v", err)
			}
			log.Println("Video upscaling finished successfully.")

		} else if *lrImgInputPath != "" { // Image upscaling mode
			log.Printf("Mode: Image Upscaling. LR: %s, Output: %s, Model: %s", *lrImgInputPath, *outputPath, absModelPath)
			if err := upscale(*lrImgInputPath, *outputPath, absModelPath); err != nil {
				log.Fatalf("Image upscaling failed: %v", err)
			}
			log.Println("Image upscaling finished successfully.")
		} else {
			log.Fatal("For 'upscale' mode, either -lr (for image) or -videoin (for video) must be specified.")
		}
	default:
		log.Printf("Server starting on port %s...", *port)
	}
	//log.Printf("Server starting on port %s...", *port)

	// HTTP routes
	//http.HandleFunc("/upscale/image", upscaleImageHandler)
	//http.HandleFunc("/upscale/video", upscaleVideoHandler)
	//log.Printf("Model path: %s", serverAbsModelPath)
	//if err := http.ListenAndServe(":"+*port, nil); err != nil {
	//	log.Fatalf("Failed to start server: %v", err)
	//}
}

//go run unblur_model3x.go -mode=train
//go run unblur_model3x.go -mode=upscale -lr=input.png -out=output.png
//go run unblur_model3x.go -mode=upscale -videoin= -videoout=

//go build -o unblur_server_binary .
//go run unblur_model3x.go -port 8080
