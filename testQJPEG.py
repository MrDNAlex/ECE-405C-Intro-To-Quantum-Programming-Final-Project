from QJPEG import QJPEG
import time

image = "Compression/DNA.jpg"

for q in range(0, 100, 5):
        
    start = time.time()
    image = QJPEG(image)

    image.saveJPEG(f"CompressionResults/Image 1-{q}.png", q)
    
    elapsed = time.time() - start
        