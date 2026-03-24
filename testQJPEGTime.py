from QJPEG import QJPEG
import time
import os
import cv2

image = "Compression/DNA.jpg"

for q in range(0, 100, 5):
        
    
    image = QJPEG(image)
    image = cv2.imread("Compression/DNA.png") # Equivalent

    start = time.time()
    image.saveJPEG(f"CompressionResults/Image 1-{q}.png", q)# Equivalent
    elapsed = time.time() - start
    
    start = time.time()
    cv2.imwrite('output.jpg', image) # Equivalent
    elapsed = time.time() - start
    
    