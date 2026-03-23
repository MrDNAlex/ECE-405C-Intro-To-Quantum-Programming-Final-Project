import numpy as np
import struct
import cv2
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector

class QJPEG:

    # ==========================================
    # CONSTANTS
    # ==========================================
    QLUMINANCE = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ])
    
    imageHeight = 0
    imageWidth = 0
    
    rawImage : np.ndarray

    def __init__(self, imagePath: str):
        # Load Image From Path
        img = cv2.imread(imagePath)
        imgYCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, [0, 2, 1]]
        self.rawImage = imgYCbCr.astype(np.float32) - 128
        
        self.imageHeight = self.rawImage.shape[0]
        self.imageWidth = self.rawImage.shape[1]
        
    def saveJPEG(self, fileName: str, quality:int = 90):
        
        # Check if Image has been loaded?
        
        print("Saving Image as JPEG...")
        
        with open(fileName, 'wb') as f:
            f.write("JPEG File")
            # SOI (Start of Image (Header))
            # APP0 (JPEG Version Info)
            # DQT (Quantization Matrix Encoding)
            # SOF0 (Color Format Specification)
            # DHTDC (DC Huffman Table Encoding)
            # DHTAC (AC Huffman Table Encoding)
            # SOS (Start of Scan of Image Compression)
            # Raw Compressed Bits of Image
            # EOI (End of Image (Footer))
        
        print("Image Saved!")
