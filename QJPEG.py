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
    
    # ISO Standard Luma Tables
    STD_DC_COUNTS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    STD_DC_SYMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    STD_AC_COUNTS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
    STD_AC_SYMS = [0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa]
    
    # ==========================================
    # PROPERTIES
    # ==========================================
    
    imageHeight: int = 0
    imageWidth: int = 0
    
    rawImage : np.ndarray

    def __init__(self, imagePath: str):
        
        self.DC_MAP : dict = self.buildJPEGStandardHuffmanMap(self.STD_DC_COUNTS, self.STD_DC_SYMS)
        self.AC_MAP : dict = self.buildJPEGStandardHuffmanMap(self.STD_AC_COUNTS, self.STD_AC_SYMS)
        
        # Load Image From Path
        img = cv2.imread(imagePath)
        imgYCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, [0, 2, 1]]
        self.rawImage = imgYCbCr.astype(np.float32) - 128
        
        self.imageHeight = self.rawImage.shape[0]
        self.imageWidth = self.rawImage.shape[1]
    
    # ==========================================
    # STANDARD JPEG ENCODER (ISO COMPLIANT)
    # ==========================================
    
    def buildJPEGStandardHuffmanMap(self, counts: list[int], symbols: list[int]):
        """Builds the Standardized Huffman Table upon starting"""
        
        huffmanMap = {}
        code, symbolID = 0, 0
        
        for i, count in enumerate(counts):
            bitLength = i + 1
            for _ in range(count):
                # Convert the Integer to Binary, pad it, then     
                binaryWithoutHex = bin(code)[2:]
                zeroPaddedBinary = binaryWithoutHex.zfill(bitLength)
                huffmanMap[symbols[symbolID]] = zeroPaddedBinary
                
                code += 1
                symbolID += 1
            
            # Bitwise Left Shift
            code <<= 1 
        
        return huffmanMap
    
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
