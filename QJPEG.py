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
    # UTILITY FUNCTIONS
    # ==========================================
    
    def scaleQuantizationMatrix(self, qMatrix: np.ndarray, quality: int = 50):
        """Scales the Quantization Matrix so that it appropriately encodes data depending on specified quality
        
        Args :
            qMatrix (numpy.ndarray) - Quantization Matrix being used
            quality (float) - The JPEG Compression Quality (1 - 100)
            
        Returns :
            (numpy.ndarray) - The scaled Quantization Matrix accounting for Quality level
        """
        quality = max(min(quality, 100), 1)
        
        scale = 0
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        
        scaled = np.floor((qMatrix * scale + 50)/100)
        
        # Clamp values to standard 8-bit JPEG limits
        scaled[scaled < 1] = 1
        scaled[scaled > 255] = 255
        
        return scaled.astype(np.float32)
    
    
    def zigZagEncoding(self, matrix: np.ndarray):
        """The ZigZag Encoder, for lack of a better term, it returns the inputted Matrix as an array of it's values following the Zig Zag pattern starting from the Top Left Corner
        
        Args :
            matrix (numpy.ndarray) - The Matrix being encoded in a Zig Zag Pattern
            
        Returns :
            (numpy.ndarray) - Array of Values from the Matrix ordered in the Zig Zag Pattern
        """
        result = []
        r, c = 0, 0
        
        h, w = np.shape(matrix)
        rows, cols = h, w
        
        for _ in range(rows * cols):
            result.append(matrix[r, c])
            
            if (r + c) % 2 == 0:
                # Move Diagonal Upright
                if c == cols - 1:
                    # Hit Right Wall, Move Down
                    r += 1
                elif r == 0:
                    # Hit Top Wall, Move Right
                    c += 1
                else:
                    r -= 1
                    c += 1
                    
            else:
                # Move Diagonal Downleft
                if r == rows - 1:
                    # Hit Bottom Wall, Move Right
                    c += 1
                elif c ==0 :
                    # Hit Left Wall, Move Down
                    r += 1
                else:
                    r += 1
                    c -= 1 
        
        return np.array(result)
    
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
    
    def getJPEGSignature(self) -> bytes:
        """Gets the Standardized JPEG File Header Bytes
        
        # Start of Image (SOI)
        JPEG Signature Bytes : FFD8
        """
        SOI = b'\xff\xd8'
        
        return SOI
    
    def getJPEGVersionInfo(self) -> bytes:
        """Gets the Standardized JPEG Version Info Section
        
        # APP0
        Section Header (\xff\xe0) (2 Bytes) \n
        Length of the Section (16) stored as Big-Endian Unsigned Short (2 Bytes) \n
        Identifier (JFIF\x00) (5 Bytes) \n
        Version (\x01\x01) (2 Bytes) \n
        Density of Units (\x00) (1 Byte) \n
        X Density (\x00\x01) (2 Bytes) \n
        Y Density (\x00\x01) (2 Bytes) \n
        Thumbnail Width (\x00) (1 Byte) \n
        Thumbnail Height (\x00) (1 Byte) \n
        """
        
        sectionHeader = b'\xff\xe0'
        sectionLength = struct.pack('>H', 16)
        identifier = b'JFIF\x00'
        version = b'\x01\x01'
        densityOfUnits = b'\x00'
        xDensity = b'\x00\x01'
        yDensity = b'\x00\x01'
        thumbnailHeight = b'\x00'
        thumbnailWidth = b'\x00'
        
        return sectionHeader + sectionLength + identifier + version + densityOfUnits + xDensity + yDensity + thumbnailHeight + thumbnailWidth
    
    def getJPEGQuantizationMatrix(self, quantizationMatrix: np.ndarray) -> bytes:
        """Gets the Quantization Matrix to store in the JPEG encoded in a ZigZag Pattern
        
        Args :
            quantizationMatrix (numpy.ndarray) - The Quantization Matrix to encode into the Image
    
        # Quantization Matrix Storage 
        Section Header (\xff\xdb) \n
        Length of Section (67) stored as Big-Endian Unsigned Short (2 Bytes) \n
        Quantization Table ID (\x00 = 0) (1 Byte) \n
        Quantization Matrix Values (64 Bytes) (Must be Added) \n
        """
        DQT = b'\xff\xdb'
        sectionLength = struct.pack('>H', 67)
        quantizationMatrixID = b'\x00'
        zigZaggedQuantizationMatrix = self.zigZagEncoding(quantizationMatrix).astype(np.uint8)
        
        return DQT + sectionLength + quantizationMatrixID + zigZaggedQuantizationMatrix.tobytes()
    
    def getJPEGColorFormatSpecification(self) -> bytes:
        """Gets the JPEG Standardized Color Format Specification. Encodes the info based off the image properties
        
        # SOF0 - Image Dimensions
        Section Header (\xff\xc0) (2 Bytes) \n
        Color Precision (Unsigned Char) (1 Byte) (must be added) \n
        Image Height (Unsigned Short) (2 Bytes) (must be added) \n
        Image Width (Unsigned Short) (2 Bytes) (must be added) \n 
        Color Channels (Unsigned Char) (1 Byte) (must be added) \n
        
        ## Luminance Subsampling (Y_QUANTIZATION_SUBSAMPLING)
        Color Channel ID (\x01 = 1) (1 Byte) \n
        Sampling Type (\x11 = 1x1) \n
        Quantization Table ID (\x00 = 0) \n
        
        ## Chroma Blue Subsampling (CB_QUANTIZATION_SUBSAMPLING)
        Color Channel ID (\x02 = 2) (1 Byte) \n
        Sampling Type (\x11 = 1x1) \n
        Quantization Table ID (\x00 = 0) \n
        
        ## Chroma Red Subsampling (CR_LUMINANCE_QUANTIZATION)
        Color Channel ID (\x03 = 3) (1 Byte) \n
        Sampling Type (\x11 = 1x1) \n
        Quantization Table ID (\x00 = 0) \n
        """
        
        SOF0 = b'\xff\xc0'
        
        Y_QUANTIZATION_SUBSAMPLING = b'\x01\x11\x00'
        CB_QUANTIZATION_SUBSAMPLING = b'\x02\x11\x00'
        CR_LUMINANCE_QUANTIZATION = b'\x03\x11\x00'
        
        colorAccuracy = 8
        colorChannels = 3
        
        SOFDimensionPayload = struct.pack('>BHHB', colorAccuracy, self.imageHeight, self.imageWidth, colorChannels)
        
        SOFPayload = SOFDimensionPayload + Y_QUANTIZATION_SUBSAMPLING + CB_QUANTIZATION_SUBSAMPLING + CR_LUMINANCE_QUANTIZATION

        SOFPayloadLength = struct.pack('>H', len(SOFPayload) + 2)
        
        return SOF0 + SOFPayloadLength + SOFPayload
    
    def getJPEGHuffmanTableDC(self) -> bytes:
        """Gets the JPEG DC Huffman Table and encodes it
        
        # DC Huffman Table
        Header (\xff\xc4) (2 Bytes) \n
        Table Length (1 Byte) \n
        Table ID (\x00) (1 Byte) \n
        DC Counts (len(Counts) Bytes)) \n
        DC Symbols (len(Symbols) Bytes)
        """
        
        HUFFMAN_TABLE_DC = b'\xff\xc4'
        
        HuffmanTableDCLength = struct.pack('>H', 2 + 1 + len(self.STD_DC_COUNTS) + len(self.STD_DC_SYMS))
        HuffmanTableID = b'\x00'
        
        return HUFFMAN_TABLE_DC + HuffmanTableDCLength + HuffmanTableID + bytes(self.STD_DC_COUNTS) + bytes(self.STD_DC_SYMS)
    
    def getJPEGHuffmanTableAC(self):
        """Gets the JPEG AC Huffman Table and encodes it
        
        # AC Huffman Table
        Header (\xff\xc4) (2 Bytes) \n
        Table Length (1 Byte) \n
        Table ID (\x10) (1 Byte) \n
        AC Counts (len(Counts) Bytes)) \n
        AC Symbols (len(Symbols) Bytes)
        """
        HUFFMAN_TABLE_AC = b'\xff\xc4'
        
        HuffmanTableACLength = struct.pack('>H', 2 + 1 + len(self.STD_AC_COUNTS) + len(self.STD_AC_SYMS))
        HuffmanTableID = b'\x10'
        
        return HUFFMAN_TABLE_AC + HuffmanTableACLength + HuffmanTableID + bytes(self.STD_AC_COUNTS) + bytes(self.STD_AC_SYMS)
    
    def saveJPEG(self, fileName: str, quality:int = 90):
        
        # Check if Image has been loaded?
        
        scaledQuantizationMatrix = self.scaleQuantizationMatrix(self.QLUMINANCE, quality)
        
        print("Saving Image as JPEG...")
        
        with open(fileName, 'wb') as f:
            f.write(self.getJPEGSignature())                                    # SOI (Start of Image (Header))
            f.write(self.getJPEGVersionInfo())                                  # APP0 (JPEG Version Info)
            f.write(self.getJPEGQuantizationMatrix(scaledQuantizationMatrix))   # DQT (Quantization Matrix Encoding)
            f.write(self.getJPEGColorFormatSpecification())                     # SOF0 (Color Format Specification)
            f.write(self.getJPEGHuffmanTableDC())                               # DHTDC (DC Huffman Table Encoding)
            f.write(self.getJPEGHuffmanTableAC())                               # DHTAC (AC Huffman Table Encoding)
            # SOS (Start of Scan of Image Compression)
            # Raw Compressed Bits of Image
            # EOI (End of Image (Footer))
        
        print("Image Saved!")
