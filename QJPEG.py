import numpy as np
import struct
import cv2
import time
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector

#
# Alexandre Dufresne-Nappert
# 20948586
# Implementation of the Quantum JPEG Processor Class, implemented to be easy to use
# Load the Image through Initialization (QJPEG("path/to/image"))
# Save the file .saveJPEG("output.jpg", quality, True/False)
#

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

    timestampStart: float

    def __init__(self, imagePath: str):
        
        self.DC_MAP : dict = self.buildJPEGStandardHuffmanMap(self.STD_DC_COUNTS, self.STD_DC_SYMS)
        self.AC_MAP : dict = self.buildJPEGStandardHuffmanMap(self.STD_AC_COUNTS, self.STD_AC_SYMS)
        
        # Load Image From Path
        img = cv2.imread(imagePath)
        imgYCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32) - 128
        
        self.imageHeight = imgYCbCr.shape[0]
        self.imageWidth = imgYCbCr.shape[1]
        
        self.Y = imgYCbCr[:, :, 0]
        
        halfW = int(np.ceil(self.imageWidth / 2))
        halfH = int(np.ceil(self.imageHeight / 2))
        self.Cr = cv2.resize(imgYCbCr[:, :, 1], (halfW, halfH), interpolation=cv2.INTER_AREA)
        self.Cb = cv2.resize(imgYCbCr[:, :, 2], (halfW, halfH), interpolation=cv2.INTER_AREA)
        
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
    
    def getPaddedDimensions(self, imageGray: np.ndarray):
        """Calculates the padded dimensions of the image to a Multiple of 8
        
        Args :
            imageGray (numpy.ndarray) - A Gray Scale or Single Color Channel Matrix representing an Image
            
        Returns :
            (int, int) - The padded dimensions the image should be so that they are multiples of 8
        """
        h, w = imageGray.shape[:2]
        return int(np.ceil(h / 8) * 8), int(np.ceil(w / 8) * 8)
    
    # ==========================================
    # QUANTUM DCT (QDCT) LOGIC
    # ==========================================
    
    def apply1DQCT(self, qc: QuantumCircuit, dataQubitsIndex: list[int], ancillaQubitIndex: int):
        """Applies the Discrete Cosine Transform in a Single Dimension. This is achieved by Mirroring the Data into 2 Basis using a Hadamard Gate, then Symmetrizing the order of the data in the |1> ket. Finally a Phase Shift is applied at the end to bring the Magnitudes to the Positive Cosine Values
        
        Args :
            qc (QuantumCircuit) - Quiskit Quantum Circuit to add the Quantum Discrete Cosine Transform to
            dataQubitsIndex (list[int]) - List of Indices that represent he Data Qubits and will have the DCT Applied to
            ancillaQubitIndex (int) - Index of the Ancilla Qubit to apply the Hadamard and Symmetrize Data
        """
        
        N = 2**len(dataQubitsIndex)
        
        # Symmetrize Data by entangling states with Hadamard, then inverting the second half of states with CX
        qc.h(ancillaQubitIndex)
        for q in dataQubitsIndex:
            qc.cx(ancillaQubitIndex, q)
            
        qftRange = list(dataQubitsIndex) + [ancillaQubitIndex]
        qc.append(QFT(len(qftRange)), qftRange)
        
        # Phase Shift so that we get the appropriately calculated Positive Magnitudes from Cosine Transform
        for j, q in enumerate(qftRange):
            qc.p((np.pi * (2**j))/(2*N), q)
    
    def prepareDCTBlocks(self, image: np.ndarray):
        """Prepares the Image by splicing it into 8x8 Blocks and Padding it to 16x16 for the Ancilla Qubit Doubling. This formats data so it can be encoded and processed in a Parallelized way using the Quantum Discrete Cosine Transform
        
        Args :
            image (numpy.ndarray) - Matrix Representing an Image (Grayscale or 3-Channel Color)
            
        Returns :
            (numpy.ndarray) - Array of the Flattened 16x16 Image Blocks to Parallelize
            (int) - The True Number of Blocks the Image is comprised of before Padding to Nearest Power of 2
        """
        
        h, w = image.shape[:2]
        newH, newW = self.getPaddedDimensions(image)
        
        # Handle Color vs Grayscale Padding and Block Extraction
        if len(image.shape) == 2:
            paddedImage = np.pad(image, ((0, newH - h), (0, newW - w)), mode='constant')
            blocks8x8 = paddedImage.reshape(newH // 8, 8, newW // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
        else:
            # Pad the height and width, but do not pad the color channels
            paddedImage = np.pad(image, ((0, newH - h), (0, newW - w), (0, 0)), mode='constant')
            
            # Extract 8x8 blocks for each channel and stack them sequentially
            blocks8x8 = np.vstack([
                paddedImage[:, :, i].reshape(newH // 8, 8, newW // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
                for i in range(3)
            ])
            
        numBlocks = blocks8x8.shape[0]
        
        # Create Padded blocks so that Ancilla Qubit Data is populated
        blocks16x16 = np.zeros((numBlocks, 16, 16), dtype=np.float32)
        blocks16x16[:, :8, :8] = blocks8x8
    
        # Make sure total number of blocks is a power of 2
        nextPow2 = 2**int(np.ceil(np.log2(numBlocks))) if numBlocks > 0 else 1
        if nextPow2 > numBlocks:
            extra = np.zeros((nextPow2 - numBlocks, 16, 16), dtype=np.float32)
            blocks16x16 = np.vstack([blocks16x16, extra])
        
        return blocks16x16.flatten(), numBlocks
    
    def processParallelQDCT(self, image: np.ndarray):
        """Processes the 8x8 Blocks from the Image using a Parallelized Quantum Discrete Cosine Transform

        Args :
            image (numpy.ndarray) - Matrix Representing an Image (Grayscale or 3-Channel Color)

        Returns:
            (numpy.ndarray) - Array of 8x8 Frequency Blocks from the Discrete Cosine Transform
        """
        
        flattenedBlocks, blocks = self.prepareDCTBlocks(image)
        numQubits = int(np.log2(len(flattenedBlocks)))
        
        norm = np.linalg.norm(flattenedBlocks)
        normalizedData = flattenedBlocks / norm if norm != 0 else flattenedBlocks
        
        # Initialize State and Circuit
        state = Statevector(normalizedData)
        qc = QuantumCircuit(numQubits)
        
        # Add the Discrete Quantum Cosine Transforms to the Circuit
        self.apply1DQCT(qc, [0, 1, 2], 3)
        self.apply1DQCT(qc, [4, 5, 6], 7)
        
        # Compute the QDCT Result in parallel across all channels
        finalState = state.evolve(qc)
        
        # Decode data and reshape it to 8x8 blocks
        reshaped = (np.real(finalState.data) * norm).reshape(-1, 16, 16)
            
        scaleMatrix = np.ones((8, 8), dtype=np.float32)
        
        inverseSqrt2 = 1 / np.sqrt(2)
        
        scaleMatrix[0, :] *= inverseSqrt2
        scaleMatrix[:, 0] *= inverseSqrt2
        
        # Extract TopLeft 8x8 and return it
        return reshaped[:blocks, :8, :8] * (2.0 * scaleMatrix)
    
    def DCTImage(self, image: np.ndarray, memoryEfficient: bool = False):
        """Applies the Quantum Discrete Cosine Transform to the Image and Handles cases where it is coloured or not
        
        Args :
            image (numpy.ndarray) - Numpy Matrix Representing individual Image Pixels
            memoryEfficient (bool) - Toggle to process channels sequentially (True) or simultaneously (False)
        
        Returns :
            (numpy.ndarray) - Array of 8x8 Frequency Blocks from the Discrete Cosine Transform
        """
        
        # Grayscale images are always processed as a single channel
        if len(image.shape) == 2: 
            return self.processParallelQDCT(image)
        
        if memoryEfficient:
            return np.stack([self.processParallelQDCT(image[:, :, i]) for i in range(3)], axis=0)
        else:
            allBlocks = self.processParallelQDCT(image)
            
            # Split the flattened output blocks back into the 3 original color channels
            numBlocksPerChannel = allBlocks.shape[0] // 3
            return allBlocks.reshape(3, numBlocksPerChannel, 8, 8)
        
    def quantizeImage(self, allBlocksChannels:np.ndarray, quantizationMatrix:np.ndarray):
        """Quantizes the Frequency Blocks by Dividing each by the Quantization Matrix and Rounding their Values
        
        Args :
            allBlocksChannel (numpy.ndarray) - Array of 8x8 Frequency Blocks from the Discrete Cosine Transform
            quantizationMatrix (numpy.ndarray) - Quantization Matrix to use
        
        Returns :
            (numpy.ndarray) - Array of Quantized 8x8 Frequency Blocks
        """
        
        quantized = np.round(allBlocksChannels / quantizationMatrix)
        return quantized.astype(np.int32)
    
    # ==========================================
    # STANDARD JPEG ENCODER (ISO COMPLIANT)
    # ==========================================
    
    def buildJPEGStandardHuffmanMap(self, counts: list[int], symbols: list[int]):
        """Builds the Standardized Huffman Table upon starting
        
        Args :
            counts (list[int]) - Indexes of the Huffman Table Counts
            symbols (list[int]) - Symbols each count will be represented as when encoded
            
        Returns :
            (dict) - Disctionary Huffman Table Map used to convert values to their encoded Huffman Table Counterpart
        """
        
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
    
    def getCategoricalBits(self, value:float, isDC:bool=False):
        """Encodes a Value into it's nearest Power of 2 and it's remaining Qubits
        
        Args :
            value (float) - Value to Encode
            isDC (bool) - Toggle to change the Encoding Limit
        
        Returns :
            (int) - Nearest Power of 2
            (int) - Extra bits needed to represent the number
        """
        
        # Define the Bit Quality (2^10, or 2^11 depending on if DC or not)
        limit = 2047 if isDC else 1023
        
        value = max(min(int(value), limit), -limit)
        
        if value == 0:
            return 0, ""
        
        absVal = abs(value)
        category = int(np.floor(np.log2(absVal))) + 1
        
        if value > 0:
            # Return Binary Outright
            bits = bin(absVal)[2:]
        else:
            # Shift the number to be in positive range, then convert to binary
            shiftedValue = (2**category - 1) + value
            bits = bin(shiftedValue)[2:].zfill(category) 
            
        return category, bits
    
    def encodeACRunLength(self, zigZag: np.ndarray, allEntries: list[(str, bin, bin)], lastNonZeroIndex: int):
        """Encodes the AC Values of the Quantized Frequency Block using Run Length Encoding
        
        Args :
            zigZag (numpy.ndarray) - Array of Values of the Quantized Frequency Block in ZigZag Pattern order
            allEntries (list[(str, bin, bin)]) - List of Tupples storing the Run Length Encoding ("DC/AC", Binary of nearest Power Of 2, Binary of extra bits) 
        """
        
        run = 0
        for i in range(1, lastNonZeroIndex + 1):
            value = int(zigZag[i])
            if value == 0:
                run += 1
            else:
                while run > 15:
                    allEntries.append(("AC", 0xF0, ""))
                    run -= 16
                category, extra = self.getCategoricalBits(value)
                
                # Use 1 Byte to Encode the Number of occurrences + lowest power of 2
                # 0010 (Occurences = 2) 0100 (Power of 2 = 2^4 = 16) 00000000... (Remaining bits needed for extra)) 
                # Use the OR (|) operator to combines the Occurences and Power of 2
                allEntries.append(("AC", (run << 4) | category, extra))
                run = 0
    
    def encodeImageToSymbols(self, quantY: np.ndarray, quantCb: np.ndarray, quantCr: np.ndarray):
        """Encodes the separated Color Channels and Quantized Frequency Blocks into a Run Length Encoding Formatted List.
    
        Args :
            quantY (numpy.ndarray) - Array of Quantized Frequency Blocks for the Luminance (Y) channel.
            quantCb (numpy.ndarray) - Array of Quantized Frequency Blocks for the Chroma Blue (Cb) channel.
            quantCr (numpy.ndarray) - Array of Quantized Frequency Blocks for the Chroma Red (Cr) channel.
            
        Returns :
            (list[(str, bin, bin)]) - List of Values Encoded using Run Length Encoding ("DC/AC", Binary of nearest Power Of 2, Binary of extra bits).
        """
        
        allEntries = []
        previousDC = [0, 0, 0]
        
        hMCUS = int(np.ceil(self.imageHeight / 16))
        wMCUS = int(np.ceil(self.imageWidth / 16))
        
        _, padWY = self.getPaddedDimensions(self.Y)
        _, padWC = self.getPaddedDimensions(self.Cb)
        blocksPerRowY = padWY // 8
        blocksPerRowC = padWC // 8
        
        for mY in range(hMCUS):
            for mX in range(wMCUS):
                # Process Luminance (Y) - 4 Blocks per MCU
                for rOff in [0, 1]:
                    for cOff in [0, 1]:
                        yID = (mY * 2 + rOff) * blocksPerRowY + (mX * 2 + cOff)
                        if yID < len(quantY):
                            self.encodeSingleBlock(quantY[yID], 0, previousDC, allEntries)
                
                # Process Chroma Blue (Cb) - 1 Block per MCU
                cID = mY * blocksPerRowC + mX
                if cID < len(quantCb):
                    self.encodeSingleBlock(quantCb[cID], 1, previousDC, allEntries)
                    
                # Process Chroma Red (Cr) - 1 Block per MCU
                if cID < len(quantCr):
                    self.encodeSingleBlock(quantCr[cID], 2, previousDC, allEntries)
                    
        return allEntries

    def encodeSingleBlock(self, block: np.ndarray, channel: int, previousDC: list, allEntries: list):
        """Encodes a single 8x8 Quantized Frequency Block and appends the results directly to the master entries list.
    
        Args :
            block (numpy.ndarray) - A single 8x8 Quantized Frequency Block to be encoded.
            channel (int) - The ID of the color channel being processed (0 for Y, 1 for Cb, 2 for Cr).
            previousDC (list) - A list containing the previous DC values for all three color channels.
            allEntries (list) - The master list where the Run Length Encoded tuples are appended.
        """
        zigZag = self.zigZagEncoding(block)
        
        # DC Difference
        difference = int(zigZag[0] - previousDC[channel])
        category, extra = self.getCategoricalBits(difference, isDC=True)
        allEntries.append(("DC", category, extra))
        previousDC[channel] = zigZag[0]

        # Find Last Non-Zero AC Value
        lastNonZeroIndex = 63
        while lastNonZeroIndex > 0 and zigZag[lastNonZeroIndex] == 0:
            lastNonZeroIndex -= 1   
            
        self.encodeACRunLength(zigZag, allEntries, lastNonZeroIndex)
        if lastNonZeroIndex < 63:
            allEntries.append(("AC", 0x00, ""))
    
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
        
        Y_QUANTIZATION_SUBSAMPLING = b'\x01\x22\x00'
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
    
    def getJPEGStartOfScan(self) -> bytes:
        """Gets the JPEG Start of Scan Header. This tells the JPEG what Huffman Tables and Quantization Matrix each Color Channel will use
        
        # Start of Scan (SOS)
        SOS (\xff\xda) (2 Bytes) \n
        SOS_COMPONENTS (\x03) (1 Byte) \n
        
        ## Luminance Huffman Table (Y_HUFFMAN_TABLE_INDEX)
        Luminance Color Channel ID (\x01 = 1) (1 Byte) \n
        Huffman Table ID (\x00 = 0) (1 Byte) \n
        
        ## Chroma Blue Huffman Table (CB_HUFFMAN_TABLE_INDEX)
        Luminance Color Channel ID (\x02 = 2) (1 Byte) \n
        Huffman Table ID (\x00 = 0) (1 Byte) \n
        
        ## Chroma Red Huffman Table (CR_HUFFMAN_TABLE_INDEX)
        Luminance Color Channel ID (\x03 = 3) (1 Byte) \n
        Huffman Table ID (\x00 = 0) (1 Byte) \n
        
        ## Spectral Selection (SOS_SPECTRAL_SELECTION)
        Start of Selection (\x00) (1 Byte) (DC Coefficient) \n
        End of Selection (\x3f = 63) (1 Byte) (63rd AC Coefficient) \n
        
        ## Successive Approximation (SOS_SUCCESSIVE_APPROXIMATION)
        Used for more advanced JPEGs that start blurry, and get more refined \n
        Set to (\x00 = 0) (1 Byte) \n
        """
        
        SOS = b'\xff\xda'
        SOS_COMPONENTS = b'\x03'
        
        Y_HUFFMAN_TABLE_INDEX = b'\x01\x00'
        CB_HUFFMAN_TABLE_INDEX = b'\x02\x00' 
        CR_HUFFMAN_TABLE_INDEX = b'\x03\x00'
        
        SOS_SPECTRAL_SELECTION = b'\x00\x3f'
        SOS_SUCCESSIVE_APPROXIMATION = b'\x00'
        
        SOSTableIndices = Y_HUFFMAN_TABLE_INDEX + CB_HUFFMAN_TABLE_INDEX + CR_HUFFMAN_TABLE_INDEX
        
        SOSPayload = SOS_COMPONENTS + SOSTableIndices + SOS_SPECTRAL_SELECTION + SOS_SUCCESSIVE_APPROXIMATION
        
        SOSPayloadLength = struct.pack('>H', len(SOSPayload) + 2)
        
        return SOS + SOSPayloadLength + SOSPayload
    
    def getJPEGByteScan(self, scaledQuantizationMatrix: np.ndarray, memoryEfficient: bool = False) -> bytes:
        """Gets the JPEG Byte Scan of the Image. Achieves this by Applying the following steps :
        
        1. Apply Quantum Discrete Cosine Transform to Image
        2. Quantizing the Frequency Blocks
        3. Encoding the Frequency Blocks using Run Length Encoding
        4. Applying Huffman Encoding to the Run Length Encoding
        5. Converting everything to a Byte String and returning it
        
        Args : 
            scaledQuantizationMatrix (numpy.ndarray) - The Scaled Quantization Matrix used to Quantize Frequency Blocks
            memoryEfficient (bool) - Toggle to process channels sequentially (True) or simultaneously (False)
        
        Returns :
            (bytes) - String of Encoded Bytes representing the JPEG Image
        """
        
        # Process the each colour channel seperately since they are difference sizes
        blocksY = self.quantizeImage(self.processParallelQDCT(self.Y), scaledQuantizationMatrix)
        blocksCb = self.quantizeImage(self.processParallelQDCT(self.Cb), scaledQuantizationMatrix)
        blocksCr = self.quantizeImage(self.processParallelQDCT(self.Cr), scaledQuantizationMatrix)
        
        entries = self.encodeImageToSymbols(blocksY, blocksCb, blocksCr)
        
        # Huffman Encoding Using Precomputed Tables
        byteString = "".join([(self.DC_MAP[e[1]] if e[0] == 'DC' else self.AC_MAP[e[1]]) + e[2] for e in entries])
        
        # Stuff the Bytes
        padding = (8 - len(byteString) % 8) % 8
        byteString += "1" * padding
        byteStringArray = bytearray(int(byteString[i:i+8], 2) for i in range(0, len(byteString), 8))
        stuffedBytes = bytes(byteStringArray).replace(b'\xff', b'\xff\x00')
        
        print("Finished Quantizing and Encoding!")
        
        return stuffedBytes
    
    def getJPEGFooter(self) -> bytes:
        """Gets the JPEG Standardized Footer Bytes
        
        JPEG Footer (\xff\xd9) (2 Bytes)
        """
        SOE = b'\xff\xd9'
        
        return SOE
    
    def saveJPEG(self, fileName: str, quality:int = 90, memoryEfficient: bool = False):
        """Saves the Loaded Image as a JPEG File with a defined Quality level between 1 - 100
        
        Args :
            fileName (str) - Name of the Saved JPEG File
            quality (int) - Quality Level to Save the JPEG as (1 - 100) (Higher Quality = Larger JPEG File Size = Higher Fidelity)
            memoryEfficient (bool) - Toggle to process channels sequentially (True) or simultaneously (False)
        """
        
        scaledQuantizationMatrix = self.scaleQuantizationMatrix(self.QLUMINANCE, quality)
        
        print("Saving Image as JPEG...")
        
        self.timestampStart = time.time()
        
        with open(fileName, 'wb') as f:
            f.write(self.getJPEGSignature())                                        # SOI (Start of Image (Header))
            f.write(self.getJPEGVersionInfo())                                      # APP0 (JPEG Version Info)
            f.write(self.getJPEGQuantizationMatrix(scaledQuantizationMatrix))       # DQT (Quantization Matrix Encoding)
            f.write(self.getJPEGColorFormatSpecification())                         # SOF0 (Color Format Specification)
            f.write(self.getJPEGHuffmanTableDC())                                   # DHTDC (DC Huffman Table Encoding)
            f.write(self.getJPEGHuffmanTableAC())                                   # DHTAC (AC Huffman Table Encoding)
            f.write(self.getJPEGStartOfScan())                                      # SOS (Start of Scan of Image Compression)
            f.write(self.getJPEGByteScan(scaledQuantizationMatrix, memoryEfficient))# Raw Compressed Bits of Image
            f.write(self.getJPEGFooter())                                           # EOI (End of Image (Footer))
        
        print("Image Saved!")
