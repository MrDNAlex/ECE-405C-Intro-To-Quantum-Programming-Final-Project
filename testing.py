import numpy as np
import struct
import matplotlib.pyplot as plt
import cv2
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
from quantizationMatrix import Q_LUMINANCE

from QJPEG import QJPEG

def zigZagTravelIndices(matrix: np.ndarray):

    h, w = np.shape(matrix)
    
    rows = h
    cols = w
    result = []
    
    r, c = 0, 0
    
    for _ in range(rows * cols):
        result.append([r, c])

        # Move upright
        if (r + c) % 2 == 0:
            if c == cols - 1:
                r += 1  # Hit right wall, move down
            elif r == 0:
                c += 1  # Hit top wall, move right
            else:
                r -= 1
                c += 1
            
        # Move Down Left
        else:
            if r == rows - 1:
                c += 1  # Hit bottom wall, move right
            elif c == 0:
                r += 1  # Hit left wall, move down
            else:
                r += 1
                c -= 1

    return np.array(result)

def zigZagTravel(matrix: np.ndarray):

    h, w = np.shape(matrix)
    
    rows = h
    cols = w
    result = []
    
    r, c = 0, 0
    
    for _ in range(rows * cols):
        result.append(matrix[r, c])

        # Move upright
        if (r + c) % 2 == 0:
            if c == cols - 1:
                r += 1  # Hit right wall, move down
            elif r == 0:
                c += 1  # Hit top wall, move right
            else:
                r -= 1
                c += 1
            
        # Move Down Left
        else:
            if r == rows - 1:
                c += 1  # Hit bottom wall, move right
            elif c == 0:
                r += 1  # Hit left wall, move down
            else:
                r += 1
                c -= 1

    return np.array(result)

def getRasterTravel(width: int, height: int, stepSize: int = 1):
    
    result = []
    
    for i in range(height//stepSize):
        for j in range(width//stepSize):
            result.append([i * stepSize, j * stepSize])
    
    return result
    
def isColoured(image:np.ndarray):
    return len(np.shape(image)) == 3
    
def getPixelValues(image:np.ndarray, xCoord: int, yCoord:int, matrixSize: int = 8):
    
    pixels = image[yCoord : yCoord + matrixSize, xCoord : xCoord + matrixSize]
    
    # Pad the block if not 8x8
    h, w = pixels.shape[:2]
    if h < matrixSize or w < matrixSize:
        padded_pixels = np.zeros((matrixSize, matrixSize), dtype=pixels.dtype)
        padded_pixels[:h, :w] = pixels
        return padded_pixels
    
    return pixels


def createQDCTCircuit(numQubits):
    """Applies a Quantum Discrete Fourier Transform
    
    We add an Extra Ancilla Qubit and apply the Hadamard state to it so that we can copy the states and create the superpositon 
    
    $$|\Psi>_data \otimes |0>_{ancilla} \rightarrow$$ 
    
    Apply Hadamard 
    
    $$|\Psi>_data|0> + |\Psi>_data|1>$$
    
    Now we have symmetric mirrors of the data, and we can apply the Quantum Fourier Transform. Since we have symmetric data, it cancels the imaginary sin portion of the 
    """
    total_qubits = numQubits + 1
    qc = QuantumCircuit(total_qubits)

    qc.h(numQubits)
    
    qft_gate = QFT(num_qubits=total_qubits, do_swaps=True).to_gate()
    qc.append(qft_gate, range(total_qubits))
    
    N = 2**numQubits
    for k in range(2 * N):
        # Conceptual representation of phase shift
        angle = -(np.pi * k) / (2 * N)
        
    return qc

def apply1DQDCT(qc, dataQubits, ancillaQubit):
    """
    Applies the 1D QDCT logic to a specific register within a circuit.
    """
    n = len(dataQubits)
    N = 2**n
    
    # 1. Symmetrization (Doubling space)
    qc.h(ancillaQubit)
    
    # 2. Reflection (Creating the mirror image)
    for q in dataQubits:
        qc.cx(ancillaQubit, q)
        
    # 3. QFT (Fourier Basis)
    # Note: We apply it to the data + ancilla (n+1 qubits)
    qftRange = list(dataQubits) + [ancillaQubit]
    qc.append(QFT(len(qftRange)), qftRange)
    
    # 4. Phase Correction (Rz gates) (Rotates into the real plane so that we get the proper values for our cosine (this is due to the shift from the DCT by a half pixel))
    # The angle for qubit j is -pi * 2^j / (2 * N)
    for j, q in enumerate(qftRange):
        angle = -(np.pi * (2**j)) / (2 * N)
        qc.rz(angle, q)

def create2DQCDTCircuit():
    """
    Creates a 2D QDCT for an 8x8 block (64 pixels).
    Uses 3 qubits for Rows, 3 for Cols, and 2 Ancillas.
    
    Based on what I learned in ECE-417-Image-Processing the 2D Fourier Transform is a Linear Operator, therefor the order of applying QFT and DCT won't matter, we get the same results
    """
    rowQubits = [0, 1, 2]
    rowAncilla = 3
    colQubits = [4, 5, 6]
    colAncilla = 7
    
    qc = QuantumCircuit(8)
    
    # Apply QDCT to Rows
    apply1DQDCT(qc, rowQubits, rowAncilla)
    
    # Apply QDCT to Columns
    apply1DQDCT(qc, colQubits, colAncilla)
    
    return qc

grid = np.zeros((8, 8))
path = zigZagTravelIndices(grid)

plt.figure(figsize=(16, 10))

rows = [p[0] for p in path]
cols = [p[1] for p in path]
indices = np.arange(len(path))

scatter = plt.scatter(cols, rows, c=indices, cmap='magma', s=100, edgecolors='black')

cbar = plt.colorbar(scatter)
cbar.set_label('Step Index in Zig-Zag', rotation=270, labelpad=15)

# Display an image in the Graph
DNAImage = cv2.imread("TestImages/DNA.jpg")
DNAImageGray = cv2.imread("TestImages/DNA.jpg", cv2.IMREAD_GRAYSCALE)

imgRGB = cv2.cvtColor(DNAImage, cv2.COLOR_BGR2RGB)

h, w, c = np.shape(DNAImage)

path = getRasterTravel(w, h, 8)

rows = [p[0] for p in path]
cols = [p[1] for p in path]
indices = np.arange(len(path))
        
plt.figure(figsize=(16, 10))
plt.imshow(imgRGB)

scatter = plt.scatter(cols, rows, c=indices, cmap='magma', s=10, edgecolors='black')
cbar = plt.colorbar(scatter)
cbar.set_label('Step Index in Zig-Zag', rotation=270, labelpad=15)

print(getPixelValues(imgRGB, 0, 0))
print(getPixelValues(DNAImageGray, 0, 0))

print(np.shape(getPixelValues(imgRGB, 0, 0)))
print(np.shape(getPixelValues(DNAImageGray, 0, 0)))

print(isColoured(imgRGB))
print(isColoured(DNAImageGray))

qdct_2d = create2DQCDTCircuit()
qdct_2d.draw("mpl") # Visual check
plt.show()

def getPaddedDimensions(imageGray: np.ndarray):
    h, w = imageGray.shape[:2]
    
    newH = int(np.ceil(h / 8) * 8)
    newW = int(np.ceil(w / 8) * 8)
    
    return (newH, newW)

def padImage(imageGray: np.ndarray):
    h, w = imageGray.shape[:2]
    newH, newW = getPaddedDimensions(imageGray)
    return np.pad(imageGray, ((0, newH - h), (0, newW - w)), mode='constant')


def blockifyImage(imageGray: np.ndarray):
    
    newH, newW = getPaddedDimensions(imageGray)
    
    blocks = imageGray.reshape(newH // 8, 8, newW // 8, 8)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, 64)
    
    return blocks

def getNextPowerOf2(value:int):
    base2 = np.log2(value)
    nextBase2 = int(np.ceil(base2))
    return 2**nextBase2

def padExtraBlocks(imageBlocks:np.ndarray):
    
    numBlocks = np.shape(imageBlocks)[0]
    nextPow2 = getNextPowerOf2(numBlocks)
    
    if nextPow2 > numBlocks:
        extraBlocks = np.zeros(( nextPow2 - numBlocks, 64))
        imageBlocks = np.vstack([imageBlocks, extraBlocks])
    
    return imageBlocks

# This is the important part
def customPrepare(imageGray: np.ndarray):
    paddedImage = padImage(imageGray)
    imageBlocks = blockifyImage(paddedImage)
    imageBlocks = padExtraBlocks(imageBlocks)
    
    return imageBlocks.flatten()

def decodeDCT(realData:np.ndarray, numBlocks):
    reshaped = realData.reshape(-1, 16, 16)
    return reshaped[:numBlocks, :8, :8]

def runCustomParallelDCT(imageGray:np.ndarray):
    
    newH, newW = getPaddedDimensions(imageGray)
    blocks = (newH * newW) // 64    
    
    print(blocks)

    flattenedBlocks = customPrepare(imageGray)
    numPixels = len(flattenedBlocks)
    
    # Includes the 6 Qubits for the 64 Pixels + the X Qubits necesary to store the nearest Power of 2 worth of blocks as register Qubits
    numQubits = int(np.log2(numPixels))
    
    norm = np.linalg.norm(flattenedBlocks)
    normalizedData = flattenedBlocks / norm
    
    state = Statevector(normalizedData)
    state = state.tensor(Statevector.from_label('00'))
    
    # Num of Qubits Calculated Earlier + 2 Ancilla to get the Symmtrical state needed for DCT
    qc = QuantumCircuit(numQubits + 2)
    
    # Apply the DCT in the 2 dimensions
    apply1DQDCT(qc, [0, 1, 2], 3)
    apply1DQDCT(qc, [4, 5, 6], 7)
    
    finalState = state.evolve(qc)
    
    return decodeDCT(np.real(finalState.data) * norm, blocks)

def DCTImage(image: np.ndarray):
    """
    Main entry point. Handles Grayscale (2D) or Color (3D) images.
    Returns frequencies in (Blocks, 8, 8) or (3, Blocks, 8, 8)
    """
    
    # Handle Grayscale
    if not isColoured(image):
        return runCustomParallelDCT(image)
    
    # Case: Coloured
    channelResults = []
    
    # Loop through RBG Channels
    for i in range(3):
        print(f"Processing Channel {i}...")
        channelData = image[:, :, i]
        frequencyBlocks = runCustomParallelDCT(channelData)
        channelResults.append(frequencyBlocks)
    
    # Stack the channels back together along the last dimension
    return np.stack(channelResults, axis=0)

def quantizeBlocks(allBlockFrequencies, quantizationMatrix, qualityFactor = 1.0):
    quantized = np.round(allBlockFrequencies / (quantizationMatrix * qualityFactor))
    return quantized.astype(np.int32)

def quantizeImage(allBlocksChannels:np.ndarray, quantizationMatrix, qualityFactor = 1.0):
    
    if len(np.shape(allBlocksChannels)) == 3:
        return quantizeBlocks(allBlocksChannels, quantizationMatrix, qualityFactor)
    
    quantizedChannels = []
    
    for i in range(np.shape(allBlocksChannels)[0]):
        quantizedChannels.append(quantizeBlocks(allBlocksChannels[i], quantizationMatrix, qualityFactor))
    
    return np.stack(quantizedChannels, axis=0)


def get_category_bits(value):
    """
    Returns (Category, Extra_Bits_String) for a quantized value.
    JPEG logic: Positive values use binary, negative use one's complement.
    """
    if value == 0:
        return 0, ""
    
    abs_val = abs(value)
    category = int(np.floor(np.log2(abs_val))) + 1
    
    if value > 0:
        # Binary representation
        bits = bin(value)[2:]
    else:
        # One's complement for negative values
        # e.g., -5 (cat 3) -> 111 (7) - 5 = 010 (2)
        bits = bin((2**category - 1) + value)[2:].zfill(category)
        
    return category, bits

def encodeRunLength(allQuantizedBlocks):
    """
    Produces (Huffman_Symbol, Extra_Bits) pairs for JPEG compatibility.
    """
    all_entries = []
    previousDC = 0
    
    for block in allQuantizedBlocks:
        zigzag = zigZagTravel(block)
        
        # 1. DC Differential Category
        diff = int(zigzag[0] - previousDC)
        category, extra_bits = get_category_bits(diff)
        all_entries.append((category, extra_bits)) # Symbol is just the category
        previousDC = zigzag[0]
        
        # 2. AC Run-Length + Category
        run = 0
        for i in range(1, 64):
            val = int(zigzag[i])
            if val == 0:
                run += 1
            else:
                # Handle long runs (max 15 zeros per JPEG symbol)
                while run > 15:
                    all_entries.append((0xF0, "")) # ZRL (Zero Run Length)
                    run -= 16
                
                category, extra_bits = get_category_bits(val)
                symbol = (run << 4) | category # Combine run and category
                all_entries.append((symbol, extra_bits))
                run = 0
        
        # 3. End of Block (EOB)
        if run > 0:
            all_entries.append((0x00, ""))
            
    return all_entries

def bitstring_to_bytes_jfif(bit_string):
    """
    Standard JPEG packing: Pads with 1s and performs Byte Stuffing.
    """
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "1" * padding # JPEG uses 1-bit padding

    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_array.append(int(bit_string[i:i+8], 2))

    # BYTE STUFFING: Follow 0xFF with 0x00 so it isn't seen as a marker
    stuffed_bytes = byte_array.replace(b'\xff', b'\xff\x00')
    return bytes(stuffed_bytes)

customResult = DCTImage(DNAImageGray)

quantizedResults = quantizeImage(customResult, Q_LUMINANCE)

for i in range(10):
    print(quantizedResults[i])


print("Custom")
print(customResult)
print(customResult.shape)

customResult = DCTImage(DNAImage)

print("Custom")
print(customResult)
print(customResult.shape)

print(customResult[0].shape)

quantizedResults = quantizeImage(customResult, Q_LUMINANCE)

for i in range(10):
    print(quantizedResults[0][i])
    
for i in range(10):
    print(quantizedResults[1][i])
    
for i in range(10):
    print(quantizedResults[2][i])


print(quantizedResults[2][0])
print(zigZagTravel(quantizedResults[2][0]))

image = QJPEG("DNA.jpg")

image.saveJPEG("teams-quantum.jpg", 95)
