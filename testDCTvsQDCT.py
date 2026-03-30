import time
import cv2
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import gc
from QJPEG import QJPEG

#
# Alexandre Dufresne-Nappert
# 20948586
# Benchmarks the DCT vs QDCT computation time on different images
# NOTE
# this has been modified to use Downsampled data now and computes a single color channel
# Previous implementation of the code did not use YCbCr and Downsampling, so the CSV present in the Results folder is semi outdated data
#

def GetAdditionalInfo(fileName):
    image = QJPEG(fileName)
    colorChannels = image.Y

    h, w = colorChannels.shape[:2]
    newH, newW = image.getPaddedDimensions(colorChannels)

    paddedImage = np.pad(
        colorChannels, ((0, newH - h), (0, newW - w)), mode="constant"
    )

    blocks8x8 = np.vstack(
        [
            paddedImage[:, :]
            .reshape(newH // 8, 8, newW // 8, 8)
            .transpose(0, 2, 1, 3)
            .reshape(-1, 8, 8)
            for i in range(3)
        ]
    )

    numBlocks = blocks8x8.shape[0]

    return (h, w, newH, newW, numBlocks)

def RunClassical(fileName):

    # CLASSICAL PREPARATION
    image = QJPEG(fileName)
    h, w = image.Y.shape

    newH, newW = image.getPaddedDimensions(image.Y)

    paddedImage = np.pad(
        image.Y, ((0, newH - h), (0, newW - w)), mode="constant"
    )

    blocks8x8 = np.vstack(
        [
            paddedImage[:, :]
            .reshape(newH // 8, 8, newW // 8, 8)
            .transpose(0, 2, 1, 3)
            .reshape(-1, 8, 8)
            for i in range(3)
        ]
    )

    numBlocks = blocks8x8.shape[0]

    # CLASSICAL BENCHMARK (MATH ONLY)
    blocks8x8 = blocks8x8.astype(np.float32)

    start_classical = time.perf_counter()

    for i in range(numBlocks):
        cv2.dct(blocks8x8[i])

    end_classical = time.perf_counter()
    dctTime = end_classical - start_classical

    print("-" * 30)
    print
    print(f"Classical cv2.dct Loop : {dctTime:.6f} seconds")

    return dctTime

def RunQuantum(fileName):

    image = QJPEG(fileName)

    # QUANTUM PREPARATION
    print("Preparing DCT Blocks...")
    flattenedBlocks, blocks = image.prepareDCTBlocks(image.Y)
    numQubits = int(np.log2(len(flattenedBlocks)))
    print("DCT Blocks Fomatted!")

    print("Normalizing Data...")
    norm = np.linalg.norm(flattenedBlocks)
    normalizedData = flattenedBlocks / norm if norm != 0 else flattenedBlocks
    print("Data Normalized!")
    
    # Clear the Memory
    del flattenedBlocks
    del blocks
    clean = gc.collect()
    print(f"Cleaned Items : {clean}")

    # Set up the state and circuit
    print("Encoding Data into State Vector...")
    state = Statevector(normalizedData)
    
    qc = QuantumCircuit(numQubits)
    image.apply1DQCT(qc, [0, 1, 2], 3)
    image.apply1DQCT(qc, [4, 5, 6], 7)
    print("Data Encoded to State Vector!")

    # Clear the Memory
    del image
    del normalizedData
    clean = gc.collect()
    print(f"Cleaned Items : {clean}")

    # QUANTUM BENCHMARK (MATH ONLY)
    print("Running Quantum Circuit...")
    startQDCT = time.perf_counter()

    state.evolve(qc)

    endQDCT = time.perf_counter()
    print("Quantum Circuit Ran!")

    qdctTime = endQDCT - startQDCT

    print(f"Quantum state.evolve   : {qdctTime:.6f} seconds")
    print("-" * 30)
    
    del state
    del qc
    gc.collect()

    return qdctTime

def RunBenchmark(fileName: str):
    name = fileName.split("/")[-1].split(".")[0]

    print("Starting benchmark...")
    dctTime = RunClassical(fileName)
    qdctTime = RunQuantum(fileName)
    h, w, newH, newW, numBlocks = GetAdditionalInfo(fileName)

    return [name, dctTime, qdctTime, h, w, newH, newW, numBlocks]

if __name__ == "__main__":

    dataFrame = pd.DataFrame(
        columns=[
            "Name",
            "DCT Time",
            "QDCT Time",
            "Height",
            "Width",
            "New Height",
            "New Width",
            "NumBlocks",
        ]
    )

    images = [
        #"Test-Images/Blue Marble.tif", # It is suggested to comment this line out during testing
        "Test-Images/DNA.jpg",
        "Test-Images/Episode3.PNG",
        "Test-Images/ESA.bmp",
        "Test-Images/ETA.bmp",
        "Test-Images/Friday.PNG",
        "Test-Images/FridayBoat.JPG",
        "Test-Images/JOJO.PNG",
        "Test-Images/JOJOW3.PNG",
        "Test-Images/Onion.PNG",
        "Test-Images/Starwars.jpg",
        "Test-Images/starwarsoriginal.PNG",
        "Test-Images/StarwarsRise.JPG",
        "Test-Images/Tanks.png",
        "Test-Images/WildFriday.PNG",
    ]

    for f in images:
        dataFrame.loc[len(dataFrame)] = RunBenchmark(f)

    dataFrame.to_csv("DCTvsQDCT.csv")

    print(dataFrame)
