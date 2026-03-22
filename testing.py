import numpy as np
import matplotlib.pyplot as plt
import cv2


def zigZagTravel(matrix: np.ndarray):

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

    return result

def getRasterTravel(width: int, height: int, stepSize: int = 1):
    
    result = []
    
    for i in range(height//stepSize):
        for j in range(width//stepSize):
            result.append([i * stepSize, j * stepSize])
    
    return result
    
def isColoured(image:np.ndarray):
    return len(np.shape(image)) == 3
    
def getPixelValues(image:np.ndarray, xCoord: int, yCoord:int, matrixSize: int = 8):
    
    # Grab the Pixel Slice that will have QFT Applied to it
    pixels = image[yCoord:yCoord+matrixSize][xCoord:xCoord+matrixSize]
    
    return pixels

grid = np.zeros((8, 8))
path = zigZagTravel(grid)

plt.figure(figsize=(16, 10))

rows = [p[0] for p in path]
cols = [p[1] for p in path]
indices = np.arange(len(path))

scatter = plt.scatter(cols, rows, c=indices, cmap='magma', s=100, edgecolors='black')

cbar = plt.colorbar(scatter)
cbar.set_label('Step Index in Zig-Zag', rotation=270, labelpad=15)

plt.show()

# Display an image in the Graph
DNAImage = cv2.imread("DNA.jpg")
DNAImageGray = cv2.imread("DNA.jpg", cv2.IMREAD_GRAYSCALE)

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

plt.show()

print(getPixelValues(imgRGB, 0, 0))
print(getPixelValues(DNAImageGray, 0, 0))

print(np.shape(getPixelValues(imgRGB, 0, 0)))
print(np.shape(getPixelValues(DNAImageGray, 0, 0)))

print(isColoured(imgRGB))
print(isColoured(DNAImageGray))