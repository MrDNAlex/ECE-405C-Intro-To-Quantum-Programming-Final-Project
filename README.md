# ECE-405C-Intro-To-Quantum-Programming-Final-Project
Repository for the development of the final project for ECE 405C

# Setup Development Environment
To setup your environment run the following commands in your terminal :

Create a new Virtual Environment :
```bash
python -m venv venv
```

Activate the Virtual Environment (Windows) :
```bash
source venv/Scripts/Activate
```

Install the required packages :
```bash
pip install -r requirements.txt
```

One step Setup
```bash
python -m venv venv
source venv/Scripts/Activate
pip install -r requirements.txt
```

# Using the Program
```python
# Load the Image
image = QJPEG("path/to/image.png")

# Save the Image
fileName = "output.jpg"
quality = 80
memoryEfficient = False
image.saveJPEG(fileName, quality, memoryEfficient)
```

# Authors
Alexandre Dufresne-Nappert (MrDNAlex) (20948586) : mr.dnalex.2003@gmail.com / a3dufres@uwaterloo.ca
Ty Davis (20939918) : t8davis@uwaterloo.ca
Ryan Becze (20958526) : rbecze@uwaterloo.ca
