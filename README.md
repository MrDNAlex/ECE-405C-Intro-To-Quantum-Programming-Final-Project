# ECE-405C-Intro-To-Quantum-Programming-Final-Project
Repository for the development of the final project for ECE 405C

# Environment Information
It is suggested to run / test the code with the following environment
- Python : 3.11.0 or higher
- Use the Pip Package Versions in "requirements.txt"
- Having 16 GB of RAM or more (64 GB Recommended for Blue Marble)

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

# Running our code
Once you setup your virtual environment the following files should run without further configuration :
- `PlotResults.py`
- `testDCTvsQDCT.py`
- `testing.py`
- `testQJPEGTime.py`

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
