from QJPEG import QJPEG
import time
import os
import cv2
import csv

#
# Ty Davis
# 20939918
# Measures the raw compute time of the either implementationas of image compressions, scrape the original size,
# the new sizes as well as the quality chosen for the compressed images.

# Directory 
os.makedirs("CompressionResults", exist_ok=True)

# Make a list of the test images
images = [#"Test-Images/Blue Marble.tif", leave this commented or suffer computer death
    "Test-Images/DNA.jpg", "Test-Images/Episode3.PNG", "Test-Images/ESA.bmp", "Test-Images/ETA.bmp",
    "Test-Images/Friday.PNG", "Test-Images/FridayBoat.JPG", "Test-Images/JOJO.PNG", "Test-Images/JOJOW3.PNG",
    "Test-Images/Onion.PNG", "Test-Images/Starwars.jpg", "Test-Images/starwarsoriginal.PNG", "Test-Images/StarwarsRise.JPG",
    "Test-Images/Tanks.png","Test-Images/WildFriday.PNG"]

# Setup CSV export file
CSV = "compression_benchmark_results.csv"
with open(CSV, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Photo Name","Quality Parameter","Original Size (KB)","QJPEG Size (KB)","CV2 Size (KB)","Time QFT (s)","Time DCT (s)"])
    for Ipath in images:
        # Check if the image exists to avoid me crashing when i forget images
        if not os.path.exists(Ipath):
            print(f"File not found:{Ipath}.Skip.")
            continue
        # Extract filename without wrap, original size, call QJPEG parameter
        name = os.path.basename(Ipath).split(".")[0]
        OrigSize = os.path.getsize(Ipath) / 1024.0
        Qimage = QJPEG(Ipath)
        # Iterate for each all quality
        for Qual in range(0, 105, 5):
            # ============================= Quantum section ===========================================
            # Each is to be processed as a jpg
            Qpath = f"CompressionResults/{name}_q{Qual}_QJPEG.jpg"
            # Time computation of QJPEG
            Qstart = time.time()
            Qimage.saveJPEG(Qpath, Qual,"Blue Marble"in name)
            QTime = time.time()-Qstart
            # Scrape size (KB)
            QSize = os.path.getsize(Qpath)/1024.0
            # ============================= Classic section ===========================================
            # Each is to be processed as a jpg (again)
            Cpath = f"CompressionResults/{name}_q{Qual}_CV2.jpg"
            # Time computation of JPEG using OpenCV library
            Cstart = time.time()
            cv2.imwrite(Cpath,cv2.imread(Ipath),[int(cv2.IMWRITE_JPEG_QUALITY), Qual]) # Make quality a OpenCV parameter
            CTime = time.time()-Cstart
            # Size (KB)
            CSize = os.path.getsize(Cpath)/1024.0
            # =========================================================================================
            # Write data to csv
            writer.writerow([name,Qual,f"{OrigSize:.2f}",f"{QSize:.2f}",f"{CSize:.2f}",f"{QTime:.4f}",f"{CTime:.4f}",])
            # Print to check if your system crashed (Blue Marble Exclusive (Aleex's Baby))
            print(f"Processed:{name}|Quality:{Qual}")
# Print again but because Alex likes it           
print(f"\nBenchmarking complete! Open '{CSV}' in Excel to view your data.")
