from QJPEG import QJPEG
import time
import os
import cv2
import csv
import gc

# 1. Ensure the output directory exists so the script doesn't crash
os.makedirs("CompressionResults", exist_ok=True)

images = [
    "Test-Images/Blue Marble.tif",
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

# 2. Setup the CSV file to record data for Excel
csv_filename = "compression_benchmark_results.csv"

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(
        [
            "Photo Name",
            "Quality Parameter",
            "Original Size (KB)",
            "QJPEG Size (KB)",
            "CV2 Size (KB)",
            "Time QFT (s)",
            "Time DCT (s)",
        ]
    )

    for img_path in images:
        # Check if the image exists to avoid crashing mid-benchmark
        if not os.path.exists(img_path):
            print(f"Warning: File not found: {img_path}. Skipping.")
            continue

        # Extract just the filename (e.g., 'DNA') for cleaner logging
        base_name = os.path.basename(img_path).split(".")[0]

        # Get original file size in Kilobytes (KB)
        orig_size_kb = os.path.getsize(img_path) / 1024.0

        # --- QJPEG (QFT) Compression ---
        image_q = QJPEG(img_path)

        for q in range(0, 100, 5):

            # Changed to .jpg extension to avoid format conflicts
            qjpeg_path = f"CompressionResults/{base_name}_q{q}_QJPEG.jpg"

            start_q = time.time()
            image_q.saveJPEG(qjpeg_path, q, "Blue Marble" in base_name)
            time_qft = time.time() - start_q

            # Get QJPEG compressed size
            qjpeg_size_kb = os.path.getsize(qjpeg_path) / 1024.0

            # --- OpenCV (DCT) Compression ---
            cv2_path = f"CompressionResults/{base_name}_q{q}_CV2.jpg"

            start_cv = time.time()
            # FIX: Properly applying the quality parameter 'q' to OpenCV
            cv2.imwrite(cv2_path, image_q.rawImage, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            time_dct = time.time() - start_cv

            # Get OpenCV compressed size
            cv2_size_kb = os.path.getsize(cv2_path) / 1024.0

            # --- Log the Data ---
            writer.writerow(
                [
                    base_name,
                    q,
                    f"{orig_size_kb:.2f}",
                    f"{qjpeg_size_kb:.2f}",
                    f"{cv2_size_kb:.2f}",
                    f"{time_qft:.4f}",
                    f"{time_dct:.4f}",
                ]
            )

            # Print to console so you know it isn't frozen
            print(f"Processed: {base_name} | Quality: {q}")

print(f"\nBenchmarking complete! Open '{csv_filename}' in Excel to view your data.")
