import cv2
import matplotlib.pyplot as plt
import numpy as np
import os 
import sys 

# Burada resimlerin bulunduğu dizinler bulunuyor.
IMAGE_FILENAMES = [
    "img/img1.png",
    "img/img2.png",
    "img/img3.png",
    "img/img4.png",
    "img/img5.png",
] 

OUTPUT_DIR = "odev4_outputs"

STRETCH_LO = 5.0  # Lower percentile (e.g., 2.0, 5.0, 10.0)
STRETCH_HI = 95.0 

def load_gray(filepath: str) -> np.ndarray | None:
    """Image'i 8‑bit grayscale olarak okuyan fonksiyon."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image '{filepath}'. Check the path and file.")
        return None
    return img

def hist_manual(img: np.ndarray, bins: int = 256) -> np.ndarray:
    
    # Histogramı hesaplayan fonksiyon

    # Image flat mı?
    img_flat = img.ravel()
    h = np.zeros(bins, dtype=np.int64)
    for pixel_value in img_flat:
        if 0 <= pixel_value < bins:
            h[pixel_value] += 1
    return h

def plot_histograms(name: str, counts: np.ndarray, log_counts: np.ndarray, bins: int = 256) -> None:
    """Plots the linear and log-scaled histograms on the same figure."""
    xs = np.arange(bins)
    plt.figure(figsize=(10, 5)) # Slightly wider figure
    plt.plot(xs, counts, label="Normal Histogram", lw=1.5, color='blue')
    plt.plot(xs, log_counts, label="Log-Histogram (log(1+N))", lw=1.5, ls="--", color='red')
    plt.title(f"Histogram ve Log-Histogram - {name}") # Turkish Title
    plt.xlabel("Piksel Değeri") # Turkish Label
    plt.ylabel("Frekans (Piksel Sayısı)") # Turkish Label
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6) # Add a light grid
    plt.tight_layout()

def cumulative_hist(counts: np.ndarray) -> np.ndarray:
    """Return cumulative sum of histogram counts."""
    return np.cumsum(counts)

def auto_contrast(img: np.ndarray, lo_perc: float = 2.0, hi_perc: float = 98.0) -> np.ndarray:
    """Linear contrast stretch between the given percentiles (lo_perc, hi_perc)."""
    # Ensure percentiles are valid
    lo_perc = max(0.0, min(100.0, lo_perc))
    hi_perc = max(0.0, min(100.0, hi_perc))
    if lo_perc >= hi_perc:
        print(f"Warning: Low percentile ({lo_perc}) >= High percentile ({hi_perc}). Using default 0-100.")
        lo_perc, hi_perc = 0.0, 100.0

    # Calculate pixel values corresponding to the percentiles
    lo_val, hi_val = np.percentile(img, (lo_perc, hi_perc))

    # Handle cases where low and high values are the same (e.g., flat image)
    if hi_val == lo_val:
        # Stretch to full range if flat, or return as is.
        # Let's return as is to avoid dividing by zero.
        # Or maybe map everything to 128? Let's return as is for now.
        print(f"Warning: Contrast range is zero ({lo_val=}, {hi_val=}). Returning original.")
        return img.copy() # Return a copy

    # Stretch the image linearly
    # Formula: output = (input - lo_val) * (255 / (hi_val - lo_val))
    scale = 255.0 / (hi_val - lo_val)
    stretched = (img.astype(np.float32) - lo_val) * scale

    # Clip values to the 0-255 range and convert back to uint8
    stretched = np.clip(stretched, 0, 255)
    return stretched.astype(np.uint8)

# --- End Helper Functions ---


# --- Main Processing Logic ---
def run_processing():
    """Main function to process the images defined in IMAGE_FILENAMES."""
    # Create the output directory if it doesn't exist
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output will be saved to directory: '{OUTPUT_DIR}'")
    except OSError as e:
        print(f"Error creating output directory '{OUTPUT_DIR}': {e}")
        return # Stop if we can't create the output folder

    # Check if enough images are provided
    if not IMAGE_FILENAMES:
        print("Error: No image filenames specified in the IMAGE_FILENAMES list.")
        print("Please edit the script and add your image file paths.")
        return
    elif len(IMAGE_FILENAMES) < 3:
         print(f"Warning: The assignment asks for at least 3 images, but only {len(IMAGE_FILENAMES)} are listed.")

    # Process each image in the list
    for filename in IMAGE_FILENAMES:
        print(f"\n--- Processing image: {filename} ---")

        # Construct the full path (assuming images are relative to the script's location)
        # You could use os.path.join if needed, but direct string is simpler here
        img_path = filename

        # Extract a base name for output files (e.g., "image1" from "image1.png")
        base_name = os.path.splitext(os.path.basename(filename))[0]

        # --- Load the Image ---
        img = load_gray(img_path)
        # If loading failed, skip to the next image
        if img is None:
            continue

        # --- a) Calculate Histogram and Compare with Built-in ---
        print("a) Calculating histogram manually and comparing with numpy...")
        counts_manual = hist_manual(img, bins=256)
        # Use numpy's histogram function for comparison
        # img.ravel() flattens the 2D image array into a 1D array
        counts_numpy, bin_edges = np.histogram(img.ravel(), bins=256, range=[0, 256])

        # Compare the results (they should be identical)
        if np.array_equal(counts_manual, counts_numpy):
            print("   OK: Manual histogram matches numpy.histogram result.")
        else:
            # This should ideally not happen if hist_manual is correct
            print("   WARNING: Manual histogram DOES NOT match numpy.histogram result!")
            diff = np.sum(np.abs(counts_manual - counts_numpy))
            print(f"   Total difference in counts: {diff}")

        # Use the manually calculated histogram for the next steps
        counts = counts_manual

        # --- b) Plot Log-histogram and Normal Histogram ---
        print("b) Plotting linear and log histograms...")
        # Calculate log histogram: log(1 + count) to avoid log(0) issues
        log_counts = np.log1p(counts)

        # Create and save the plot
        plot_histograms(base_name, counts, log_counts)
        plot_filename_hist = os.path.join(OUTPUT_DIR, f"{base_name}_hist_log.png")
        try:
            plt.savefig(plot_filename_hist, dpi=200) # Save with decent resolution
            print(f"   Saved histograms plot to: {plot_filename_hist}")
        except Exception as e:
            print(f"   Error saving histogram plot: {e}")
        plt.close() # Close the plot window to free memory

        # --- c) Calculate and Plot Cumulative Histogram ---
        print("c) Calculating and plotting cumulative histogram...")
        cum_counts = cumulative_hist(counts)

        # Create and save the cumulative histogram plot
        plt.figure(figsize=(8, 4)) # Another figure
        plt.plot(np.arange(256), cum_counts, color='green')
        # You could also plot the normalized version:
        # plt.plot(np.arange(256), cum_counts / cum_counts[-1], color='green')
        plt.title(f"Kümülatif Histogram - {base_name}") # Turkish Title
        plt.xlabel("Piksel Değeri") # Turkish Label
        plt.ylabel("Kümülatif Piksel Sayısı") # Turkish Label
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xlim(0, 255) # Ensure x-axis covers the full 0-255 range
        plt.ylim(bottom=0) # Ensure y-axis starts at 0
        plt.tight_layout()
        plot_filename_cumhist = os.path.join(OUTPUT_DIR, f"{base_name}_cumhist.png")
        try:
            plt.savefig(plot_filename_cumhist, dpi=200)
            print(f"   Saved cumulative histogram plot to: {plot_filename_cumhist}")
        except Exception as e:
            print(f"   Error saving cumulative histogram plot: {e}")
        plt.close() # Close the plot window

        # --- d) Apply Automatic Contrast Adjustment ---
        print(f"d) Applying auto contrast stretch (Percentiles: {STRETCH_LO}% - {STRETCH_HI}%)...")
        stretched_img = auto_contrast(img, STRETCH_LO, STRETCH_HI)

        # Save the stretched image
        # Include percentile values in the filename for clarity
        stretch_filename = os.path.join(OUTPUT_DIR, f"{base_name}_stretch_{STRETCH_LO}-{STRETCH_HI}.png")
        success = cv2.imwrite(stretch_filename, stretched_img)
        if success:
            print(f"   Saved stretched image to: {stretch_filename}")
        else:
             print(f"   Error saving stretched image to: {stretch_filename}")

        # --- (Optional) Display images ---
        # Uncomment the lines below if you want to see the images pop up
        # cv2.imshow(f"Original - {base_name}", img)
        # cv2.imshow(f"Stretched {STRETCH_LO}-{STRETCH_HI}% - {base_name}", stretched_img)
        # print("   Press any key in an image window to continue...")
        # cv2.waitKey(0) # Wait indefinitely until a key is pressed
        # cv2.destroyAllWindows() # Close all OpenCV windows

# --- Run the script ---
# This standard Python construct ensures the processing runs only when
# the script is executed directly (not when imported as a module)
if __name__ == "__main__":
    run_processing()
    print("\nProcessing finished.")
    # Optional: Add a pause if running from a double-click on Windows
    # input("Press Enter to exit...")