#!/usr/bin/env python3
import os
import cv2                          
import numpy as np                  
import matplotlib.pyplot as plt     

IMAGES = [                          
    "img/img1.png",
    "img/img2.png",
    "img/img3.png",
    "img/img4.png",
    "img/img5.png",
]  

OUTDIR = "odev4_out"            
LO, HI = 5, 95              

def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Resim bulunamadı: {path}")
    return img

def manual_histogram(img: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=int)
    for px in img.flatten():
        hist[px] += 1
    return hist

def plot_hist(name: str, hist: np.ndarray):
    xs = np.arange(256)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, hist, label="Normal")
    plt.plot(xs, np.log1p(hist), label="Log(1+N)", ls="--")
    plt.title(f"Histogram – {name}")
    plt.xlabel("Piksel Değeri")
    plt.ylabel("Frekans")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{name}_hist.png"), dpi=200)
    plt.close()

def plot_cumulative(name: str, hist: np.ndarray):
    cum = hist.cumsum()
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(256), cum)
    plt.title(f"Kümülatif Histogram – {name}")
    plt.xlabel("Piksel Değeri")
    plt.ylabel("Kümülatif Frekans")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{name}_cum.png"), dpi=200)
    plt.close()

def auto_stretch(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    lo_val, hi_val = np.percentile(img, (lo, hi))
    # Bölü sıfır koruması
    if hi_val == lo_val:
        return img.copy()
    stretched = (img - lo_val) * 255.0 / (hi_val - lo_val)
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    for path in IMAGES:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"--> {name} işleniyor…")

        img = read_gray(path)

        # a) Histogram
        hist = manual_histogram(img)

        # b) Normal + Log histogram grafiği
        plot_hist(name, hist)

        # c) Kümülatif histogram
        plot_cumulative(name, hist)

        # d) Otomatik kontrast
        stretched = auto_stretch(img, LO, HI)
        cv2.imwrite(os.path.join(OUTDIR, f"{name}_stretch.png"), stretched)
        print("    ✔ tamamlandı")

    print("\nBitti. Çıktılar:", OUTDIR)

if __name__ == "__main__":
    main()
