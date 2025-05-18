import numpy as np
import random 
import logging
from scipy.spatial import distance # ÖZELLİK HESAPLARI doğrulaması için

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("7.Odev - Veri Matrisinin Oluşturulması ve Uzaklık Hesaplamaları")

ornek_sayisi = 10
ozellik_sayisi = 5

ozellik_araliklari = [
    (0, 1),       
    (1, 10), (50, 150),    
    (-5, 5),      
    (1000, 2000)  
]

veri_matrisi = np.zeros((ornek_sayisi, ozellik_sayisi))

for j in range(ozellik_sayisi):  
    min_val, max_val = ozellik_araliklari[j]
    for i in range(ornek_sayisi): 
        veri_matrisi[i, j] = random.uniform(min_val, max_val)

print("OLUŞTURULAN VERİ MATRİSİ")
np.set_printoptions(precision=3, suppress=True)
print(veri_matrisi)
print("-" * 40)

# 1) Her bir özniteliğin ortalama, standart sapma ve varyans değerleri hesaplanması
print("\n 1) ÖZNİTELİK İSTATİSTİKLERİ HESAPLANIYOR")
for j in range(ozellik_sayisi):
    ozellik_verisi = veri_matrisi[:, j] 

    ortalama_hesaplanan = np.sum(ozellik_verisi) / ornek_sayisi
    ortalama_numpy = np.mean(ozellik_verisi)
    varyans_hesaplanan = np.sum((ozellik_verisi - ortalama_hesaplanan)**2) / ornek_sayisi
    std_sapma_hesaplanan = np.sqrt(varyans_hesaplanan)
    std_sapma_numpy = np.std(ozellik_verisi)
    varyans_numpy = np.var(ozellik_verisi) 

    print(f"\nÖzellik {j+1} (Aralık: {ozellik_araliklari[j][0]}-{ozellik_araliklari[j][1]}):")
    print(f"  Hesaplanan Ortalama: {ortalama_hesaplanan:.3f} (NumPy: {ortalama_numpy:.3f})")
    print(f"  Hesaplanan Std. Sapma: {std_sapma_hesaplanan:.3f} (NumPy: {std_sapma_numpy:.3f})")
    print(f"  Hesaplanan Varyans: {varyans_hesaplanan:.3f} (NumPy: {varyans_numpy:.3f})")

print("-" * 40)

print("\n--- 2) KOVARYANS MATRİSİ ---")
kovaryans_matrisi = np.cov(veri_matrisi, rowvar=False)
print(kovaryans_matrisi)
print("-" * 40)

print("\n 3) UZAKLIK HESAPLAMALARI")

ornek_indeksleri = list(range(ornek_sayisi))
random.shuffle(ornek_indeksleri) 

if ornek_sayisi >= 6:
    secilen_ciftler_indeks = [
        (ornek_indeksleri[0], ornek_indeksleri[1]),
        (ornek_indeksleri[2], ornek_indeksleri[3]),
        (ornek_indeksleri[4], ornek_indeksleri[5])
    ]
else: 
    secilen_ciftler_indeks = [(0,1), (1,2), (0,2)]
    if ornek_sayisi < 3: 
        secilen_ciftler_indeks = [(0,1)]


secilen_ciftler_indeks = [
    (0, 1),
    (3, 7),
    (5, 9)
]

for i, cift in enumerate(secilen_ciftler_indeks):
    ornek1_idx, ornek2_idx = cift
    vektor1 = veri_matrisi[ornek1_idx, :]
    vektor2 = veri_matrisi[ornek2_idx, :]

    print(f"\n--- Örnek Çifti {i+1} (Örnek {ornek1_idx+1} ve Örnek {ornek2_idx+1}) ---")

    # a) Euclidean Uzaklığı
    # Elle hesaplama:
    fark_kare_toplam = 0
    for k in range(ozellik_sayisi):
        fark_kare_toplam += (vektor1[k] - vektor2[k])**2
    euclidean_hesaplanan = np.sqrt(fark_kare_toplam)
    euclidean_numpy = np.linalg.norm(vektor1 - vektor2)
    euclidean_scipy = distance.euclidean(vektor1, vektor2)
    print(f"  Euclidean Uzaklığı (Hesaplanan): {euclidean_hesaplanan:.3f}")
    print(f"  Euclidean Uzaklığı (NumPy linalg.norm): {euclidean_numpy:.3f}")
    print(f"  Euclidean Uzaklığı (SciPy): {euclidean_scipy:.3f}")

    # b) Cosine Uzaklığı (1 - Cosine Benzerliği)
    # Elle hesaplama (Cosine Benzerliği):
    dot_product = np.dot(vektor1, vektor2)
    norm_vektor1 = np.linalg.norm(vektor1)
    norm_vektor2 = np.linalg.norm(vektor2)
    if norm_vektor1 == 0 or norm_vektor2 == 0: 
        cosine_benzerligi_hesaplanan = 0 if dot_product == 0 else 1 
    else:
        cosine_benzerligi_hesaplanan = dot_product / (norm_vektor1 * norm_vektor2)
    cosine_uzakligi_hesaplanan = 1 - cosine_benzerligi_hesaplanan
    # SciPy ile (doğrulama):
    cosine_scipy = distance.cosine(vektor1, vektor2) # Bu direkt uzaklığı verir
    print(f"  Cosine Uzaklığı (1 - Benzerlik_Hesaplanan): {cosine_uzakligi_hesaplanan:.3f}")
    print(f"  Cosine Uzaklığı (SciPy): {cosine_scipy:.3f}")

    # c) Manhattan (City Block) Uzaklığı
    mutlak_fark_toplam = 0
    for k in range(ozellik_sayisi):
        mutlak_fark_toplam += abs(vektor1[k] - vektor2[k])
    manhattan_hesaplanan = mutlak_fark_toplam
    manhattan_numpy = np.sum(np.abs(vektor1 - vektor2))
    manhattan_scipy = distance.cityblock(vektor1, vektor2)
    print(f"  Manhattan Uzaklığı (Hesaplanan): {manhattan_hesaplanan:.3f}")
    print(f"  Manhattan Uzaklığı (NumPy sum(abs)): {manhattan_numpy:.3f}")
    print(f"  Manhattan Uzaklığı (SciPy cityblock): {manhattan_scipy:.3f}")

    # d) Mahalanobis Uzaklığı
    try:
        ters_kovaryans_matrisi = np.linalg.inv(kovaryans_matrisi)
        # Elle hesaplama: sqrt((v1-v2).T * inv(COV) * (v1-v2))
        fark_vektoru = vektor1 - vektor2
        mahalanobis_hesaplanan = np.sqrt(np.dot(np.dot(fark_vektoru.T, ters_kovaryans_matrisi), fark_vektoru))
        mahalanobis_scipy = distance.mahalanobis(vektor1, vektor2, ters_kovaryans_matrisi)
        print(f"  Mahalanobis Uzaklığı (Hesaplanan): {mahalanobis_hesaplanan:.3f}")
        print(f"  Mahalanobis Uzaklığı (SciPy): {mahalanobis_scipy:.3f}")

    except np.linalg.LinAlgError:
        print("  Mahalanobis Uzaklığı: Kovaryans matrisi tekil (singular), tersi alınamıyor.")

print("-" * 40)
print("\nÖDEV TAMAMLANDI.")