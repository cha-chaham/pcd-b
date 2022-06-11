import cv2
import matplotlib.pyplot as plt

# Membaca Citra
nemo = cv2.imread("nemo.jpg")
# Konversi Citra BGR menjadi Citra RGB
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
# Konversi Citra RGB menjadi Citra HSV
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)

#deklarasi batas bawah (orange cerah)
light_orange=(1,50,1)
#deklarasi batas atas (dark orange)
dark_orange =(18, 255, 255)

# Lakukan Thresholding
mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
# Mask dengan Citra Asli
result = cv2.bitwise_and(nemo, nemo, mask=mask)

# Plot
plt.subplot(2,2,1), plt.title("Citra Asli RGB")
plt.imshow(nemo)
plt.subplot(2,2,2), plt.title("Citra HSV")
plt.imshow(hsv_nemo)
plt.subplot(2,2,3), plt.title("Hasil Mask")
plt.imshow(mask)
plt.subplot(2,2,4)
plt.imshow(result), plt.title("Hasil Segmentasi")
plt.show()
