import cv2
import numpy as np

def detect_card(image_path, draw=True):
    # Baca gambar asli
    original = cv2.imread(image_path)
    
    #Buat salinan gambar asl;i agar gambar asli masih tersimpan
    image = original.copy()
    
    #Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Filter greenscreen
    green_lower = np.array([10, 40, 50])      #jika ingin masking warna kuning ubah lower hue = 10, upper hue = 30
    green_upper = np.array([30, 255, 255]) 
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lower, green_upper)
    mask = cv2.bitwise_not(mask)

    # Gabungkan hasil filter dengan gambar asli
    greenscreen = cv2.bitwise_and(original, original, mask=mask)
    
    green = greenscreen.copy()
    
    #Deteksi tepi dengan Canny
    edges = cv2.Canny(greenscreen, 50, 150)
    
    #Temukan contour
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_corners = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4:
            card_corners.append(approx)
            if draw:
                for i in range(len(approx)):
                    start_point = tuple(approx[i][0])
                    end_point = tuple(approx[(i + 1) % len(approx)][0])                  
                    #gambar tepi kartu
                    cv2.line(green, start_point, end_point, (255, 0, 255), 2)
                    #lingkari ujung kartu
                    cv2.circle(green, (approx[i][0][0], approx[i][0][1]), 3, (255, 0, 0), 3)

    cv2.imshow('Kartu Asli', original)
    cv2.imshow('masking', greenscreen)
    cv2.imshow('Kartu Terdeteksi', green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return card_corners

image_path = 'kartumiring.jpg'
detect_card(image_path, draw=True)