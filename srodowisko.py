import cv2 as cv
import numpy as np
import torch as t
import torchvision


# lol = t.randn(5, 3)
# print(lol)
# odpo = t.cuda.is_available()
# print(odpo)

mati_sciezka = "C:/Users/mateu/Downloads/4K Road traffic video for object detection and tracking - free download now!.mp4"

cap = cv.VideoCapture(mati_sciezka)
backSub = cv.createBackgroundSubtractorMOG2()

# funkcja stabliziująca
def stablize_frame(prev_frame, curr_frame, transforms):

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

    # wykrywanie kluczowych punktów do trakowania
    # zwraca wektor 2d punktów czyli wymiar X na 2
    prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    # zwraca optyczny przepływ między klatkami
    curr_pts, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # filtrowanie punktów
    good_prev_pts = prev_pts[status == 1]
    good_curr_pts = curr_pts[status == 1]
    print(good_prev_pts)

    # szacowanie tego jaka nastąpiła transformacja
    matrix, _ = cv.estimateAffinePartial2D(good_prev_pts, good_curr_pts)

    # Przechowywanie transformacji
    dx, dy = matrix[0, 2], matrix[1, 2]

    # Uśrednianie z poprzednimi transformacjami (filtracja mikroruchów)
    transforms.append((dx, dy))
    if len(transforms) > 30:  # Przechowuj do 10 poprzednich klatek
        transforms.pop(0)

    avg_dx = np.mean([t[0] for t in transforms])
    avg_dy = np.mean([t[1] for t in transforms])

    # Koryguj bieżącą klatkę o uśrednioną transformację
    stabilization_matrix = np.array([[1, 0, avg_dx], [0, 1, avg_dy]])

    # przesuwanie obecnej klatki o obliczoną transformację
    stabilized_frame = cv.warpAffine(curr_frame, stabilization_matrix, (curr_frame.shape[1], curr_frame.shape[0]))

    return stabilized_frame

ret, prev_frame = cap.read()

transforms = []

stabilizer = cv.videostab.

while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    stablized_frame = stablize_frame(prev_frame, curr_frame, transforms)

    prev_frame = stablized_frame

    # konwersja do skali szarości
    video = cv.cvtColor(stablized_frame, cv.COLOR_BGR2GRAY)

    # wycinanie tła
    fg_mask = backSub.apply(video)

    # rozmazywanie
    video = cv.blur(video, ksize=(5, 5))

    # filtr bilaterlany
    # video = cv.bilateralFilter(video, d=9, sigmaColor=75, sigmaSpace=75)

    # gray_foreground = cv.bitwise_and(video, video, mask=fg_mask)



    cv.imshow('frame', video)



    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

