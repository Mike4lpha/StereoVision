import cv2

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(2)

num = 0

while True:

    succes1, imgL = capL.read()
    succes2, imgR = capR.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', imgL)
        cv2.imwrite('images/stereoright/imageR' + str(num) + '.png', imgR)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1',imgL)
    cv2.imshow('Img 2',imgR)