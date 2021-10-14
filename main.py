import cv2
import numpy as np
import matplotlib.pyplot as plt

def showImage(window_name,image):
    cv2.imshow(window_name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def showPlotOfImage(image):
    plt.imshow(image)
    plt.show()

def requiredParfOfImage(image, vertices):
    mask = np.zeros_like(image)
    matchMaskColor = 255
    cv2.fillPoly(mask,vertices,matchMaskColor)
    maskedImage = cv2.bitwise_and(image,mask)
    return maskedImage

def drawLines(image, lines):
    image = np.copy(image)
    blankImage = np.zeros((image.shape[0], image.shape[1],3),dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blankImage,(x1,y1),(x2,y2),(255,0,0),thickness = 3)

    image = cv2.addWeighted(image, 0.8, blankImage, 1, 0.0)
    return image

#image = cv2.imread("road.png")

def process(image):
    height = image.shape[0]
    width = image.shape[1]
    colorChannel = image.shape[2]

    vercitesOfRequiredPartOfImage = [(-100,height),(width/2,height*7/10),(width-300,height)]

    grayImage = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    cannyImage = cv2.Canny(grayImage,50,300)
    croppedImage = requiredParfOfImage(cannyImage,np.array([vercitesOfRequiredPartOfImage],np.int32))

    partOfImage = requiredParfOfImage(image,np.array([vercitesOfRequiredPartOfImage],np.int32))
    #showImage("partofimage",partOfImage)

    lines = cv2.HoughLinesP(croppedImage,
                            rho = 3,
                            theta = np.pi/180,
                            threshold=30,
                            lines=np.array([]),
                            minLineLength=50,
                            maxLineGap=20)

    imageWithLines = drawLines(image,lines)

    return imageWithLines

cap = cv2.VideoCapture("Lane.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow("frame", frame)
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()