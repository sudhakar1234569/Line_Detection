import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny
def make_border(image,line_prameters):
    slope, intercept = line_prameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/ slope)
    return np.array([x1,y1,x2,y2])


def avarage_lines(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_ave = np.average(left_fit,axis=0)
    right_fit_ave = np.average(right_fit,axis=0)
    left_line = make_border(image,left_fit_ave)
    right_line = make_border(image,right_fit_ave)

    return np.array([left_line,right_line])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), 10)
    return line_image

def regition_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return  masked_image


cap = cv2.VideoCapture("road.mp4")
while (cap.isOpened()):
     _, frame =  cap.read()
     Canny = canny(frame)
     crop = regition_interest(Canny)
     lines = cv2.HoughLinesP(crop, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
     avarage_line = avarage_lines(frame, lines)
     hit = display_lines(frame, avarage_line)
     path = cv2.addWeighted(frame, 0.8, hit, 1, 1)
     cv2.imshow("demo", path)
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break
cap.release()
cv2.destroyAllWindows()