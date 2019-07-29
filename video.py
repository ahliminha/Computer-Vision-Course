import cv2
import numpy as np
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt 
 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))


def show(image):    
    cv2.imshow('image', image)

def show_hsv(hsv):
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    show(rgb)

def show_mask(mask):
    cv2.imshow('mask', mask)
    
def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    show(img)

def find_biggest_contour(image):
    
    # Copy to prevent modification
    image = image.copy()
    _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = [-1, 1, 2]
    biggest_contour = np.array(biggest_contour).astype(np.int32)    
    if contour_sizes:
      biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
      print(biggest_contour.shape)
      print(biggest_contour)
    else:
      pass
      

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, 1)
    return biggest_contour, mask

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(0)


# Read until video is completed
while(True):
  # Capture frame-by-frame
  ret, frame = cap.read()
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Blur image slightly
  image_blur_hsv = cv2.medianBlur(hsv, 7)

  out.write(frame)
  # defining the red on hsv space
  # 0-10 hue
  min_red = np.array([0, 100, 100])
  max_red = np.array([10, 255, 255])
  image_red1 = cv2.inRange(image_blur_hsv, min_red, max_red)

  # 170-180 hue
  min_red2 = np.array([170, 150, 100])
  max_red2 = np.array([180, 255, 255])
  image_red2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

  

  if ret == True:
 
   # Display the resulting frame
    cv2.imshow('Frame',frame)
    image_red1 = cv2.inRange(image_blur_hsv, min_red, max_red)
    image_red2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)
    image_red = image_red1 + image_red2
    cv2.imshow('image_red', image_red)
    show_mask(image_red)

   # Cleaning up the masks 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

   # Fill small gaps
    image_red_closed = cv2.morphologyEx(image_red, cv2.MORPH_CLOSE, kernel)
    show_mask(image_red_closed)

   # Remove specks
    image_red_closed_then_opened = cv2.morphologyEx(image_red_closed, cv2.MORPH_OPEN, kernel)
    show_mask(image_red_closed_then_opened)

    big_contour, red_mask = find_biggest_contour(image_red_closed_then_opened)
    show_mask(red_mask)

   # Bounding ellipse
    image_with_ellipse = frame.copy()
    ellipse = cv2.fitEllipse(big_contour)
    cv2.ellipse(image_with_ellipse, ellipse, (0,255,0), 2)
    cv2.imshow('final', image_with_ellipse)

  # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
    else: 
      pass
  
  out.write(image_with_ellipse)
 
# When everything done, release the video capture object
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows()