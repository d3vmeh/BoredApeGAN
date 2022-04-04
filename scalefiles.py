import cv2
import os

datadirectory = r"bayc"
for image in os.listdir(datadirectory):
      try:
            imagepath = os.path.join(datadirectory,image)
            imagearray = cv2.imread(imagepath)
            newimagearray=cv2.resize(imagearray,(512,512))
            cv2.imwrite(imagepath,newimagearray)
      except:
            imagepath = os.path.join(datadirectory,image)
            os.remove(imagepath)
            print(imagepath)
