import numpy as np
import cv2
from keras.models import load_model

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# LOAD THE MODEL
model = load_model('./TSD_model.h5')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img) # 图像直方图均衡化（分布在0-255）
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255 # [0,1]
    return img

def getClassName(classNo):
    if classNo == 0:
        return 'Speed limit (5km/h)'
    elif classNo == 1:
        return 'Speed limit (15km/h)'
    elif classNo == 2:
        return 'Speed limit (30km/h)'
    elif classNo == 3:
        return 'Speed limit (40km/h)'
    elif classNo == 4:
        return 'Speed limit (50km/h)'
    elif classNo == 5:
        return 'Speed limit (60km/h)'
    elif classNo == 6:
        return 'Speed limit (70km/h)'
    elif classNo == 7:
        return 'Speed limit (80km/h)'
    elif classNo == 8:
        return 'No straight or left'
    elif classNo == 9:
        return 'No straight or right'
    elif classNo == 10:
        return 'No straight'
    elif classNo == 11:
        return 'No left'
    elif classNo == 12:
        return 'No left or right'
    elif classNo == 13:
        return 'No right'
    elif classNo == 14:
        return 'No overtake'
    elif classNo == 15:
        return 'No U-turn'
    elif classNo == 16:
        return 'No vehicle'
    elif classNo == 17:
        return 'No honk'
    elif classNo == 18:
        return 'End of speed limit (40km/h)'
    elif classNo == 19:
        return 'End of speed limit (50km/h)'
    elif classNo == 20:
        return 'Go straight or right'
    elif classNo == 21:
        return 'Go straight'
    elif classNo == 22:
        return 'Turn left'
    elif classNo == 23:
        return 'Turn left or right'
    elif classNo == 24:
        return 'Turn right'
    elif classNo == 25:
        return 'Left driving'
    elif classNo == 26:
        return 'Right driving'
    elif classNo == 27:
        return 'Roundabout mandatory'
    elif classNo == 28:
        return 'Vehicle lane'
    elif classNo == 29:
        return 'Honk'
    elif classNo == 30:
        return 'Bicycle lane'
    elif classNo == 31:
        return 'U-turn'
    elif classNo == 32:
        return 'Detour left or right'
    elif classNo == 33:
        return 'Traffic lights'
    elif classNo == 34:
        return 'Take caution'
    elif classNo == 35:
        return 'Pedestrian caution'
    elif classNo == 36:
        return 'Bicycle caution'
    elif classNo == 37:
        return 'School caution'
    elif classNo == 38:
        return 'Sharp right turn'
    elif classNo == 39:
        return 'Sharp left turn'
    elif classNo == 40:
        return 'Down steep slope'
    elif classNo == 41:
        return 'Up steep slope'
    elif classNo == 42:
        return 'Slow down'
    elif classNo == 43:
        return 'T-Crossroads right'
    elif classNo == 44:
        return 'T-Crossroads left'
    elif classNo == 45:
        return 'Village caution'
    elif classNo == 46:
        return 'S-Turn'
    elif classNo == 47:
        return 'Unguarded railway crossing'
    elif classNo == 48:
        return 'Road repair'
    elif classNo == 49:
        return 'Curves Ahead'
    elif classNo == 50:
        return 'Guarded railway crossing'
    elif classNo == 51:
        return 'Accident black spot'
    elif classNo == 52:
        return 'STOP'
    elif classNo == 53:
        return 'No entry'
    elif classNo == 54:
        return 'No parking'
    elif classNo == 55:
        return 'Do not enter'
    elif classNo == 56:
        return 'YIELD'
    elif classNo == 57:
        return 'Stop for Inspection'
    
def TSD_single_predict(imgOrignal):

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    # cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict(img)
    classIndex = np.argmax(classIndex, axis=1) # 输出最大的可能性
    className = getClassName(classIndex)
    probabilityValue = np.amax(predictions)
    # if probabilityValue > threshold:
    #     cv2.putText(imgOrignal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2,cv2.LINE_AA)
    #     cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.imshow("Result", imgOrignal)

    return imgOrignal, className, probabilityValue

    # if cv2.waitKey(1) and 0xFF == ord('q'):
    #     cv2.destroyAllWindows()