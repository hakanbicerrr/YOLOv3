import numpy as np
import argparse
import time
import cv2
import os



#net = cv2.dnn.readNet("yolov3_coco/yolov3.weights","cfg/yolov3.cfg")
net = cv2.dnn.readNet("custom_cfg_weight/yolov3_custom_7000.weights","custom_cfg_weight/yolov3_custom.cfg")

#classes = ["People","Bicycle"]
#classes = []
classes = ["Bird","Car","Cat","Dog","Person"]
#with open("yolov3_coco/coco.names","r",encoding="Latin-1") as f:
#    classes = [line.strip() for line in f.readlines()]
print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))
#Load image
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Driving.mp4")
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    frame_id +=1
    _, img = cap.read() #Frame
    height,width,channels = img.shape
    #Detecting objects
    blob = cv2.dnn.blobFromImage(img,0.00392,(320,320),(0,0,0),True)

    #for b in blob:
    #    for n,img_blob in enumerate(b):
    #        cv2.imshow(str(n),img_blob)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >0.5:#If you decrease this, more objects will be detected but with less accuracy
                #Object detected
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    print(len(boxes))
    number_objects_detected = len(boxes)
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.6,0.4)
    print(indexes)


    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.rectangle(img, (x-1, y - 15), (x + w+1, y), color, -1)
            cv2.putText(img,label + " " + str(round(confidence,2)) ,(x,y-3),font,1,(255,255,255),2)


    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(img,"FPS: "+str(fps),(10,30),font,3,(0,0,0),1)
    cv2.imshow("room",img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()