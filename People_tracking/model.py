import cv2 as cv
import numpy
import os
import numpy as np
import time


probablity_minimum = 0.5
threshold = 0.3
writer = None

def loading_model(loading_cfg,loading_weight):

    print("[INFO] loading model...")
    net = cv.dnn.readNetFromDarknet(loading_cfg, loading_weight)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) 

    return net 



def last_layer(net):
    output_layer_names = ["yolo_82", "yolo_94", "yolo_106"]
    return output_layer_names


def output_blob(frame):
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (416, 416),swapRB=True, crop=False)

    return blob


def overlap(bounding_boxes,confidences):
    results = cv.dnn.NMSBoxes(bounding_boxes, confidences, probablity_minimum,threshold)

    return results
    


def bounding_box(frame,output_from_network,colours,output_file,w,h,classes):

    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probablity_minimum:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    results = overlap(bounding_boxes,confidences)

    drawing_bounding_box(frame,results,bounding_boxes,confidences,colours,class_numbers,output_file,classes)


def drawing_bounding_box(frame,results,bounding_boxes,confidences,colours,class_numbers,output_file,classes):

    global writer
    if len(results) > 0:

        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]

            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            colour_box_current = colours[class_numbers[i]].tolist() 

            cv.rectangle(frame, (x_min, y_min),(x_min + box_width, y_min + box_height),colour_box_current, 2)

            text_box_current = '{}: {:.4f}'.format(classes[int(class_numbers[i])],confidences[i])
                
            cv.putText(frame, text_box_current, (x_min, y_min - 5),cv.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    if writer is None:
        writer = cv.VideoWriter(output_file,cv.VideoWriter_fourcc('M','J','P','G'),30,(frame.shape[1], frame.shape[0]), True)
    
    writer.write(frame)
    

        

def foreward_pass(net,blob,output):

    net.setInput(blob)
    start = time.time()
    output_from_network = net.forward(output)
    end = time.time()

    return output_from_network,start,end
    

    

def process(input_file,output_file,output,colours,net,classes):

    w=None
    h=None
    f=0
    t=0
    global writer


    video = cv.VideoCapture(input_file)

    while True:
        ret,frame = video.read()

        if not ret:
            break

        if h is None or w is None:
            h, w = frame.shape[:2]
    
        blob = output_blob(frame)

        output_network,start,end = foreward_pass(net,blob,output)

        f+=1
        t = start-end
        
        ## Getting bounding boxes and removing overlapping

        print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))
        bounding_box(frame,output_network,colours,output_file,w,h,classes)

    print()
    print('Total number of frames', f)
    print('Total amount of time {:.5f} seconds'.format(t))
  
    video.release()
    writer.release()

        

    
def main():


    ##Taking input

    class_file = "coco-labels.txt"
    classes = None

    with open(class_file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    loading_weight = "yolov3.weights"
    loading_cfg = "yolov3.cfg"
    input_file = "bottle-detection.mp4"
    output_file = "result.avi"

    ###Importing yolo and setting colours for each label

    net = loading_model(loading_cfg,loading_weight)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    colours = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')


    ###Getting the last layer of yolo

    output = last_layer(net)


    ### Processing and getting the result

    process(input_file,output_file,output,colours,net,classes)



if __name__ == '__main__':
    main()
    
   
    

    
    