#!/usr/bin/env python
# -*- coding: utf-8 -*-

#STEP-1
import sys
import logging as log
from scipy.spatial import distance
import numpy as np
import math
import time

import cv2
from openvino.inference_engine import IENetwork, IEPlugin

globalReIdVec = []
def findMatchingPerson(newReIdVec):
    global globalReIdVec
    size = len(globalReIdVec)
    print("size=" + str(size))

    idx = size
    for i in range(size):
        # t1 = time.time() 
        cosSim = cosineSimilarity(newReIdVec, globalReIdVec[i])
        # t2 = time.time()
        # elapsed_time1 = t2-t1
        # print(f"cosSim 経過時間：{elapsed_time1}") 
        # print("cosSim={:.9f}".format(cosSim))

        '''
        t1 = time.time() 
        cossim2 = cos_sim(newReIdVec,  globalReIdVec[i])
        t2 = time.time()
        elapsed_time2 = t2-t1
        print(f"cos_sim2 経過時間：{elapsed_time2}") 
        print("cos_sim2={:.9f}".format(cossim2))

        t1 = time.time() 
        cossim3 = cos_sim3(newReIdVec,  globalReIdVec[i])
        t2 = time.time()
        elapsed_time3 = t2-t1
        print(f"cos_sim3 経過時間：{elapsed_time3}") 
        print("cos_sim3={:.9f}".format(cossim3))
        '''
        if cosSim > 0.7:
            globalReIdVec[i] = newReIdVec.copy()
            idx = i
            break

    print("idx=" + str(idx))
    if idx < size:
        return idx
    else:
        globalReIdVec.append(newReIdVec)
        return len(globalReIdVec) - 1

def cosineSimilarity(u, v):
    # print("U={},v={}".format(str(u.shape),str(v.shape)))
    return float(1 - distance.cosine(u, v))

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cos_sim3(vecA, vecB):
    mul = float(0)
    denomA = float(0)
    denomB = float(0)
    for i in range(vecA.shape[0]):
        A = vecA[i]
        B = vecB[i]
        mul = mul + A * B
        denomA = denomA + A * A
        denomB = denomB + B * B
        # print("A={0:.7f},B={1:.7f}".format(A,B))
    return mul / (math.sqrt(denomA) * math.sqrt(denomB))

#STEP-2
model_xml='C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/intel_models/person-vehicle-bike-detection-crossroad-0078/FP32/person-vehicle-bike-detection-crossroad-0078.xml'
model_bin='C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/intel_models/person-vehicle-bike-detection-crossroad-0078/FP32/person-vehicle-bike-detection-crossroad-0078.bin'
net = IENetwork.from_ir(model=model_xml, weights=model_bin)

model_reid_xml='C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/intel_models/person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.xml'
model_reid_bin='C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/intel_models/person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.bin'
net_reid = IENetwork.from_ir(model=model_reid_xml, weights=model_reid_bin)

plugin = IEPlugin(device='CPU', plugin_dirs=None)
plugin.add_cpu_extension('C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension.dll')

if "CPU" in plugin.device:
    supported_layers = plugin.get_supported_layers(net)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
        log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
        sys.exit(1)

#STEP-3
exec_net = plugin.load(network=net, num_requests=1)
exec_net_reid = plugin.load(network=net_reid, num_requests=1)

#STEP-4
input_blob = next(iter(net.inputs))  #input_blob = 'data'
out_blob   = next(iter(net.outputs)) #out_blob   = 'detection_out'
model_n, model_c, model_h, model_w = net.inputs[input_blob]  #model_n, model_c, model_h, model_w = 1, 3, 1024, 1024

input_blob_reid = next(iter(net_reid.inputs))  #input_blob = 'data'
out_blob_reid   = next(iter(net_reid.outputs)) #out_blob   = 'detection_out'
model_reid_n, model_reid_c, model_reid_h, model_reid_w = net_reid.inputs[input_blob_reid]  #model_n, model_c, model_h, model_w = 1, 3, 160, 64
# print(net_reid.inputs[input_blob_reid])

del net
del net_reid

#STEP-5
url = "http://192.168.1.16:8081/?action=stream"
videof = "video.mp4"
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #STEP-6
    cap_w = cap.get(3)
    cap_h = cap.get(4)
    in_frame = cv2.resize(frame, (model_w, model_h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((model_n, model_c, model_h, model_w))

    #STEP-7
    exec_net.start_async(request_id=0, inputs={input_blob: in_frame})

    if exec_net.requests[0].wait(-1) == 0:
        res = exec_net.requests[0].outputs[out_blob]

        #print(res.shape)
        #STEP-8
        for obj in res[0][0]:
            class_id = int(obj[1])
            if class_id == 1:
                if obj[2] > 0.8:
                    xmin = int(obj[3] * cap_w)
                    ymin = int(obj[4] * cap_h)
                    xmax = int(obj[5] * cap_w)
                    ymax = int(obj[6] * cap_h)
                    # Draw box and label\class_id

                    frame_org = frame.copy()
                    person = frame_org[ymin:ymax, xmin:xmax]
                    in_frame_reid = cv2.resize(person, (model_reid_w, model_reid_h))
                    in_frame_reid = in_frame_reid.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                    in_frame_reid = in_frame_reid.reshape((model_reid_n, model_reid_c, model_reid_h, model_reid_w))
                    cv2.imshow("Reid", person)

                    exec_net_reid.start_async(request_id=0, inputs={input_blob: in_frame_reid})

                    if exec_net_reid.requests[0].wait(-1) == 0:
                        res_reid = exec_net_reid.requests[0].outputs[out_blob_reid]
                        reIdVector = res_reid[0].reshape(-1,)
                        # print(reIdVector)
                        foundId = findMatchingPerson(reIdVector)
                        print("REID:" + str(foundId))

                    color = (255, 0, 0)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    # cv2.putText(frame, str(class_id) + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                    idColor = (0, 0, 255)
                    cv2.putText(frame, "ID_{0:4d}".format(foundId), (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.8, idColor, 1)

                
    #STEP-9
    cv2.imshow("Detection Results", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

print("MAX_REID={0}".format(len(globalReIdVec)))
#STEP-10
cv2.destroyAllWindows()
del exec_net
del exec_net_reid
del plugin