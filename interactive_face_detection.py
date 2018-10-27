#!/usr/bin/env python
#STEP-1
import sys
import logging as log
import datetime
import time
import os
import json

from time import sleep
from scipy.spatial import distance
import cv2
from argparse import ArgumentParser

from openvino.inference_engine import IENetwork, IEPlugin
from azure.eventhub import EventHubClient, Sender, EventData


globalReIdVec = []
def findMatchingPerson(newReIdVec):
    global globalReIdVec
    size = len(globalReIdVec)
    print("size=" + str(size))

    idx = size
    for i in range(size):
        # t1 = time.time() 
        cosSim = cosineSimilarity(newReIdVec, globalReIdVec[i])

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
    return float(1 - distance.cosine(u, v))

def build_argparser():
    parser = ArgumentParser()
    # parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. 'cam' for capturing video stream from camera", required=True,
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                        default=0.5, type=float)
    parser.add_argument("-ad", "--address", help="", required=True, type=str)
    parser.add_argument("-u", "--user", help="", required=True, type=str)
    parser.add_argument("-k", "--key", help="", required=True, type=str)

    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    #STEP-2
    model_xml='C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml'
    model_bin='C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.bin'
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)

    model_age_xml='C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/intel_models/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml'
    model_age_bin='C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/intel_models/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin'
    net_age = IENetwork.from_ir(model=model_age_xml, weights=model_age_bin)

    model_reid_xml='C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/intel_models/face-reidentification-retail-0001/FP32/face-reidentification-retail-0001.xml'
    model_reid_bin='C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/intel_models/face-reidentification-retail-0001/FP32/face-reidentification-retail-0001.bin'
    net_reid = IENetwork.from_ir(model=model_reid_xml, weights=model_reid_bin)

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        # plugin.add_cpu_extension(args.cpu_extension)
        plugin.add_cpu_extension('C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension.dll')


    # plugin = IEPlugin(device='CPU', plugin_dirs=None)
    # plugin.add_cpu_extension('C:/Intel/computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension.dll')
    # plugin.add_cpu_extension('/opt/intel/computer_vision_sdk/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension.so')

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
    exec_net_age = plugin.load(network=net_age, num_requests=1)
    exec_net_reid = plugin.load(network=net_reid, num_requests=1)

    #STEP-4
    input_blob = next(iter(net.inputs))  #input_blob = 'data'
    out_blob   = next(iter(net.outputs)) #out_blob   = 'detection_out'
    model_n, model_c, model_h, model_w = net.inputs[input_blob]  #model_n, model_c, model_h, model_w = 1, 3, 300, 300

    input_blob_age = next(iter(net_age.inputs))  #input_blob = 'data'
    out_blob_age   = next(iter(net_age.outputs)) #out_blob   = ''
    out_blob_prob   = "prob"
    model_age_n, model_age_c, model_age_h, model_age_w = net_age.inputs[input_blob_age]  #model_n, model_c, model_h, model_w = 1, 3, 62, 62

    input_blob_reid = next(iter(net_reid.inputs))  #input_blob = 'data'
    out_blob_reid   = next(iter(net_reid.outputs)) #out_blob   = ''
    model_reid_n, model_reid_c, model_reid_h, model_reid_w = net_reid.inputs[input_blob_reid]  #model_n, model_c, model_h, model_w = 1, 3, 62, 62
    print("B={},C={},H={},W={}".format(model_reid_n, model_reid_c, model_reid_h, model_reid_w))

    del net
    del net_age
    del net_reid
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    
    #STEP-5
    url = "http://192.168.1.16:8081/?action=stream"
    video = "video.mp4"
    cap = cv2.VideoCapture(input_stream)

    personData = {}
    ADDRESS = args.address
    USER = args.user
    KEY = args.key
    client = EventHubClient(ADDRESS, debug=False, username=USER, password=KEY)
    sender = client.add_sender(partition="0")
    client.run()

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

            #STEP-8
            for obj in res[0][0]:
                if obj[2] > args.prob_threshol:
                    xmin = int(obj[3] * cap_w)
                    ymin = int(obj[4] * cap_h)
                    xmax = int(obj[5] * cap_w)
                    ymax = int(obj[6] * cap_h)

                    frame_org = frame.copy()
                    face = frame_org[ymin:ymax, xmin:xmax]

                    in_frame_age = cv2.resize(face, (model_age_w, model_age_h))
                    in_frame_age = in_frame_age.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                    in_frame_age = in_frame_age.reshape((model_age_n, model_age_c, model_age_h, model_age_w))
                    # cv2.rectangle(dst,(xmin, ymin), (xmax, ymax), (255, 0 ,0), 1)
                    cv2.imshow("age", face)

                    exec_net_age.start_async(request_id=0, inputs={input_blob: in_frame_age})

                    if exec_net_age.requests[0].wait(-1) == 0:
                        res_gender = exec_net_age.requests[0].outputs[out_blob_prob]
                        # res_age = exec_net_age.requests[0].outputs[out_blob_age]
                        # print(res_age[0].reshape(-1,))
                        AgeGender = res_gender[0].reshape(-1,)
                        # print(AgeGender)

                    in_frame_reid = cv2.resize(face, (model_reid_w, model_reid_h))
                    in_frame_reid = in_frame_reid.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                    in_frame_reid = in_frame_reid.reshape((model_reid_n, model_reid_c, model_reid_h, model_reid_w))
                    exec_net_reid.start_async(request_id=0, inputs={input_blob: in_frame_reid})
                    if exec_net_reid.requests[0].wait(-1) == 0:
                        res_reid = exec_net_reid.requests[0].outputs[out_blob_reid]
                        reIdVector = res_reid[0].reshape(-1,)
                        # print(reIdVector)
                        foundId = findMatchingPerson(reIdVector)
                    
                        print("ID:" + str(foundId))

                    class_id = int(obj[1])
                    # Draw box and label\class_id
                    color = (0, 0, 255)
                    genderColor = (147, 20, 255) if AgeGender[1] < 0.5 else (255, 0, 0)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), genderColor, 2)
                    # cv2.putText(frame, str(class_id) + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1)
                    cv2.putText(frame, "ID_{0:4d}".format(foundId), (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1)

                    try:
                        personData["detecttime"] = datetime.datetime.now().isoformat()
                        personData["No"] = str(1)
                        personData["faceId"] = str(foundId)
                        message = json.dumps(personData)
                        sender.send(EventData(message))
                    except:
                        raise

    #STEP-9
        cv2.imshow("Detection Results", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    #STEP-10
    cv2.destroyAllWindows()
    del exec_net
    del plugin
    client.stop()

if __name__ == '__main__':
    sys.exit(main() or 0)