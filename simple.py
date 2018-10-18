#!/usr/bin/env python
#STEP-1
import sys
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IEPlugin

#STEP-2
model_xml='/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml'
model_bin='/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.bin'
net = IENetwork.from_ir(model=model_xml, weights=model_bin)
plugin = IEPlugin(device='MYRIAD', plugin_dirs=None)
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

#STEP-4
input_blob = next(iter(net.inputs))  #input_blob = 'data'
out_blob   = next(iter(net.outputs)) #out_blob   = 'detection_out'
model_n, model_c, model_h, model_w = net.inputs[input_blob]  #model_n, model_c, model_h, model_w = 1, 3, 384, 672

del net

#STEP-5
url = "http://192.168.1.16:8081/?action=stream"
cap = cv2.VideoCapture(url)
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
            if obj[2] > 0.5:
                xmin = int(obj[3] * cap_w)
                ymin = int(obj[4] * cap_h)
                xmax = int(obj[5] * cap_w)
                ymax = int(obj[6] * cap_h)
                class_id = int(obj[1])
                # Draw box and label\class_id
                color = (255, 0, 0)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, str(class_id) + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    #STEP-9
    cv2.imshow("Detection Results", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

#STEP-10
cv2.destroyAllWindows()
del exec_net
del plugin
