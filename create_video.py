import cv2
import os
import glob

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('video.mp4', fourcc, 20.0, (1280, 960))

path = 'C://Users/tsuna/Downloads/0713Original/Original/'
files = []

for x in os.listdir(path):
    if os.path.isfile(path + x):  #isdirの代わりにisfileを使う
        files.append(x) 

file_list = sorted(glob.glob('C://Users/tsuna/Downloads/0713Original/Original/*.jpg'))

for filename in file_list:
    img = cv2.imread(filename)
    img = cv2.resize(img, (1280,960))
    video.write(img)

video.release()