#!/usr/bin/env python
import argparse
import datetime
import os
import pathlib
import time

import cv2
import imutils


#function for image extraction
def extractImages(video_source_path,seconds,out_path):
    #works only on certain file formats with msecs, includes avi and mp4
    #default is .25 secs or 250 millisecs
    vidcap = cv2.VideoCapture(str(video_source_path))
    res, test = vidcap.read()
    count = 0
    #due to limitations of video containers and opencv
    #must first get total duration of video before extracting frames by sec
    
    
    
    
    n = seconds   # Desired interval of milisseconds to include
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frameCount/fps
    print(n)
    print(duration)

    if not res:
        raise Exception('empty video in file : ' + str(video_source_path))
    while res:
      msecs= (count*n)*1000
      vidcap.set(0,(msecs))      
      res, image = vidcap.read()
      print ('current at : ' +str(vidcap.get(cv2.CAP_PROP_POS_MSEC)))
      print ('count is: ' + str(count))
      print ('selection is: ' +str(msecs))
      print ('{}.count reading a new frame: {} '.format(count,res))
      cv2.imwrite(os.path.join(out_path, '{}_COUNT_{}.png'.format(os.path.basename(video_source_path),str(count))), image)     # save frame as PNG file

      count += 1
    vidcap.release()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--videos', help='path to directory with videos')
    a.add_argument('--seconds', help='interval of seconds', default=0.25)
    a.add_argument('--out', help='path to images')
    args = a.parse_args()

    for videopath in pathlib.Path(args.videos).glob('**/*'):
        folderpath = (os.path.join(args.out,os.path.basename((os.path.splitext(videopath)[0]))))
        print('saving to ' + str(folderpath))
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
        print(videopath.absolute())
        extractImages(videopath, args.seconds, folderpath)
