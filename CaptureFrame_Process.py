import cv2
import Localization
import time
import csv

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    # Create a VideoCapture object and read from input file
    with open('Output.csv', 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(['License plate', 'Frame no.', 'Timestamp(seconds)'])
    vidcap = cv2.VideoCapture(file_path)
    success, image = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalf = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FPS is ",fps)
    print("Processing frame number 0")
    abcd = "Total no of frames " + str(totalf)
    print(abcd)
    #multiplier = 30
    multiplier= sample_frequency
    count = 1

    # imgOriginalScene = cv2.imread("Stored/frame20.png")
    # Localization.plate_detection(imgOriginalScene)
    while (success):
        success, image = vidcap.read()
        frameId = int(round(vidcap.get(1)))
        timestamp=0
        if (frameId % multiplier == 0):
            fileName = "Stored/frame" + str(count)
            cv2.imwrite(fileName + ".png", image)
            print("Processing frame number ", frameId)
            count = count + multiplier

    fid = 1
    fileName = "Stored/frame" + str(fid)
    imgOriginalScene = cv2.imread(fileName + ".png")

    while (fid <= totalf - multiplier):
        Localization.plate_detection(imgOriginalScene, fid,fid/fps)
        fid = fid + multiplier
        fileName = "Stored/frame" + str(fid)
        imgOriginalScene = cv2.imread(fileName + ".png")

    vidcap.release()

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # out = cv2.VideoWriter('firstoutput.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, (frame_width, frame_height))

    # Check if file opened successfully
    # if (cap.isOpened() == False):
    #     print("Error opening video stream or file")

    # Read until video is completed
    #  while (cap.isOpened()):
    # Capture frame-by-frame
    #    ret, frame = cap.read()
    #   if ret == True:
    # Display the resulting frame
    # cv2.imshow('The Frames', frame)

    # Write the frame into the file 'firstoutput.avi'
    #      out.write(frame)

    # Display the resulting frame
    #     cv2.imshow('The frame being written', frame)

    # Press Q on keyboard to stop recording
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #       break
    # Press Q on keyboard to  exit
    #  if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

    # Break the loop
    # else:
    #   break
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    # When everything done, release the video capture object

    # out.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    # useCapturedFrames()

# def useCapturedFrames():
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#   cap = cv2.VideoCapture('firstoutput.avi')
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# Check if file opened successfully
#  if (cap.isOpened() == False):
#     print("Error opening video stream or file")

# Read until video is completed
# while (cap.isOpened()):
# Capture frame-by-frame
#   ret, frame = cap.read()
#  #if ret == True:
# Localization.plate_detection(frame)
# Break the loop
# else:
#   break
# cap.release()
# Closes all the frames
# cv2.destroyAllWindows()

