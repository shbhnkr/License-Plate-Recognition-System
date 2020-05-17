import cv2
import numpy as np
import math
import random
import os
import csv
import time

import PossibleChar
import PossiblePlate
import main

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
MIN_PIXEL_AREA = 40
MIN_PIXEL_WIDTH = 1
MIN_PIXEL_HEIGHT = 4

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)

MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 0.8

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8

MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

MAX_CHANGE_IN_AREA = 0.6

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MIN_NUMBER_OF_MATCHING_CHARS = 3

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package) In our case it is a 3d array with dimensions-(460,640,3).
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""


def plate_detection(imgOriginalScene, frameId,timestamp):

    listOfPossiblePlates = detectPlatesInScene(imgOriginalScene)

    listOfPossiblePlates = detectCharsInPlates(listOfPossiblePlates)  # detect chars in plates
    if main.showSteps == True:
        cv2.imshow("imgOriginalScene", imgOriginalScene)  # show scene image

    if len(listOfPossiblePlates) == 0:  # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
    else:  # else
        # if we get in here list of possible plates has at leat one plate

        # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

        # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]
        if main.showSteps == True:
            cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
            cv2.imshow("imgThresh", licPlate.imgThresh)

        # if len(licPlate.strChars) == 0:  # if no chars were found in the plate
        #     print("\nno characters were detected\n\n")  # show message
        #     return  # and exit program
        # # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)  # draw red rectangle around plate

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)  # write license plate text on the image

        ##Write to the csv file Output.csv.
        if len(licPlate.strChars) != 0:
            with open('Output.csv', 'a', newline='') as f:
                thewriter = csv.writer(f)
                thewriter.writerow([licPlate.strChars, frameId,timestamp])

        if main.showSteps == True:
            cv2.imshow("imgOriginalScene", imgOriginalScene)  # re-show scene image
            print("djfiofj d")
            cv2.imwrite("imgOriginalScene.jpg", imgOriginalScene)  # write image out to file

    # end if else

    cv2.waitKey(0)  # hold windows open until user presses a key

    return


###################################################################################################

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)  # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0  # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0  # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX  # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0  # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))  # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)  # call getTextSize

    # unpack roatated rect into center point, width and height, and angle
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)  # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)  # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))  # write the chars in below the plate
    else:  # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))  # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize  # unpack text size width and height

    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX - (textSizeWidth / 2))  # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY + (textSizeHeight / 2))  # based on the text area center, width, and height

    # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)


# end function

###################################################################################################
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:  # if list of possible plates is empty
        return listOfPossiblePlates  # return
    # end if

    # at this point we can be sure the list of possible plates has at least one plate

    for possiblePlate in listOfPossiblePlates:  # for each possible plate, this is a big for loop that takes up most of the function

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = preprocess(
            possiblePlate.imgPlate)  # preprocess to get grayscale and threshold images

        if main.showSteps == True:  # show steps ###################################################
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        # end if # show steps #####################################################################

        # increase size of plate image for easier viewing and char detection
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx=1.6, fy=1.6)

        # threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0,
                                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if main.showSteps == True:  # show steps ###################################################
            cv2.imshow("5d", possiblePlate.imgThresh)
        # end if # show steps #####################################################################

        # find all possible chars in the plate,
        # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if main.showSteps == True:  # show steps ###################################################
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]  # clear the contours list

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, main.SCALAR_WHITE)

            cv2.imshow("6", imgContours)
        # end if # show steps #####################################################################

        # given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if main.showSteps == True:  # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("7", imgContours)
        # end if # show steps #####################################################################

        if (len(listOfListsOfMatchingCharsInPlate) == 0):  # if no groups of matching chars were found in the plate

            if main.showSteps == True:  # show steps ###############################################
                print("chars found in plate number " + str(
                    intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if # show steps #################################################################

            possiblePlate.strChars = ""
            continue  # go back to top of for loop
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):  # within each list of matching chars
            listOfListsOfMatchingCharsInPlate[i].sort(
                key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(
                listOfListsOfMatchingCharsInPlate[i])  # and remove inner overlapping chars
        # end for

        if main.showSteps == True:  # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("8", imgContours)
        # end if # show steps #####################################################################

        # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

        # suppose that the longest list of matching chars within the plate is the actual list of chars
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if main.showSteps == True:  # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, main.SCALAR_WHITE)

            cv2.imshow("9", imgContours)
        # end if # show steps #####################################################################

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if main.showSteps == True:  # show steps ###################################################
            print("chars found in plate number " + str(
                intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
        # end if # show steps #####################################################################

    # end of big for loop that takes up most of the function

    if main.showSteps == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates


# end function

###################################################################################################
# this is where we apply the actual char recognition
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""  # this will be the return value, the chars in the lic plate
    strJop = ""
    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR,
                 imgThreshColor)  # make color version of threshold image so we can draw contours in color on it
    for currentChar in listOfMatchingChars:  # for each char in plate
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth),
               (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, main.SCALAR_GREEN, 2)  # draw green box around the char

        # crop char out of threshold image
        imgROI = imgThresh[
                 currentChar.intBoundingRectY: currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                 currentChar.intBoundingRectX: currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (
            RESIZED_CHAR_IMAGE_WIDTH,
            RESIZED_CHAR_IMAGE_HEIGHT))  # resize image, this is necessary for char recognition

        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))  # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats

        # retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              # finally we can call findNearest !!!
        #
        # strCurrentChar = str(chr(int(npaResults[0][0])))            # get character from results
        #
        # strChars = strChars + strCurrentChar                        # append current char to full string        jop = 0
        # for i in range(1,18):
        # path = 'SameSizeLetters/'+str(i)+'.bmp'
        known = ['B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '0', '1', '2',
                 '3', '4', '5', '5', '6', '7', '8', '9']
        mse = 0
        jop = 0
        qwe = 1
        while qwe < 29:
            path = 'SameSizeLetters/' + str(qwe) + '.bmp'
            template = cv2.imread(path)
            # make border around template
            image = cv2.copyMakeBorder(template, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
            img = image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            image = cv2.Canny(image, 100, 200)
            contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            x, y, w, h = cv2.boundingRect(contours[0])
            crop_img = img[y:y + h, x:x + w]
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            Nimg = cv2.resize(gray, (30, 20))  # image resizing
            Nimg2 = cv2.resize(imgROIResized, (30, 20))
            sum = 0
            for x in range(len(Nimg)):
                for j in range(len(Nimg[x])):
                    if Nimg[x][j] == Nimg2[x][j]:
                        sum = sum + 1;

            if (sum > mse):
                mse = sum
                jop = qwe
            qwe = qwe + 1
            # err = np.sum((Nimg - Nimg2) ** 2)
            # err /= float(Nimg.shape[0] * Nimg.shape[1])
            # if(mse>err):
            #     mse = err
            #     jop = i
        # end for
        print(mse, " ", known[jop - 1])

        strJop = strJop +known[jop-1]

    if main.showSteps == True:  # show steps #######################################################
        cv2.imshow("10", imgThreshColor)
    # end if # show steps #########################################################################

    print('-'.isalpha())
    if(not len(strJop)==6):
        return ""
    for q in range(len(strJop)):
        if q + 3 < len(strJop):
            if (strJop[q]).isalpha() and (strJop[q + 1]).isalpha() and (strJop[q + 2]).isalpha() and (strJop[q + 3]).isalpha() and not strJop[q]=='-' and not strJop[q+1]=='-' and not strJop[q+2]=='-' and not strJop[q+3]=='-':
                strJop = strJop[0:q + 2] + '-' + strJop[q + 2:len(strJop)]
        if q + 1 < len(strJop):
            if (strJop[q].isalpha()) and (not strJop[q + 1].isalpha()) and not strJop[q]=='-' and not strJop[q+1]=='-':
                strJop = strJop[0:q + 1] + '-' + strJop[q + 1:len(strJop)]
            if (not strJop[q].isalpha()) and (strJop[q + 1].isalpha()) and not strJop[q]=='-' and not strJop[q+1]=='-':
                strJop = strJop[0:q + 1] + '-' + strJop[q + 1:len(strJop)]
    print(strJop)
    for q in range(len(strJop)):
        if q+2 <len(strJop):
            if strJop[q] == '-':
                if strJop[q+2]=='-':
                    return ""
    return strJop


# end function
###################################################################################################
# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)  # this will be the return value

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:  # if current char and other char are not the same char . . .
                # if current char and other char have center points at almost the same location . . .
                if distanceBetweenChars(currentChar, otherChar) < (
                        currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # if we get in here we have found overlapping chars
                    # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:  # if current char is smaller than other char
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:  # if current char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)  # then remove current char
                        # end if
                    else:  # else if other char is smaller than current char
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:  # if other char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)  # then remove other char
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved


# end function

###################################################################################################
###################################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []  # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

    # find all contours in plate
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:  # for each contour
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(
                possibleChar):  # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)  # add to list of possible chars
        # end if
    # end if

    return listOfPossibleChars


# end function

###################################################################################################

def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []  # this will be the return value

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if main.showSteps == True:  # show steps #######################################################
        cv2.imshow("0", imgOriginalScene)
    # end if # show steps #########################################################################

    imgGrayscaleScene, imgThreshScene = preprocess(imgOriginalScene)  # preprocess to get grayscale and threshold images

    if main.showSteps == True:  # show steps #######################################################
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)
    # end if # show steps #########################################################################

    # find all possible chars in the scene,
    # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    if main.showSteps2 == True:  # show steps #######################################################
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene)))  # 131 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, main.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)
        cv2.waitKey(0)
    # end if # show steps #########################################################################

    # given a list of all possible chars, find groups of matching chars
    # in the next steps each group of matching chars will attempt to be recognized as a plate
    listOfListsOfMatchingCharsInScene = findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if main.showSteps2 == True:  # show steps #######################################################
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))  # 13 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)
    # end if # show steps #########################################################################

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:  # for each group of matching chars
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)  # attempt to extract plate

        if possiblePlate.imgPlate is not None:  # if plate was found
            listOfPossiblePlates.append(possiblePlate)  # add to list of possible plates
        # end if
    # end for

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")  # 13 with MCLRNF1 image

    if main.showSteps == True:  # show steps #######################################################
        print("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), main.SCALAR_RED, 2)

            cv2.imshow("4a", imgContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        # end for

        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)
    # end if # show steps #########################################################################

    return listOfPossiblePlates


# end function


# end if # show steps #########################################################################


###################################################################################################
def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh


def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue


# end function


def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat


def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []  # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)  # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):  # for each contour

        if main.showSteps == True:  # show steps ###################################################
            cv2.drawContours(imgContours, contours, i, main.SCALAR_WHITE)
        # end if # show steps #####################################################################

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if checkIfPossibleChar(
                possibleChar):  # if contour is a possible char, note this does not compare to other chars (yet) . . .
            intCountOfPossibleChars = intCountOfPossibleChars + 1  # increment count of possible chars
            listOfPossibleChars.append(possibleChar)  # and add to list of possible chars
        # end if
    # end for

    if main.showSteps == True:  # show steps #######################################################
        print("\nstep 2 - len(contours) = " + str(len(contours)))  # 2362 with MCLRNF1 image
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))  # 131 with MCLRNF1 image
        cv2.imshow("2a", imgContours)
    # end if # show steps #########################################################################

    return listOfPossibleChars


# end function


def checkIfPossibleChar(possibleChar):
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if


# end function


def findListOfListsOfMatchingChars(listOfPossibleChars):
    # with this function, we start off with all the possible chars in one big list
    # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
    # note that chars that are not found to be in a group of matches do not need to be considered further
    listOfListsOfMatchingChars = []  # this will be the return value

    for possibleChar in listOfPossibleChars:  # for each possible char in the one big list of chars
        listOfMatchingChars = findListOfMatchingChars(possibleChar,
                                                      listOfPossibleChars)  # find all chars in the big list that match the current char

        listOfMatchingChars.append(possibleChar)  # also add the current char to current possible list of matching chars

        if len(
                listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:  # if current possible list of matching chars is not long enough to constitute a possible plate
            continue  # jump back to the top of the for loop and try again with next char, note that it's not necessary
            # to save the list in any way since it did not have enough chars to be a possible plate
        # end if

        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        listOfListsOfMatchingChars.append(listOfMatchingChars)  # so add to our list of lists of matching chars

        listOfPossibleCharsWithCurrentMatchesRemoved = []

        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(
            listOfPossibleCharsWithCurrentMatchesRemoved)  # recursive call

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:  # for each list of matching chars found by recursive call
            listOfListsOfMatchingChars.append(
                recursiveListOfMatchingChars)  # add to our original list of lists of matching chars
        # end for

        break  # exit for

    # end for

    return listOfListsOfMatchingChars


# end function


def findListOfMatchingChars(possibleChar, listOfChars):
    # the purpose of this function is, given a possible char and a big list of possible chars,
    # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    listOfMatchingChars = []  # this will be the return value

    for possibleMatchingChar in listOfChars:  # for each char in big list
        if possibleMatchingChar == possibleChar:  # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
            # then we should not include it in the list of matches b/c that would end up double including the current char
            continue  # so do not add to list of matches and jump back to top of for loop
        # end if
        # compute stuff to see if chars are a match
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(
            abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(
            possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(
            abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(
            possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(
            abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(
            possibleChar.intBoundingRectHeight)

        # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
                fltChangeInArea < MAX_CHANGE_IN_AREA and
                fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
                fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
            listOfMatchingChars.append(
                possibleMatchingChar)  # if the chars are a match, add the current char to list of matching chars
        # end if
    # end for

    return listOfMatchingChars  # return result
# end function













# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))


# end function

def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:  # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)  # if adjacent is not zero, calculate angle
    else:
        fltAngleInRad = 1.5708  # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)  # calculate angle in degrees

    return fltAngleInDeg


# end function


###################################################################################################
def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()  # this will be the return value

    listOfMatchingChars.sort(
        key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right based on x position

    # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[
                             0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = (
        tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg)

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape  # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))  # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped  # copy the cropped plate image into the applicable member variable of the possible plate

    return possiblePlate
# end function
