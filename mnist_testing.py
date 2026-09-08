import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from math import dist
from sklearn.model_selection import train_test_split
import math
import random


(train_x, train_y), (test_x, test_y) = mnist.load_data()

print('X_train: ' + str(train_x.shape))
print('y_train: ' + str(train_y.shape))
print('X_test: ' + str(test_x.shape))
print('y_test: ' + str(test_y.shape))

skeletonized_train = [skeletonize(train_x[i]) for i in range(len(train_x))]
skeletonized_test = [skeletonize(test_x[i]) for i in range(len(test_x))]


def whitePixels(skelPic):
    return np.argwhere(skelPic).tolist()

def circleIdentification(skelPic, position, visited, depth=0, max_depth=100):
    if depth > max_depth:
        return []
    if len(visited) > 15: #!!!!! can be changed
        for i in range(len(visited)-4):
            if dist(position, visited[i]) <= 1:
                return visited[i:]
    xPossibilities = [position[0] - 1, position[0], position[0] + 1]
    yPossibilities = [position[1] - 1, position[1], position[1] + 1]
    for xPos in xPossibilities:
        for yPos in yPossibilities:
            if xPos == position[0] and yPos == position[1]:
                continue
            try:
                if (skelPic[xPos][yPos] == True) and [xPos, yPos] not in visited[len(visited)-4:]:
                    visited.append([xPos, yPos])
                    possibleAnswer = circleIdentification(skelPic, [xPos, yPos], visited, depth + 1, max_depth)
                    if len(possibleAnswer) > 0:
                        return possibleAnswer
                    visited.pop()
            except IndexError:
                pass
    return []




#Intersection could be determined in two ways
#The first that came to mind was the intersection of already established lines.
#The problem that came with this was unsureness over the actual validity and requirement of leniency. There would be
#Cross-checking with the actual pixels and the expected.
#The other solution was looking at intersections. This seemed most to interact with pixels already labelled as eliminated,
#But for versatility I chose to code it as such.
def interestingPointIdentifier(skelPic): #for the sake of ease end and practice are both addressed by this function.!!!!!~
    endCoords = []
    whitePixels = []
    intersections = []
    corners = []
    alreadyin = False
    pathCount = 0 #counts the amount of bright pixels next to the current observed
    for i in range(len(skelPic)):
        for j in range(len(skelPic[i])): #Used to travel through each pixel
            xPossibilities = [i - 1, i, i + 1]
            yPossibilities = [j - 1, j, j + 1]
            if skelPic[i][j] == True:
                whitePixels = []
                pathCount = 0
                # Corners can be found in interacts of 2 surrounding paths. In most numbers the angle between corners are almost never
                # more than a bit over 90 degrees. Therefore I used specific code meant to account for such scenarios. This could be adjusted
                # For more leniency.
                for xPos in xPossibilities: #used to look at all pixels surrounding
                    for yPos in yPossibilities:
                        if xPos == i and yPos == j: #Skips over checked pixel
                            continue
                        try:
                            if skelPic[xPos][yPos] == True:
                                whitePixels.append([xPos, yPos])
                        except IndexError:
                            pass
                if (len(whitePixels) == 1): #This assumes an endpoint
                    endCoords.append([i,j])
                elif (len(whitePixels) >= 3):
                    #print(pathCount)
                    #print([i,j])
                    for L in intersections:
                        if dist(([i,j]), L) < 3: #This assumes intersections.
                            alreadyin = True
                    if alreadyin == False:
                        intersections.append([i,j])
                    alreadyin = False
                elif len(whitePixels) == 2:
                    #Twice, once for each pixel, look at that pixel. Lookat the surrounding pixel and if it has two paths only, look at the one that is not the original. Then compare distance.
                    p1 = whitePixels[0]
                    p2 = [i,j]
                    q1 = whitePixels[1]
                    testerArrayForCorner = []
                    for xpos in [q1[0]-1, q1[0], q1[0]+1]:
                        for ypos in [q1[1]-1, q1[1], q1[1]+1]:
                            if (xpos == q1[0] and ypos == q1[1]): #This ensures it skips over the checked surrounding pixel
                                continue
                            try:
                                if skelPic[xpos][ypos] == True:
                                    testerArrayForCorner.append([xpos, ypos])
                            except IndexError:
                                pass
                    if len(testerArrayForCorner) == 2:
                        if p2 in testerArrayForCorner:
                            testerArrayForCorner.remove(p2)
                        if p1 in testerArrayForCorner:
                            testerArrayForCorner.remove(p1)
                        try:
                            q2 = testerArrayForCorner[0]
                        except:
                            pass
                    try:
                        v1 = np.array([p2[1]-p1[1],p2[0]-p1[0]])
                        v2 = np.array([q2[1]-q1[1],q2[0]-q1[0]])
                        norm_v1 = np.linalg.norm(v1) #make the values closer to each other
                        norm_v2 = np.linalg.norm(v2)
                        if norm_v1 == 0 or norm_v2 == 0: #account for 0
                            continue
                        unit_v1 = v1/np.linalg.norm(v1)
                        unit_v2 = v2/np.linalg.norm(v2)
                        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1, 1)
                        angle_rad = np.arccos(dot_product)
                        angle_deg = np.degrees(angle_rad)
                        if angle_deg == 90:
                            #print(p1, p2, q1, q2)
                            corners.append([i, j])
                    except:
                        pass
                pathCount = 0
    return [endCoords, intersections, corners]


def lineCoords(skelPic, significantPoints, endpoints):  # !!!!!
    # endpoint establishment so that they don't cross reference
    # Equation Creation
    fullTime = len(significantPoints)
    currentPixelCount = 0
    totalAttempts = 0
    lineCount = 0
    checkList = []
    pixelList = []
    strictList = []
    failures = []
    endCheck = endpoints
    for i in range(fullTime):
        for j in range(i + 1, fullTime):
            checkList = []
            for k in range(abs(((significantPoints[i][0]) - (
            significantPoints[j][0])) + 1) * 10):  # 1 is used to account for if the other point is an endpoint
                if significantPoints[i][1] not in endpoints and significantPoints[j][1] not in endpoints:
                    try:
                        if significantPoints[i][1] < significantPoints[j][1]:
                            lower = significantPoints[i][1]
                        else:
                            lower = significantPoints[j][1]
                        slope = ((significantPoints[j][0]) - (significantPoints[i][0])) / (
                                    significantPoints[j][1] - significantPoints[i][1])
                        y = slope * ((lower + k * .1) - significantPoints[i][1]) + ((significantPoints[i][0]))
                        try:
                            if True == skelPic[round(y) + 1][round(lower + k * .1)]:  # Or go by amount counted, then
                                currentPixelCount += 1
                                checkList.append([round(y) + 1, round(lower + k * .1)])
                                if [round(y) + 1,round(lower + k * .1)] in endCheck:
                                    endCheck.remove([round(y) + 1,round(lower + k * .1)])
                            elif True == skelPic[round(y)][round(lower + k * .1)]:  # Or go by amount counted, then
                                currentPixelCount += 1
                                checkList.append([round(y), round(lower + k * .1)])
                                if [round(y),round(lower + k * .1)] in endCheck:
                                    endCheck.remove([round(y),round(lower + k * .1)])
                            elif True == skelPic[round(y) - 1,round(lower + k * .1)]:  # Or go by amount counted, then
                                currentPixelCount += 1
                                checkList.append([round(y) - 1, round(lower + k * .1)])
                                if [round(y) - 1,round(lower + k * .1)] in endCheck:
                                    endCheck.remove([round(y) - 1,round(lower + k * .1)])
                            else:
                                failures.append([round(y) + 1, round(lower + k * .1)])
                            totalAttempts += 1
                        except IndexError:
                            pass
                    except ZeroDivisionError:
                        if (significantPoints[j][0]) <= (significantPoints[i][0]):
                            lowerY = significantPoints[j][0]
                            higherY = significantPoints[i][0]
                        else:
                            lowerY = significantPoints[i][0]
                            higherY = significantPoints[j][0]
                        for l in range(lowerY, higherY):
                            if True == skelPic[significantPoints[i][1]][l]:
                                currentPixelCount += 1
                                pixelList.append([significantPoints[i][1], l])
                            if True == skelPic[significantPoints[i][1]-1][l]:
                                currentPixelCount += 1
                                pixelList.append([significantPoints[i][1], l])
                            if True == skelPic[significantPoints[i][1]+1][l]:
                                currentPixelCount += 1
                                pixelList.append([significantPoints[i][1], l])
            strictList = [list(p) for p in set(tuple(p) for p in pixelList)]
            try:
                if totalAttempts >= 10 and currentPixelCount / totalAttempts > .7:  # THIS IS ALARMING
                    lineCount += 1
                    pixelList += checkList
            except ZeroDivisionError:
                pass
            currentPixelCount = 0
            totalAttempts = 0
    for i in range(len(endCheck)):
        for j in range(i + 1, len(endCheck)):
            checkList = []
            for k in range(abs(((endCheck[i][0]) - (
            endCheck[j][0])) + 1) * 10):  # 1 is used to account for if the other point is an endpoint
                try:
                    if endCheck[i][1] < endCheck[j][1]:
                        lower = endCheck[i][1]
                    else:
                        lower = endCheck[j][1]
                    slope = ((endCheck[j][0]) - (endCheck[i][0])) / (
                            endCheck[j][1] - endCheck[i][1])
                    y = slope * ((lower + k * .1) - endCheck[i][1]) + ((endCheck[i][0]))
                    try:
                        if True == skelPic[round(y) + 1][round(lower + k * .1)]:
                            currentPixelCount += 1
                            checkList.append([round(y) + 1, round(lower + k * .1)])
                        elif True == skelPic[round(y)][round(lower + k * .1)]:
                            currentPixelCount += 1
                            checkList.append([round(y), round(lower + k * .1)])
                        elif True == skelPic[round(y) - 1][round(lower + k * .1)]:
                            currentPixelCount += 1
                            checkList.append([round(y) - 1, round(lower + k * .1)])
                        else:
                            failures.append([round(y) + 1, round(lower + k * .1)])
                        totalAttempts += 1
                    except IndexError:
                        pass
                except ZeroDivisionError:
                    if (endCheck[j][0]) <= (endCheck[i][0]):
                        lowerY = endCheck[j][0]
                        higherY = endCheck[i][0]
                    else:
                        lowerY = endCheck[i][0]
                        higherY = endCheck[j][0]
                    for l in range(lowerY, higherY):
                        if True == skelPic[significantPoints[i][1]][l]:
                            currentPixelCount += 1
                            pixelList.append([significantPoints[i][1], l])
                        if True == skelPic[significantPoints[i][1]-1][l]:
                            currentPixelCount += 1
                            pixelList.append([significantPoints[i][1], l])
                        if True == skelPic[significantPoints[i][1]+1][l]:
                            currentPixelCount += 1
                            pixelList.append([significantPoints[i][1], l])
            strictList = [list(p) for p in set(tuple(p) for p in pixelList)]
            try:
                if totalAttempts >= 10 and currentPixelCount / totalAttempts > .7:
                    #print(checkList)
                    #print(totalAttempts)
                    #print(currentPixelCount)
                    #print(len(checkList))
                    lineCount += 1
                    pixelList += checkList
            except ZeroDivisionError:
                pass
            currentPixelCount = 0
            totalAttempts = 0
    return [pixelList, lineCount]


def main(picture_data):
    interestingPoints = interestingPointIdentifier(picture_data)
    endpoints = interestingPoints[0]
    endCount = len(interestingPoints[0])
    intersections = interestingPoints[1]
    intersectionsCount = len(interestingPoints[1])
    corners = interestingPoints[2]
    cornersCount = len(interestingPoints[2])
    remainingWhite = whitePixels(picture_data)
    initialWhitePixelCount = len(whitePixels(picture_data))
    lineCount = 0
    loopCount = 0
    iAmTrying = (whitePixels(picture_data))
    result = circleIdentification(picture_data, iAmTrying[0], [iAmTrying[0]])
    if result:
        for j in result:
            try:
                remainingWhite.remove(j)
            except ValueError:
                pass
        loopCount = 1
    lineResults = lineCoords(picture_data, endpoints + intersections + corners, endpoints)
    lineCount = lineResults[1]
    result2 = lineResults[0]
    for j in range(len(result2)):
        if result2[j] in remainingWhite:
            remainingWhite.remove(result2[j])
    featureVector = ([float(len(remainingWhite)/initialWhitePixelCount), float(endCount), float(intersectionsCount), float(cornersCount), float(lineCount), float(loopCount)])
    return featureVector



training = []
test = []
traintracking = 0
testtracking = 0
for d in range(10):
    indexer = np.where(train_y == d)[0]
    testIndexer = np.where(test_y == d)[0]
    maxLength = len(indexer)
    np.random.shuffle(indexer)
    trainer = indexer
    tester = testIndexer
    training.append(trainer)
    test.append(tester)


#featureVectors will be used to hold all the vectors
featureVectors = [[] for _ in range(10)]
#finalVectors will have the average vector
finalVectors = []
for d in range(10):
    for i in training[d]:
        featureVectors[d].append(np.array(main(skeletonized_train[i]), dtype = float))#FLoat is to ensure values between 0 and 1
        traintracking += 1
        if traintracking % 100 == 0:
            print(traintracking)
digitVectors = np.vstack([np.array(vector) for vectors in featureVectors for vector in vectors])
min_vals = np.min(digitVectors, axis=0)
max_vals = np.max(digitVectors, axis=0)
#print(min_vals, max_vals)

for d in range(10):
    for i in range(len(featureVectors[d])):
        for j in range(6):
            featureVectors[d][i][j] = (featureVectors[d][i][j] - min_vals[j])/ (max_vals[j]- min_vals[j])###
    finalVectors.append(np.mean(featureVectors[d],axis=0))

#RatioWhite, ends, intersections, corners, lineAmount,loop

correctCount = 0
totalCount = 0


#accounts for the fact that test is already organized
for d in range(len(test)):
    for idx in test[d]:
        allDistances = []
        specificDistance = [] #Vector To Be Summed
        distance = 0
        result = main(skeletonized_test[idx])
        for e in range(6):
            result[e] = (result[e] - min_vals[e])/(max_vals[e]-min_vals[e])
        for f in range(10):
            specificDistance = []
            for g in range(6):
                specificDistance.append((result[g] - finalVectors[f][g]) ** 2)
            distance = np.sum(specificDistance)
            allDistances.append(distance)
        lowestValue = min(allDistances)
        prediction = allDistances.index(lowestValue)
        if prediction == d:
            correctCount += 1
        totalCount += 1
        testtracking += 1
        if testtracking % 100 == 0:
            print(testtracking)
print(correctCount/totalCount)
print("Final Vectors" +  str(np.array(finalVectors)))
print("Min vector:" +  str(np.min(finalVectors, axis=0)))
print("Max vector:" + str(np.max(finalVectors, axis=0)))
