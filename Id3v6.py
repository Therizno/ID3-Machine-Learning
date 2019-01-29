from math import *
from graphics import *


mainDataList = []       #list of all data from the train file
testDataList = []       #list of all data from the test file

predictColumn = 0       #the column that we're trying to predict

graphWidth = 1300
graphHeight = 650

originY = 10            #the y value where the visualization starts drawing

defaultPredict = "no"    #the value the prediction algorithm defaults to


#data processing

def fileOpen():

    global predictColumn

    file = open("example_data.csv", "r")

    for line in file:

        l = line.strip("\n").split(",")
        mainDataList.append(l)

    predictColumn = len(mainDataList[0])-1        #assumes the column we're
                                                  #trying to predict is the last
                                                  #column


def testOpen():

    file = open("example_test_data.csv", "r")

    for line in file:

        l = line.strip("\n").split(",")
        testDataList.append(l)



########



#classes


class DataSplit:

    "Subdivisions of the data by feature"

    infoGain = 0
    dataList = []
    splitCriteria = ""
    
    def __init__(self, pInfoGain, pDataList, pSplitCriteria):
        self.infoGain = pInfoGain
        self.dataList = pDataList
        self.splitCriteria = pSplitCriteria

        

########

            



#function definitions


#takes a list of lists of data values, returns a float
def entropy(pList):

    ent = 0

    countDict = {}

    for item in pList:              #counting values for probability

        if item[predictColumn] in countDict:       

            countDict[item[predictColumn]] += 1

        else:

            countDict[item[predictColumn]] = 1


    for value in countDict:         #entropy calculation

        prob = countDict[value]/len(pList)

        try:
            ent -= prob * log(prob, 2)
        except ValueError:
            ent -= 0

    
    return ent




def information_gain(dataL, col):

    sumEntropy = 0
    hL = []
    splitOn = []
    hList = []
    hDict = {}
    
    hDict = split(dataL, col)       #stores all data split by unique data value


    for key in hDict:
                                    #turns the dict into two lists, one of the
        hList.append(hDict[key])    #data, and the other of the categories that                                   
        splitOn.append(key)         #the data was split on
        

    for h in hList:

        sumEntropy += (len(h)/len(dataL)) * entropy(h)
        hL.append(h)

    ig = entropy(dataL) - sumEntropy


    return DataSplit(ig, hL, (col, splitOn))


    

def split(dataL, colNum):

    uniqueValues = {}       #dictionary of unique values in the column

    for row in dataL:

        if row[colNum] in uniqueValues:

            uniqueValues[row[colNum]].append(row)

        else:

            uniqueValues[row[colNum]] = [row]


    return uniqueValues





#gives the value at the end of the branch.
#if the branch is pure, will return the expected value.
#otherwise, returns the modal value.
def endOfTree(dataL):

    predictCount = {}
    maxMode = 0
    leafValue = ""

    for row in dataL:

        if row[predictColumn] in predictCount:      #counts rows of each
                                                    #predicted value
            predictCount[row[predictColumn]] += 1

        else:
            predictCount[row[predictColumn]] = 1


    for item in predictCount:

        if predictCount[item] > maxMode:

            leafValue = item
            maxMode = predictCount[item]
            

    assert leafValue != ""

    return leafValue


        


def buildTree(dataL):

    maxInfoGain = 0
    splitCandidate = ""


    for i in range(predictColumn - 1):          #assumes the column we're trying
                                                #to predict is the last column
        
        splitObject = information_gain(dataL, i)

        if splitObject.infoGain > maxInfoGain:  #finds the best split candidate

            splitCandidate = splitObject
            maxInfoGain = splitObject.infoGain


    if maxInfoGain == 0:

        return endOfTree(dataL)


    #print("splitting on:", splitCandidate.splitCriteria)
    #print("information gained:", maxInfoGain)


    splitList = []

    for split in splitCandidate.dataList:

        splitList.append(buildTree(split))

            
    return DataSplit(maxInfoGain, splitList, splitCandidate.splitCriteria)
    


def displayTree(treeObj):

    yIncr = 50

    win = GraphWin("DECISION TREE", graphWidth, graphHeight)

    displayBranch(graphWidth/2, originY, treeObj, win, "", graphWidth, 0)



def displayBranch(centerX, y, tree, win, lastSplit, graphW, xMin):

    if isinstance(tree, DataSplit):

        displayString = "split on: "+str(tree.splitCriteria[0])

        printNode(int((y-originY)/60), lastSplit + " " + displayString)

        partition = graphW/len(tree.dataList)
        newX = 0

        for i in range(len(tree.dataList)):

            newX += partition

            disX = xMin + newX - partition/2

            l = Line(Point(centerX, y+10), Point(disX, y+35))
            l.draw(win)
            displayBranch(disX, y+60, tree.dataList[i], win, tree.splitCriteria[1][i], graphW/len(tree.dataList), xMin + i*graphW/len(tree.dataList))

    else:

        displayString = tree
        printNode(int((y-originY)/60), "result: "+tree)


    lastSplitMessage = Text(Point(centerX, y - 15), lastSplit)
    lastSplitMessage.draw(win)
        
    splitMessage = Text(Point(centerX, y), displayString)
    splitMessage.draw(win)

    


def printNode(depth, message):

    string = ""

    for i in range(depth):

        string += "     "

    string += message

    print(string)




def predictLine(line, tree):

    if isinstance(tree, DataSplit):

        splitOnLine = tree.splitCriteria[0]

        for i in range(len(tree.splitCriteria[1])):

            if line[splitOnLine] == tree.splitCriteria[1][i]:

                return predictLine(line, tree.dataList[i])
            
        return defaultPredict

    else:

        return tree





def predictData(data, tree):

    trueAssesments = 0
    n = 0

    for line in data:

        pred = predictLine(line, tree)

        if pred == line[len(line)-1]:

            trueAssesments += 1

        n += 1

    return trueAssesments/n



 ########           

    





#call functions

fileOpen()
testOpen()

print("predicting column:", predictColumn)
print("data entropy:", entropy(mainDataList))


fTree = buildTree(mainDataList)
displayTree(fTree)

print("accuracy:", predictData(testDataList, fTree))



########


