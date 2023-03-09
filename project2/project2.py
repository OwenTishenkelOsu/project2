
import random as rand;
import numpy as np;
import math;
import matplotlib.pyplot as plt;
def dataPoints():
    data = [];
    for i in range(75):
        x = rand.uniform(0,1)
        h = rand.uniform(-0.1, 0.1) + 0.5 + 0.4 * math.sin(2*math.pi*x);
        data.append([x,h,0])
    data = np.array(data)
    return data;
def whichGaussianCenter(centers,point):
    center = 0;
    distance = math.sqrt(math.pow((point[0] - centers[center,0]),2) + math.pow((point[1] - centers[center,1]),2))
    currentCenter = 0;
    for i in centers:
        currentDistance = math.sqrt(math.pow((centers[currentCenter,0]-point[0]),2) + math.pow((centers[currentCenter,1] - point[1]),2))
        if(currentDistance < distance):
           center = currentCenter;
           distance =  currentDistance;
        currentCenter +=1;
    return (center,distance)
def varianceCenter(distances):
    sumDistSquared=np.sum(np.power(distances,2));
    variance = sumDistSquared/len(distances);
    return variance;
def updateCenter(points):
    meanPoint = np.sum(points,0);
    meanPoint = meanPoint/len(points);
    return np.array(meanPoint);
def gaussianFunction(x,center,width):
   return math.exp((-1/(2*width))*(math.pow(x-center,2)));
def randWeights(baseNum):
    weightList = [];
    for i in range(baseNum+1):
        weightList.append(rand.uniform(-1,1));
    return np.array(weightList);
def outputLayer(w,x):
    return np.sum(np.matmul(w,x));
def sampleFunction():
    data = [];
    inputVal = np.linspace(0,1,num=75);
    for i in range(75):
        h = 0.5 + 0.4 * math.sin(2*math.pi*inputVal[i]);
        data.append(np.array([inputVal[i],h]))
    data = np.array(data)
    return data;
def simpleWidth(centers):
    largestDist = 0;
    for point1 in centers:
        for point2 in centers:
            dist = math.sqrt(math.pow((point1[0]-point2[0]),2) + math.pow((point1[1]-point2[1]),2))
            if dist > largestDist:
                largestDist = dist;
    return math.pow(largestDist/len(centers),2);
bases = [2, 4, 7, 11, 16];
n = [0.01 , 0.02]
for rate in n:
    for base in range(len(bases)):

        data= dataPoints();
        centers= np.zeros((bases[base],2))
        for i in range(bases[base]):
            centers[i]=  data[rand.randint(0, len(data)-1)][0 : 2]
        distanceCenter = np.zeros((len(data),bases[base]))
        centerUpdated = True;
        while(centerUpdated):
            centerUpdated = False;
            for point in range(len(data)):
                data[point][2], distanceCenter[point,0] = whichGaussianCenter(centers,data[point][0:2])
                distanceCenter[point,1] = data[point][2];
            for i in range(bases[base]):
                centerPoints = [];
                for point in data:
                    if(point[2] ==i ):
                        centerPoints.append(point[0:2]);
                centerPoints= np.array(centerPoints)
                if(len(centerPoints)!=0):
                    newCenter=updateCenter(centerPoints);
                    if(newCenter[0] != centers[i][0] or newCenter[1] != centers[i][1]):
                        centerUpdated = True;
                        centers[i][0:2] = newCenter[0:2];
        epochs = 100;
        widths = np.zeros(bases[base]);
        averageWidths = [];
        for i in range(bases[base]):
            distances = []
            for distance in distanceCenter:
                if distance[1] == i:
                    distances.append(distance[0]);
            distances = np.array(distances)
            if len(distances) >1:
                widths[i] = varianceCenter(distances);
            else:
                averageWidths.append(i);
        for i in averageWidths:
            widths[i] = np.sum(widths)/(len(widths)- len(averageWidths))
        weights= randWeights(bases[base]);
        for epoch in range(epochs):
            for dataPoint in data:
                outputInner = [1]; #1 is bias for perceptron
                desired = 0.5 + 0.4 * math.sin(2*math.pi*dataPoint[0]);
                for i in range(bases[base]):
                    outputInner.append(gaussianFunction(dataPoint[0], centers[i][0],widths[i]))
                outputInner = np.array(outputInner)
                output = outputLayer(weights,outputInner);
                weights = weights + rate*(desired - output)* outputInner;
        inputData=np.linspace(0,1,num=75)
        functionOutput = np.zeros((75,2))
        for inputX in range(len(inputData)):
            outputInner = [1];
            for i in range(bases[base]):
                    outputInner.append(gaussianFunction(inputData[inputX], centers[i][0],widths[i]))
            functionOutput[inputX][0] = inputData[inputX];
            functionOutput[inputX][1] = outputLayer(weights,outputInner);
        plt.figure(base);
        sampleFunctionVals= sampleFunction();
        test = data[:,0:2];
        plt.plot(sampleFunctionVals[:,0],sampleFunctionVals[:,1],label = "Original Function" )
        plt.plot(functionOutput[:,0],functionOutput[:,1], label = "RBF Function")
        plt.plot(data[:,0],data[:,1], "r.")
        plt.legend()
        plt.title('Learning Curve of Base Number: '+str(bases[base])+" and Learning Rate: "+str(rate)+" with Differing widths");
        plt.show()

for rate in n:
    for base in range(len(bases)):

        data= dataPoints();
        centers= np.zeros((bases[base],2))
        for i in range(bases[base]):
            centers[i]=  data[rand.randint(0, len(data)-1)][0 : 2]
        distanceCenter = np.zeros((len(data),bases[base]))
        centerUpdated = True;
        while(centerUpdated):
            centerUpdated = False;
            for point in range(len(data)):
                data[point][2], distanceCenter[point,0] = whichGaussianCenter(centers,data[point][0:2])
                distanceCenter[point,1] = data[point][2];
            for i in range(bases[base]):
                centerPoints = [];
                for point in data:
                    if(point[2] ==i ):
                        centerPoints.append(point[0:2]);
                centerPoints= np.array(centerPoints)
                if(len(centerPoints)!=0):
                    newCenter=updateCenter(centerPoints);
                    if(newCenter[0] != centers[i][0] or newCenter[1] != centers[i][1]):
                        centerUpdated = True;
                        centers[i][0:2] = newCenter[0:2];
        epochs = 100;
        widths = np.zeros(bases[base]);
        averageWidths = [];
        simpleWidthVal = simpleWidth(centers)
        for i in range(bases[base]):
            widths[i] = simpleWidthVal;
        weights= randWeights(bases[base]);
        for epoch in range(epochs):
            for dataPoint in data:
                outputInner = [1]; #1 is bias for perceptron
                desired = 0.5 + 0.4 * math.sin(2*math.pi*dataPoint[0]);
                for i in range(bases[base]):
                    outputInner.append(gaussianFunction(dataPoint[0], centers[i][0],widths[i]))
                outputInner = np.array(outputInner)
                output = outputLayer(weights,outputInner);
                weights = weights + rate*(desired - output)* outputInner;
        inputData=np.linspace(0,1,num=75)
        functionOutput = np.zeros((75,2))
        for inputX in range(len(inputData)):
            outputInner = [1];
            for i in range(bases[base]):
                    outputInner.append(gaussianFunction(inputData[inputX], centers[i][0],widths[i]))
            functionOutput[inputX][0] = inputData[inputX];
            functionOutput[inputX][1] = outputLayer(weights,outputInner);
        plt.figure(base);
        sampleFunctionVals= sampleFunction();
        test = data[:,0:2];
        plt.plot(sampleFunctionVals[:,0],sampleFunctionVals[:,1],label = "Original Function" )
        plt.plot(functionOutput[:,0],functionOutput[:,1], label = "RBF Function")
        plt.plot(data[:,0],data[:,1], "r.")
        plt.legend()
        plt.title('Learning Curve of Base Number: '+str(bases[base])+" and Learning Rate: "+str(rate)+" with the same width");
        plt.show()
        


    

