
import random as rand;
import numpy as np;
import math;
import sklearn; 
def dataPoints():
    data = [];
    for i in range(75):
        x = rand.uniform(0,1)
        h = rand.uniform(-0.1, 0.1) + 0.5 + 0.4 * math.sin(2*math.pi*x);
        data.append(np.array([x,h,0]))
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
    sumDistSquared=np.sum(math.pow(distances,2));
    variance = sumDistSquared/len(distances);
    return variance;
def updateCenter(points):
    meanPoint = np.sum(points,0);
    meanPoint = meanPoint/len(points);
    return meanPoint;
def gaussianFunction(x,width):
   return math.exp((-1/(2*width))*x);
def randWeights(baseNum):
    weightList = [];
    for i in range(baseNum):
        weightList.append(rand.uniform(-1,1));
    return np.array(weightList);


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
                newCenter=updateCenter(centerPoints);
                if(newCenter[0] != centers[i][0] or newCenter[1] != centers[i][1] ):
                    centerUpdated = True;
                    centers[i][0:2] = newCenter[0:2];
        epochs = 100;
        widths = np.zeros(bases[base]);
        averageWidths = [];
        for i in range(bases[base]):
            distances = []
            for distance in range(len(distanceCenter)):
                if distance[1] == i:
                    distances.append(distance[0]);
            if len(distances) >1:
                widths[i] = varianceCenter(distances);
            else:
                averageWidths.append(i);
        for i in averageWidths:
            widths[i] = np.sum(widths)/(len(widths)- len(averageWidths))
        weights= randWeights(bases[base]);
        for epoch in range(epochs):



    

