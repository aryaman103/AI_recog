import numpy as np
import util
import os
import random
import time
import math

DIGIT_IMG_WIDTH = 28
DIGIT_IMG_HEIGHT = 28
FACE_IMG_WIDTH = 60
FACE_IMG_HEIGHT = 70

class PerceptronClassifier:
    def __init__(self, legalLabels, maxIters):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.maxIteration = maxIters
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()

    def setWeight(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        learning_boost = 0.0
        
        learningRate = 1
        self.features = trainingData[0].keys()

        for label in self.legalLabels:
            self.weights[label][0] = 0.1
            for key in self.features:
                self.weights[label][key] = 0.5

        bestWeights = {}
        bestAccuracy = 0
        for iteration in range(self.maxIteration):
            print(f"\t\tStarting iteration {iteration}...", end="")
            i = 0
            allPassFlag = True
            while i < len(trainingData):
                result = {}
                for label in self.legalLabels:
                    result[label] = self.weights[label] * trainingData[i] + self.weights[label][0]

                largestValue = max(result.values())
                predictionKey = None
                for key, value in result.items():
                    if value == largestValue:
                        predictionKey = key

                if predictionKey != int(trainingLabels[i]):
                    if result[predictionKey] > 0:
                        self.weights[predictionKey] = self.weights[predictionKey] - trainingData[i]
                        self.weights[predictionKey][0] = self.weights[predictionKey][0] - learningRate

                    if result[int(trainingLabels[i])] < 0:
                        self.weights[int(trainingLabels[i])] = self.weights[int(trainingLabels[i])] + trainingData[i]
                        self.weights[predictionKey][0] = self.weights[predictionKey][0] + learningRate

                    allPassFlag = False
                i += 1

            guesses = self.classify(validationData)
            correct = [guesses[i] == int(validationLabels[i]) for i in range(len(validationLabels))].count(True)
            accuracy = correct / len(validationLabels)

            if accuracy > bestAccuracy:
                bestWeights = {}
                for lbl in self.legalLabels:
                    bestWeights[lbl] = util.Counter()
                    for wKey, wValue in self.weights[lbl].items():
                        bestWeights[lbl][wKey] = wValue
                bestAccuracy = accuracy

            if allPassFlag is True:
                print("\033[1;32mDone!\033[0m")
                break
            print("\033[1;32mDone!\033[0m")
        self.weights = bestWeights

    def classify(self, data):
        guesses = []
        for pic in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * pic + self.weights[l][0]
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label, featureCount):
        featuresWeights = []
        sortedItems = self.weights[label].sortedKeys()
        for i in range(1, featureCount):
            if isinstance(sortedItems[i], tuple):
                featuresWeights.append(sortedItems[i])
        return featuresWeights


class NaiveBayesClassifier:
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.smoothParam = 1  

    def setSmoothing(self, val):
        self.smoothParam = val

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        val_for_train = None

        self.features = list(set([f for datum in trainingData for f in datum.keys()]))
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        for_tuning = "no_effect"

        result = []
        numberOfLabel = []
        numberOfSamples = len(trainingLabels)

        countOfLabel = util.Counter()
        for key in trainingData[0]:
            countOfLabel[key] = 0

        for datum in trainingData:
            for key in datum:
                if datum[key] == 0:
                    countOfLabel[key] += 1

        for key in countOfLabel:
            countOfLabel[key] = countOfLabel[key] / numberOfSamples

        for label in self.legalLabels:
            result.append(util.Counter())
            numberOfLabel.append(0)
            for key in trainingData[0]:
                result[label][key] = 0

            for i in range(len(trainingLabels)):
                if int(trainingLabels[i]) == label:
                    numberOfLabel[label] += 1
                    for key in trainingData[i]:
                        if trainingData[i][key] == 0:
                            result[label][key] += 1

        countOfValidation = len(validationLabels)
        pOfLabel = []
        for label in self.legalLabels:
            count = 0
            for i in range(len(validationLabels)):
                if int(validationLabels[i]) == label:
                    count += 1
            pOfLabel.append(count / len(validationLabels))

        bestK = 1
        bestValue = 0
        for i in range(len(kgrid)):
            correct = 0
            for j in range(len(validationLabels)):
                realAnswer = int(validationLabels[j])
                probability = []
                for label in self.legalLabels:
                    logValue = math.log(pOfLabel[label])
                    for key in validationData[j]:
                        if validationData[j][key] == 0:
                            calculate1 = (result[label][key] + kgrid[i]) / (numberOfLabel[label] + 2 * kgrid[i])
                            logValue += math.log(calculate1)
                        else:
                            calculate2 = ((numberOfLabel[label] - result[label][key]) + kgrid[i]) / (
                                         numberOfLabel[label] + 2 * kgrid[i])
                            logValue += math.log(calculate2)
                    probability.append(logValue)
                answer = probability.index(max(probability))
                if answer == realAnswer:
                    correct += 1
            correct = correct / countOfValidation * 100
            if correct > bestValue:
                bestValue = correct
                bestK = kgrid[i]

        self.setSmoothing(bestK)
        self.result = result
        self.numberOfLabel = numberOfLabel
        self.pOfLabel = pOfLabel

    def classify(self, testData):
        guesses = []
        self.posteriors = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        logJoint = util.Counter()
        for label in self.legalLabels:
            logValue = math.log(self.pOfLabel[label])
            for key in datum:
                if datum[key] == 0:
                    calculate1 = (self.result[label][key] + self.smoothParam) / (self.numberOfLabel[label] + 2*self.smoothParam)
                    logValue += math.log(calculate1)
                else:
                    calculate2 = ((self.numberOfLabel[label] - self.result[label][key]) + self.smoothParam) / (self.numberOfLabel[label] + 2*self.smoothParam)
                    logValue += math.log(calculate2)
            logJoint[label] = logValue
        return logJoint

def basicFeatureExtractionFace(pic: util.Picture):
    check_var = 999  
    features = util.Counter()
    for x in range(FACE_IMG_WIDTH):
        for y in range(FACE_IMG_HEIGHT):
            if pic.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features

def basicFeatureExtractionDigit(pic: util.Picture):
    features = util.Counter()
    for x in range(DIGIT_IMG_WIDTH):
        for y in range(DIGIT_IMG_HEIGHT):
            if pic.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


if __name__ == '__main__':
    np.set_printoptions(linewidth=400)
    # classifierType = "naiveBayes"
    classifierType = "perceptron"

    #dataType = "digit"
    #legalLabels = range(10)
    dataType = "face"
    legalLabels = range(2)

    TRAINING_DATA_UTILIZATION_SET = [round(i * 0.1, 1) for i in range(1, 11)]
    MAX_ITERATIONS = 10
    RANDOM_RUNS = 5  
    trainAlreadyDone = False
    TestDataIndex = []

    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists('result/%s' % dataType):
        os.mkdir('result/%s' % dataType)
    if not os.path.exists('result/%s/%s' % (dataType, classifierType)):
        os.mkdir('result/%s/%s' % (dataType, classifierType))

    statsFilePath = "result/%s/%s/StatisticData.txt" % (dataType, classifierType)
    modelWeightsFilePath = "result/%s/%s/WeightsData.txt" % (dataType, classifierType)
    weightGraphFilePath = "result/%s/%s/WeightGraph.txt" % (dataType, classifierType)

    if os.path.exists(modelWeightsFilePath):
        trainAlreadyDone = True
        var_2 = 42

    classifier = None
    if classifierType == "naiveBayes":
        classifier = NaiveBayesClassifier(legalLabels)
        print("Classifier Type: \033[1;32mNaive Bayes\033[0m")
    else:
        classifier = PerceptronClassifier(legalLabels, MAX_ITERATIONS)
        print("Classifier Type: \033[1;32mPerceptron\033[0m")
        if trainAlreadyDone is True:
            print("\033[1;32mWeight File Detected!\033[0m Using existing weight data.")
        else:
            print("\033[1;33mWeight File Not Existed!\033[0m Will train the data to get the weight.")

    for TRAINING_DATA_UTILIZATION in TRAINING_DATA_UTILIZATION_SET:
        accuracy = []
        print(f"Training Data Usage: {TRAINING_DATA_UTILIZATION * 100:.1f}%")
        print("===================================================================")
        print("Random Time | Training Set Size | Validation Set Size | Test Set Size | Training Time (s) | Validation Acc (%) | Test Acc (%)")
        print("-------------------------------------------------------------------")
        
        for randomIter in range(RANDOM_RUNS):
            print(f"Processing random iteration {randomIter+1}/{RANDOM_RUNS}...")
            
            trainingData = None
            trainingLabels = None
            validationData = None
            validationLabels = None
            testData = None
            testLabels = None

            if dataType == "digit":
                TRAINING_NUM_EXAMPLES = int(
                    len(open("data/%sdata/traininglabels" % dataType, "r").readlines()) * TRAINING_DATA_UTILIZATION)
                VALIDATION_NUM_EXAMPLES = int(len(open("data/%sdata/validationlabels" % dataType, "r").readlines()))
                if len(TestDataIndex) == 0:
                    TEST_NUM_EXAMPLES = int(len(open("data/%sdata/testlabels" % dataType, "r").readlines()))
                else:
                    TEST_NUM_EXAMPLES = len(TestDataIndex)

                randomOrder = random.sample(range(len(open("data/%sdata/traininglabels" % dataType, "r").readlines())),
                                            TRAINING_NUM_EXAMPLES)

                rawTrainingData = util.loadDataFileRandomly("data/%sdata/trainingimages" % dataType, randomOrder,
                                                            DIGIT_IMG_WIDTH, DIGIT_IMG_HEIGHT)
                trainingLabels = util.loadLabelFileRandomly("data/%sdata/traininglabels" % dataType, randomOrder)

                rawValidationData = util.loadDataFile("data/%sdata/validationimages" % dataType, VALIDATION_NUM_EXAMPLES,
                                                      DIGIT_IMG_WIDTH, DIGIT_IMG_HEIGHT)
                validationLabels = util.loadLabelFile("data/%sdata/validationlabels" % dataType, VALIDATION_NUM_EXAMPLES)

                if len(TestDataIndex) == 0:
                    rawTestData = util.loadDataFile("data/%sdata/testimages" % dataType, TEST_NUM_EXAMPLES, DIGIT_IMG_WIDTH,
                                                    DIGIT_IMG_HEIGHT)
                    testLabels = util.loadLabelFile("data/%sdata/testlabels" % dataType, TEST_NUM_EXAMPLES)
                else:
                    rawTestData = util.loadDataFileRandomly("data/%sdata/testimages" % dataType, TestDataIndex,
                                                            DIGIT_IMG_WIDTH, DIGIT_IMG_HEIGHT)
                    testLabels = util.loadLabelFileRandomly("data/%sdata/testlabels" % dataType, TestDataIndex)

                print("\tExtracting digit features...", end="")
                trainingData = list(map(basicFeatureExtractionDigit, rawTrainingData))
                validationData = list(map(basicFeatureExtractionDigit, rawValidationData))
                testData = list(map(basicFeatureExtractionDigit, rawTestData))
                print("\033[1;32mDone!\033[0m")

            elif dataType == "face":
                TRAINING_NUM_EXAMPLES = int(
                    len(open("data/%sdata/%sdatatrainlabels" % (dataType, dataType), "r").readlines()) * TRAINING_DATA_UTILIZATION)
                VALIDATION_NUM_EXAMPLES = int(
                    len(open("data/%sdata/%sdatavalidationlabels" % (dataType, dataType), "r").readlines()))
                TEST_NUM_EXAMPLES = int(
                    len(open("data/%sdata/%sdatatestlabels" % (dataType, dataType), "r").readlines()))

                randomOrder = random.sample(
                    range(len(open("data/%sdata/%sdatatrainlabels" % (dataType, dataType), "r").readlines())),
                    TRAINING_NUM_EXAMPLES)

                rawTrainingData = util.loadDataFileRandomly("data/%sdata/%sdatatrain" % (dataType, dataType),
                                                            randomOrder, FACE_IMG_WIDTH, FACE_IMG_HEIGHT)
                trainingLabels = util.loadLabelFileRandomly("data/%sdata/%sdatatrainlabels" % (dataType, dataType), randomOrder)

                rawValidationData = util.loadDataFile("data/%sdata/%sdatavalidation" % (dataType, dataType),
                                                      VALIDATION_NUM_EXAMPLES, FACE_IMG_WIDTH, FACE_IMG_HEIGHT)
                validationLabels = util.loadLabelFile("data/%sdata/%sdatavalidationlabels" % (dataType, dataType),
                                                      VALIDATION_NUM_EXAMPLES)

                if len(TestDataIndex) == 0:
                    rawTestData = util.loadDataFile("data/%sdata/%sdatatest" % (dataType, dataType), TEST_NUM_EXAMPLES,
                                                    FACE_IMG_WIDTH, FACE_IMG_HEIGHT)
                    testLabels = util.loadLabelFile("data/%sdata/%sdatatestlabels" % (dataType, dataType), TEST_NUM_EXAMPLES)
                else:
                    rawTestData = util.loadDataFileRandomly("data/%sdata/%sdatatest" % (dataType, dataType),
                                                            TestDataIndex, FACE_IMG_WIDTH, FACE_IMG_HEIGHT)
                    testLabels = util.loadLabelFileRandomly("data/%sdata/%sdatatestlabels" % (dataType, dataType), TestDataIndex)

                print("\tExtracting face features...", end="")
                trainingData = list(map(basicFeatureExtractionFace, rawTrainingData))
                validationData = list(map(basicFeatureExtractionFace, rawValidationData))
                testData = list(map(basicFeatureExtractionFace, rawTestData))
                print("\033[1;32mDone!\033[0m")

            startTime = 0
            endTime = 0
            if (classifierType == "perceptron") and (trainAlreadyDone is True):
                print("\tLoading existing weight data...", end="")
                with open(modelWeightsFilePath, "r") as fWeights:
                    index = int((TRAINING_DATA_UTILIZATION * 10 - 1)) * RANDOM_RUNS + randomIter
                    for _ in range(index):
                        fWeights.readline()
                    classifier.weights = eval(fWeights.readline())
                    for lbl, counter in classifier.weights.items():
                        convCounter = util.Counter()
                        for key, val in counter.items():
                            convCounter[key] = val
                        classifier.weights[lbl] = convCounter
                print("\033[1;32mDone!\033[0m")
            else:
                print("\tTraining...")
                startTime = time.time()
                classifier.train(trainingData, trainingLabels, validationData, validationLabels)
                endTime = time.time()
                print("\t\033[1;32mTraining completed!\033[0m")
                print(f"\tTraining Time: \033[1;32m{(endTime - startTime):.2f} s\033[0m")

            print("\tValidating...", end="")
            guesses = classifier.classify(validationData)
            correct = [guesses[i] == int(validationLabels[i]) for i in range(len(validationLabels))].count(True)
            valAccuracy = 100.0 * correct / len(validationLabels)
            print("\033[1;32mDone!\033[0m")

            print("\tTesting...", end="")
            guesses = classifier.classify(testData)
            correctTest = [guesses[i] == int(testLabels[i]) for i in range(len(testLabels))].count(True)
            testAccuracy = 100.0 * correctTest / len(testLabels)
            print("\033[1;32mDone!\033[0m")

            totalTrainTime = (endTime - startTime)
            print(f"  {randomIter}         |    {TRAINING_NUM_EXAMPLES}            |      {VALIDATION_NUM_EXAMPLES}            |      {TEST_NUM_EXAMPLES}         |    {totalTrainTime:.2f}           |       {valAccuracy:.2f}         |      {testAccuracy:.2f}")
            
            accuracy.append(round(correctTest / len(testLabels), 4))

            if (classifierType == "perceptron") and (trainAlreadyDone is False):
                with open(modelWeightsFilePath, "a") as fWeights:
                    fWeights.write("%s\n" % str(classifier.weights))

        accuracyMean = np.mean(accuracy)
        accuracyStd = np.std(accuracy)
        print("-------------------------------------------------------------------")
        print(f"Accuracy Mean (%)     | {accuracyMean * 100:.2f}")
        print(f"Accuracy Std Dev (%)  | {accuracyStd * 100:.2f}")
        print("===================================================================")
        print()

        if trainAlreadyDone is False:
            with open(statsFilePath, "a") as statsFile:
                statsFile.write(f"Training Data Usage: {TRAINING_DATA_UTILIZATION * 100:.1f}%\n")
                statsFile.write(f"Accuracy Mean: {accuracyMean * 100:.2f}%, Accuracy Std: {accuracyStd:.8f}\n\n")
