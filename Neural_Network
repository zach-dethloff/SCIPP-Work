import csv, sys
import uproot
import pandas as pd
import numpy as np
from numpy import array
import subprocess

np.set_printoptions(threshold=sys.maxsize)
import shap
import tensorflow as tf
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
import os

# don't use these in notebook mode
#from matplotlib.backends.backend_pdf import PdfPages
#matplotlib.use("PDF")
import math
import time
from math import log, sqrt
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  # Normalized data to range from (0,1)
from sklearn.metrics import (
    precision_recall_curve,
    plot_precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix
)
from datetime import datetime

# Checking if a GPU is available, not sure it will run in Jupyter
status = len(tf.config.experimental.list_physical_devices("GPU"))

# If we need a random seed.
seed = 42

def plotPR(x, y, t):
    plt.subplot(411)
    plt.plot(t, x[:-1], "b--", label="Precision")
    plt.plot(t, y[:-1], "g-", label="Recall")
    plt.ylim([0.00, 1.05])
    plt.xlabel("Threshold")
    plt.title("Precision/Recall vs. Threshold Curve")
    plt.legend(loc="lower right")
    plt.grid()


def plotROC(x, y, AUC):
    plt.subplot(412)
    plt.plot(x, y, lw=1, label="ROC (area = %0.6f)" % (AUC))
    plt.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.grid()


def getZPoisson(s, b, stat, syst):
    """
    The significance for optimisation.
    s: total number of signal events
    b: total number of background events
    stat: relative MC stat uncertainty for the total bkg.
    syst: relative syst uncertainty on background
    Note that the function already accounts for the sqrt(b)
    uncertainty from the data, so we only need to pass in additional
    stat and syst terms.  e.g. the stat term above is really only
    characterizing the uncertainty due to limited MC statistics used
    to estimate the background yield.
    """
    n = s + b

    # this is a relative uncertainty
    sigma = math.sqrt(stat ** 2 + syst ** 2)

    # turn into the total uncertainty
    sigma = sigma * b

    if s <= 0 or b <= 0:
        return 0

    factor1 = 0
    factor2 = 0

    if sigma < 0.01:
        # In the limit where the total BG uncertainty is zero,
        # this reduces to approximately s/sqrt(b)
        factor1 = n * math.log((n / b))
        factor2 = n - b
    else:
        factor1 = n * math.log((n * (b + sigma ** 2)) / ((b ** 2) + n * sigma ** 2))
        factor2 = ((b ** 2) / (sigma ** 2)) * math.log(
            1 + ((sigma ** 2) * (n - b)) / (b * (b + sigma ** 2))
        )

    signif = 0
    try:
        signif = math.sqrt(2 * (factor1 - factor2))
    except ValueError:
        signif = 0

    return signif
    
    
  def Event_Combination(Input_Directory, Background = False, Slice = False):
    Directories = os.listdir(Input_Directory)
    TotalEvents = 0
    HistogramArray = []
    PathArray = []
    Branches = []
    CrossSections = []
    MET = []
    j1PT = []
    mjj = []
    j1Eta = []
    j1Phi = []
    j2PT = []
    j2Eta = []
    j2Phi = []
    weight = []
    Scales = []
    METPhi = []
    j3PT = []
    j3Eta = []
    j3Phi = []

    #Test Directories to see if they actually contain a valid histogram file
    for item in Directories:
        composite = ["""grep "Cross-section" """+Input_Directory+item+"/docker_mgpy.log"+"| tail -1 | awk '{print $8}'"]
        CrossSectionOutput = subprocess.run(composite, shell=True, capture_output=True)
        if os.path.exists(Input_Directory+item+"/analysis/histograms.root") != True and os.path.exists(Input_Directory+item+"/analysis/SimpleAna.root") != True:
            Directories.remove(item)
            print("Error, Histogram not found for"+Input_Directory+item+"/analysis/histograms.root")
    #Test Directories to see if they contain a valid Cross Section Output, if not, give feedback and omit directory
    for item in Directories:
        composite = ["""grep "Cross-section" """+Input_Directory+item+"/docker_mgpy.log"+"| tail -1 | awk '{print $8}'"]
        CrossSectionOutput = subprocess.run(composite, shell=True, capture_output=True)
        try:
            float(CrossSectionOutput.stdout)
        except:
            Directories.remove(item)
            Statement = "File "+item+" Unable to be combined, could not find Cross Section"
            print(Statement)
    #Ignore any background directories not of the current slice
    for item in Directories:
        composite = ["""grep "Cross-section" """+Input_Directory+item+"/docker_mgpy.log"+"| tail -1 | awk '{print $8}'"]
        CrossSectionOutput = subprocess.run(composite, shell=True, capture_output=True)
        if Slice == 1:
            if "1000_4000" in item:
                a=1
            else:
                Directories.remove(item)
        elif Slice == 2:
            if "4000_7000" in item:
                a=1
            else:
                Directories.remove(item)
        elif Slice == 3:
            if "7000_10000" in item:
                a=1
            else:
                Directories.remove(item)
            
        elif Slice == 4:
            if "10000_-1" in item:
                a=1
            else:
                Directories.remove(item)
        elif Slice == False:
            a=1
    #Add cross sections and valid histograms to files
    for item in Directories:
        composite = ["""grep "Cross-section" """+Input_Directory+item+"/docker_mgpy.log"+"| tail -1 | awk '{print $8}'"]
        CrossSectionOutput = subprocess.run(composite, shell=True, capture_output=True)
        CrossSections.append(float(CrossSectionOutput.stdout))
        if os.path.exists(Input_Directory+item+"/analysis/histograms.root") == True:
            HistogramArray.append(Input_Directory+item+"/analysis/histograms.root")
        elif os.path.exists(Input_Directory+item+"/analysis/SimpleAna.root") == True:
            HistogramArray.append(Input_Directory+item+"/analysis/SimpleAna.root")
    #Apply a mask cut and add remaining events to output arrays
    for item in HistogramArray:
        PathArray.append(uproot.open(item)['allev/hftree'])
    for item in PathArray:
        Branches.append(item.arrays())
        
    #appending relevant information to directories

    for item in Branches:
        mask = (item[b"mjj"] > 1000)&(item[b"MET"] > 200)&(item[b"njet"] >= 2)&(item[b"nElec"] == 0)&(item[b"nMuon"] == 0)
        for element in item[b"MET"][mask]:
            MET.append(element)
        for element in item[b"j1PT"][mask]:
            j1PT.append(element)
        for element in item[b"mjj"][mask]:
            mjj.append(element)
        for element in item[b"j1Eta"][mask]:
            j1Eta.append(element)
        for element in item[b"j1Phi"][mask]:
            j1Phi.append(element)
        for element in item[b"j2PT"][mask]:
            j2PT.append(element)
        for element in item[b"j2Eta"][mask]:
            j2Eta.append(element)
        for element in item[b"j2Phi"][mask]:
            j2Phi.append(element)
        for element in item[b"METPhi"][mask]:
            METPhi.append(element)
        for element in item[b"j3PT"][mask]:
            j3PT.append(element)
        for element in item[b"j3Eta"][mask]:
            j3Eta.append(element)
        for element in item[b"j3Phi"][mask]:
            j3Phi.append(element)
        #Take the weights and scale them by number of inputs
    i=0 
    for item in Branches:
        mask = (item[b"mjj"] > 1000)&(item[b"MET"] > 200)&(item[b"njet"] >= 2)&(item[b"nElec"] == 0)&(item[b"nMuon"] == 0)
        scale = (CrossSections[i]/sum(item[b"weight"]))
        i += 1
        Scales.append(scale)
        Scaled_weight = item[b"weight"][mask]*scale/len(Directories)
        for item in Scaled_weight:
            weight.append(item)
    TotalEvents = len(MET)
    #Use this Output function to implement and variables you want to calculate from this Combination Function
    Output = {"MET": MET, "j1PT":j1PT, 
              "mjj":mjj, "j1Eta":j1Eta, "j1Phi":j1Phi, "j2PT":j2PT,
              "j2Eta":j2Eta, "j2Phi":j2Phi, "weight":weight, "Events":TotalEvents, 'METPhi':METPhi, "j3PT":j3PT, "j3Eta":j3Eta,
             "j3Phi":j3Phi}
    return(Output)
    
    def Background(Input_Directory):
    Directories = []
    TotalEvents = 0
    CrossSections = []
    MET = []
    j1PT = []
    mjj = []
    j1Eta = []
    j1Phi = []
    j2PT = []
    j2Eta = []
    j2Phi = []
    METPhi = []
    weight = []
    j3PT = []
    j3Eta = []
    j3Phi = []
    Slices = [1, 2, 3, 4]
    for item in Slices:
        partition = Event_Combination(Input_Directory,True, item)
        for item in partition["MET"]: 
            MET.append(item)
        for item in partition["j1PT"]: 
            j1PT.append(item)
        for item in partition["mjj"]: 
            mjj.append(item)
        for item in partition["j1Eta"]: 
            j1Eta.append(item)
        for item in partition["j1Phi"]: 
            j1Phi.append(item)
        for item in partition["j2PT"]: 
            j2PT.append(item)
        for item in partition["j2Eta"]: 
            j2Eta.append(item)
        for item in partition["j2Phi"]: 
            j2Phi.append(item)
        for item in partition["weight"]: 
            weight.append(item)
        for item in partition['METPhi']:
            METPhi.append(item)
        TotalEvents+=partition['Events']
        for item in partition['j3PT']:
            j3PT.append(item)
        for item in partition['j3Eta']:
            j3Eta.append(item)
        for item in partition['j3Phi']:
            j3Phi.append(item)
    Output = {"MET": MET, "j1PT":j1PT, 
              "mjj":mjj, "j1Eta":j1Eta, "j1Phi":j1Phi, "j2PT":j2PT,
              "j2Eta":j2Eta, "j2Phi":j2Phi, "weight":weight, "Events":TotalEvents, "METPhi":METPhi, "j3PT":j3PT, "j3Eta":j3Eta,
             "j3Phi":j3Phi}
    return(Output)

#Need 3rd jet info

# Data read from file.
tree = 'allev/hftree'

#paths to signal and background directories

filepath="/data/users/jupyter-blonsbro/SUSY"

signal_path = filepath+'/Generations/13TeV/Signal/150mjj/'
EWK_path = filepath+'/EWKBDirect/'
QCD_path = filepath+'/QCDBDirect/'

#sending data through sorting and weighting functions

Signal = Event_Combination(signal_path)
EWK = Background(EWK_path)
QCD = Background(QCD_path)

print('Done!')

#turning important data into dataframe to be easily read or transferred if need be

list_signal = pd.DataFrame(Signal)
list_ewk = pd.DataFrame(EWK)
list_qcd = pd.DataFrame(QCD)

sig_mjj = list_signal['mjj']
s_w = list_signal['weight']

#determining input neuron count

numBranches = len(list_signal.keys()) - 3

#getting counts for each input

nsig = len(list_signal['MET'])
nEWKbkg = len(list_ewk['MET'])
nQCDbkg = len(list_qcd['MET'])


df_background = list_ewk.append(list_qcd, ignore_index=True)

nbkg = len(df_background['MET'])


# The 3 backgrounds are concatenated we shuffle to make sure they are not sorted.
shuffleBackground = shuffle(df_background, random_state=seed)

bkg_mjj = shuffleBackground['mjj']
b_w = shuffleBackground['weight']

# Signal and shuffle background data.
rawdata = pd.concat([list_signal, shuffleBackground])

X = rawdata.drop(["weight", "mjj", "Events"], axis=1)

# Normalized the data with a Gaussian distrubuition with 0 mean and unit variance.
X = sc.fit_transform(X)

# Signal

scalefactor = 0.00232 * 0.608791
sigw = rawdata["weight"][: len(list_signal.keys())] * scalefactor
bkgw = rawdata["weight"][len(list_signal.keys()) :]

# Labeling data with 1's and 0's to distinguish.(1/positve/signal and 0/negative/background)
# Truth Labels.
y = np.concatenate((np.ones(len(list_signal)), np.zeros(len(shuffleBackground))))

# Shuffle full data and split into train/test and validation set.
X_dev, X_eval, y_dev, y_eval = train_test_split(
    X, y, test_size=0.01, random_state=seed, stratify=y
)
X_train, X_test, y_train, y_test = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=seed, stratify=y_dev
)

# NN model defined as a function.
def build_model(network,RATE):

    # Create a NN model. Barebones model with no layers.
    model = Sequential()

    # Best option for most NN.
    opt = keras.optimizers.Nadam()

    # Activation function other options possible.
    act = "relu"  # Relu is 0 for negative values, linear for nonzero values.

    # Use model.add() to add one layer at a time, 1st layer needs input shape, So we pass the 1st element of network.
    # Dense Layers are fully connected and most common.

    model.add(Dense(network[0], input_shape=(numBranches,), activation=act))

    # Loop through and add layers (1,(n-2)) where n is the number of layers. We end at n-2 because we start at 1 not zero and
    # we  the input layer is added above with input dimension. Therefore we must remove 2 from layers.
    for i in range(1, len(network) - 1):
        model.add(Dense(network[i], activation=act))  # Hidden layers.
        # Turning off nuerons of layer above in loop with pr  obability = 1-r, so r = 0.25, then 75% of nerouns are kept.
        model.add(Dropout(RATE, seed=seed))

    # Last layer needs to have one neuron for a binary classification(BC) which yields from 0 to 1.
    model.add(
        Dense(network[len(network)-1], activation="sigmoid")
    )  # Output layer's activation function for BC needs to be sigmoid.

    # Last step is compiling.
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=tf.keras.metrics.Precision(),
    )
    return model
    
    def compare_train_test(kModel, X_train, y_train, X_test, y_test, bins=30):
    """
    This creates the signal and background distrubution.
    """
    passing_sig = []
    passing_bkg = []
    decisions = []
    for X, y in ((X_train, y_train), (X_test, y_test)):
        d1 = model.predict(X[y > 0.5]).ravel()  # signal
        d2 = model.predict(X[y < 0.5]).ravel()  # background
        decisions += [d1, d2]
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = array([low, high])
    
    #counting the index of the signal values with a score of greater than or equal to our optimal signal point
    i = 0
    for value in d1:
        if value > 0.15:
            passing_sig.append(i)
        i += 1
    #counting index of background values meeting the same parameter
    i = 0
    for value in d2:
        if value > 0.15:
            passing_bkg.append(i)
        i += 1
    
    plt.subplot(212)
    plt.hist(
        decisions[0],
        color="r",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="S (train)",
    )
    plt.hist(
        decisions[1],
        color="b",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="B (train)",
    )

    hist, bins = np.histogram(decisions[2], bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt="o", c="r", label="S (test)")

    hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.title("Net Learning Feedback")

    plt.errorbar(center, hist, yerr=err, fmt="o", c="b", label="B (test)")
    plt.legend(loc='upper center')
    
    chosen_sig = []
    sig_weight = []
    chosen_bkg = []
    bkg_weight = []
    
    #setting up a list for background and signal that passes output scores
    
    for index in passing_sig:
        chosen_sig.append(sig_mjj[index])
        sig_weight.append(s_w[index])
    for index in passing_bkg:
        chosen_bkg.append(bkg_mjj[index])
        bkg_weight.append(b_w[index])
        
    #histogram for mjj of passing signal/background events

    plt.figure()
    plt.hist(chosen_sig, bins=100, color='red', alpha=0.5, label='signal', weights=sig_weight)
    plt.hist(chosen_bkg, bins=100, color='purple', alpha=0.5, label='background', weights=bkg_weight)
    plt.title('Mjj Distribution')
    plt.xlabel('mjj')
    plt.ylabel('count')
    plt.yscale('log')
    plt.legend()
    
    def runNN(LAYER, BATCH, RATE):
    """
    NN structure ex. [5,5,5,5,1] 4 layers with 5 neurons each and one output layer. LAYER value is
    the number of hidden layers excluding the output layer. Each hidden layer will contain the same
    amount of neurons (It is hard coded to be the number of features). The BATCH is the batch size,
    powers of 2 are perfered but any positive number works. RATE is the drop out rate; so a RATE = .5
    is half of the neurons being randomly turned off.
    """
    network = []
    numEpochs = 120  # Number of times the NN gets trained.
    batchSize = BATCH
    numLayers = LAYER
    neurons = numBranches

    # This creates a list that has the stucture of the NN.
    for i in range(numLayers - 1):
        network.append(neurons)
    network.append(1)
    numNeurons = sum(network)

    # This is a conformation that the script is starting and the NN structure is displayed.
    print("Script starting....\n", network)

    # This tags the output files with either GPU or CPU.
    if status == 1:
        print("GPU")
        sufix = ".GPU"
    else:
        sufix = ".CPU"
        print("CPU")

    # Start time for file name.
    startTime = datetime.now()
    pre = time.strftime("%Y.%m.%d_") + time.strftime("%H.%M.%S.")

    # Filename for keras model to be saved as.
    h5name = (
        "numLayers"
        + str(LAYER)
        + ".numBr anches"
        + str(neurons)
        + ".batchSize"
        + str(BATCH)
    )
    
    #determines user path to generate a model
    
    path = os.getcwd()
    
    modelName = path + '/' + pre + h5name + sufix + ".h5"

    # Filename for plots to be identified by saved model.
    figname = path + '/' + pre + ".plots"

    # Using model and setting parameters.
    model = build_model(network,RATE)
    
    model.save(modelName)

    # This checkpoint is used for recovery of trained weights incase of interuption.
    checkPointsCallBack = ModelCheckpoint("temp.h5", save_best_only=True)

    # This terminates early if the monitor does not see an improvement after a certain
    # amount of epochs given by the patience.
    earlyStopCallBack = EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )

    # This is where the training starts.
    kModel = model.fit(
        X_train,
        y_train,
        epochs=numEpochs,
        batch_size=batchSize,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[earlyStopCallBack, checkPointsCallBack]
    )
    
    return model,kModel,startTime,modelName
    
    def storeModel(model,startTime,modelName,aucroc):        
    # computes max signif
    numbins = 100000
    allScore = model.predict(X)
    sigScore = model.predict(X[y > 0.5]).ravel()
    bkgScore = model.predict(X[y < 0.5]).ravel()
    sigSUM = len(sigScore)
    bkgSUM = len(bkgScore)
    xlimit = (0, 1)
    tp = []
    fp = []
    hist, bins = np.histogram(sigScore, bins=numbins, range=xlimit, density=False)
    count = 0
    for i in range(numbins - 1, -1, -1):
        count += hist[i] / sigSUM
        tp.append(count)
    hist, bins = np.histogram(bkgScore, bins=numbins, range=xlimit, density=False)
    count = 0
    for j in range(numbins - 1, -1, -1):
        count += hist[j] / bkgSUM
        fp.append(count)
    area = auc(fp, tp)
    xplot = tp
    yplot = fp
    # computes max signif
    sigSUM = len(sigScore) * scalefactor
    tp = np.array(tp) * sigSUM
    fp = np.array(fp) * bkgSUM
    syst = 0.0
    stat = 0.0
    maxsignif = 0.0
    maxs = 0
    maxb = 0
    bincounter = numbins - 1
    bincountatmaxsignif = 999
    for t, f in zip(tp, fp):
        signif = getZPoisson(t, f, stat, syst)
        if f >= 10 and signif > maxsignif:
            maxsignif = signif
            maxs = t
            maxb = f
            bincountatmaxsignif = bincounter
            score = bincountatmaxsignif / numbins
        bincounter -= 1
    print(
        "\n Score = %6.3f\n Signif = %5.2f\n nsig = %d\n nbkg = %d\n"
        % (score, maxsignif, maxs, maxb)
    )
    runtime = datetime.now() - startTime
    areaUnderCurve = "{:.4f}".format(aucroc)
    maxsignif = "{:5.2f}".format(maxsignif)
    # This is the predicted score. Values range between [0,1]
    y_predicted = model.predict(X_test)
    # The score is rounded; values are 0 or 1.  This isn't actually used?
    y_predicted_round = [1 * (x[0] >= 0.5) for x in y_predicted]
    average_precision = average_precision_score(y_test, y_predicted)
    avgPer = "{0:0.4f}".format(average_precision)
    score = "{0:6.3f}".format(score)
    maxs = "%10d" % (maxs)
    maxb = "%10d" % (maxb)
    cm = confusion_matrix(y_test, y_predicted_round)
    CM = [cm[0][0], cm[0][1]], [cm[1, 0], cm[1, 1]]
    modelParam = [
        "FileName",
        "ConfusionMatrix [TP FP] [FN TN]",
        "Run Time",
        "AUC",
        "Avg.P",
        "Score",
        "Max Signif",
        "nsig",
        "nbkg",
    ]
    df = pd.DataFrame(
        np.array(
            [
                [
                    modelName,
                    CM,
                    runtime,
                    areaUnderCurve,
                    avgPer,
                    score,
                    maxsignif,
                    nsig,
                    nbkg,
                ]
            ]
        ),
        columns=modelParam,
    )
    df.to_csv(modelName, mode="a", header=False, index=False)
    print(df.to_string(justify="left", columns=modelParam, header=True, index=False))
    print("Saving model.....")
    print("old auc: \n", aucroc, "\n new auc", areaUnderCurve)
    model.save(modelName)  # Save Model as a HDF5 filein Data folder
    print("Model Saved")
    
    def checkTraining(model,kModel):
    # This is the predicted score. Values range between [0,1]
    y_predicted = model.predict(X_test)

    # Prediction, fpr,tpr and threshold values for ROC.
    fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
    aucroc = auc(fpr, tpr)
    precision, recall, thresRecall = precision_recall_curve(y_test, y_predicted)

    plt.xlabel("Score")
    plt.ylabel("Distribution")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.subplot(211)
    plt.plot(fpr, tpr, "r-", label="ROC (area = %0.6f)" % (aucroc))
    plt.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # AUC
    
    # plot1 = plt.figure(1)
    # plotROC(fpr, tpr, aucroc)
    # plotPR(precision,recall,thresRecall)
    compare_train_test(kModel, X_train, y_train, X_test, y_test)
    if 0:
        # This plots the important features.
        plot2 = plt.figure(2)
        backgrounds = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        explainer = shap.DeepExplainer(model, backgrounds)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(
            shap_values,
            X_train,
            plot_type="bar",
            feature_names=branches[:-1],
            max_display=25,
            title='Overfitting Check',
            legend_labels=['Training', 'Testing']
       )
    return aucroc
    
    batch = 512

layers = 3

# This runs the training. A for loop can be used to vary the parameters. 
model,kModel,startTime,modelName=runNN(layers,batch,0.5)

aucroc=checkTraining(model,kModel)
print("Signal: ", nsig, "\n EWK: ", nEWKbkg, "\n QCD: ", nQCDbkg, "\n All Background: ", nbkg, "\n S/N: ", nsig/nbkg)
storeModel(model,startTime,modelName,aucroc)

val_loss = kModel.history['val_loss']
loss = kModel.history['loss']
epochs = []

i = 0
while i < len(val_loss):
    epochs.append(i)
    i += 1

plt.plot(epochs, loss, color='blue', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Test Loss')
plt.legend(loc='upper right')
plt.title("Loss Comparisons")
plt.xlabel('Epoch')
plt.ylabel('Loss')

precision = kModel.history['precision']
val_precision = kModel.history['val_precision']

plt.plot(epochs, precision, color='green', label='Training Accuracy')
plt.plot(epochs, val_precision, color='orange', label='Testing Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy measures')
plt.xlabel('Epoch')
plt.ylabel('Precision')

#Signal Data

s_MET = list_signal['MET']
s_mjj = list_signal['mjj']
s_j1PT = list_signal['j1PT']
s_j1Eta = list_signal['j1Eta']
s_j1Phi = list_signal['j1Phi']
s_j2PT = list_signal['j2PT']
s_j2Eta = list_signal['j2Eta']
s_j2Phi = list_signal['j2Phi']
s_METPhi = list_signal['METPhi']
s_weight = list_signal['weight']
s_j3PT = list_signal['j3PT']
s_j3Eta = list_signal['j3Eta']
s_j3Phi = list_signal['j3Phi']

#Summed Background Data

b_MET = shuffleBackground['MET']
b_mjj = shuffleBackground['mjj']
b_j1PT = shuffleBackground['j1PT']
b_j1Eta = shuffleBackground['j1Eta']
b_j1Phi = shuffleBackground['j1Phi']
b_j2PT = shuffleBackground['j2PT']
b_j2Eta = shuffleBackground['j2Eta']
b_j2Phi = shuffleBackground['j2Phi']
b_METPhi = shuffleBackground['METPhi']
b_weight = shuffleBackground['weight']
b_j3PT = shuffleBackground['j3PT']
b_j3Eta = shuffleBackground['j3Eta']
b_j3Phi = shuffleBackground['j3Phi']

#EWK Background Data

e_MET = list_ewk['MET']
e_mjj = list_ewk['mjj']
e_j1PT = list_ewk['j1PT']
e_j1Eta = list_ewk['j1Eta']
e_j1Phi = list_ewk['j1Phi']
e_j2PT = list_ewk['j2PT']
e_j2Eta = list_ewk['j2Eta']
e_j2Phi = list_ewk['j2Phi']
e_METPhi = list_ewk['METPhi']
e_weight = list_ewk['weight']
e_j3PT = list_ewk['j3PT']
e_j3Eta = list_ewk['j3Eta']
e_j3Phi = list_ewk['j3Phi']

#QCD Background Data

q_MET = list_qcd['MET']
q_mjj = list_qcd['mjj']
q_j1PT = list_qcd['j1PT']
q_j1Eta = list_qcd['j1Eta']
q_j1Phi = list_qcd['j1Phi']
q_j2PT = list_qcd['j2PT']
q_j2Eta = list_qcd['j2Eta']
q_j2Phi = list_qcd['j2Phi']
q_METPhi = list_qcd['METPhi']
q_weight = list_qcd['weight']
q_j3PT = list_qcd['j3PT']
q_j3Eta = list_qcd['j3Eta']
q_j3Phi = list_qcd['j3Phi']

plt.subplot(211)
plt.title('MET & Combined Backgrounds')
plt.hist(s_MET, bins=100, weights=s_weight, range=(200,3000), alpha=0.5, color='orange', label='Signal MET')
plt.hist(b_MET, bins=100, weights=b_weight, range=(200,3000), alpha=0.5, color='purple', label='Background MET')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('MET & Seperate Backgrounds')
plt.hist(e_MET, bins=100, weights=e_weight, range=(200,3000), alpha=0.5, color='red', label='EWK MET')
plt.hist(q_MET, bins=100, weights=q_weight, range=(200,3000), alpha=0.5, color='blue', label='QCD MET')
plt.hist(s_MET, bins=100, weights=s_weight, range=(200,3000), alpha=0.5, color='orange', label='Signal MET')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('mjj & Combined Backgrounds')
plt.hist(s_mjj, bins=100, weights=s_weight, range=(200,12000), alpha=0.5, color='orange', label='Signal mjj')
plt.hist(b_mjj, bins=100, weights=b_weight, range=(200,12000), alpha=0.5, color='purple', label='Background mjj')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('mjj & Seperate Backgrounds')
plt.hist(e_mjj, bins=100, weights=e_weight, range=(200,12000), alpha=0.5, color='red', label='EWK mjj')
plt.hist(q_mjj, bins=100, weights=q_weight, range=(200,12000), alpha=0.5, color='blue', label='QCD mjj')
plt.hist(s_mjj, bins=100, weights=s_weight, range=(200,12000), alpha=0.5, color='orange', label='Signal mjj')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('j1PT & Combined Backgrounds')
plt.hist(s_j1PT, bins=100, weights=s_weight, range=(200,2000), alpha=0.5, color='orange', label='Signal j1PT')
plt.hist(b_j1PT, bins=100, weights=b_weight, range=(200,2000), alpha=0.5, color='purple', label='Background j1PT')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('j1PT & Seperate Backgrounds')
plt.hist(e_j1PT, bins=100, weights=e_weight, range=(200,2000), alpha=0.5, color='red', label='EWK j1PT')
plt.hist(q_j1PT, bins=100, weights=q_weight, range=(200,2000), alpha=0.5, color='blue', label='QCD j1PT')
plt.hist(s_j1PT, bins=100, weights=s_weight, range=(200,2000), alpha=0.5, color='orange', label='Signal j1PT')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('j1Eta & Combined Backgrounds')
plt.hist(s_j1Eta, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j1Eta')
plt.hist(b_j1Eta, bins=100, weights=b_weight, range=(-5,5), alpha=0.5, color='purple', label='Background j1Eta')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('j1Eta & Seperate Backgrounds')
plt.hist(e_j1Eta, bins=100, weights=e_weight, range=(-5,5), alpha=0.5, color='red', label='EWK j1Eta')
plt.hist(q_j1Eta, bins=100, weights=q_weight, range=(-5,5), alpha=0.5, color='blue', label='QCD j1Eta')
plt.hist(s_j1Eta, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j1Eta')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('j1Phi & Combined Backgrounds')
plt.hist(s_j1Phi, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j1Phi')
plt.hist(b_j1Phi, bins=100, weights=b_weight, range=(-5,5), alpha=0.5, color='purple', label='Background j1Phi')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('j1Phi & Seperate Backgrounds')
plt.hist(e_j1Phi, bins=100, weights=e_weight, range=(-5,5), alpha=0.5, color='red', label='EWK j1Phi')
plt.hist(q_j1Phi, bins=100, weights=q_weight, range=(-5,5), alpha=0.5, color='blue', label='QCD j1Phi')
plt.hist(s_j1Phi, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j1Phi')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('j2PT & Combined Backgrounds')
plt.hist(s_j2PT, bins=100, weights=s_weight, range=(200,2000), alpha=0.5, color='orange', label='Signal j2PT')
plt.hist(b_j2PT, bins=100, weights=b_weight, range=(200,2000), alpha=0.5, color='purple', label='Background j2PT')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('j2PT & Seperate Backgrounds')
plt.hist(e_j2PT, bins=100, weights=e_weight, range=(200,2000), alpha=0.5, color='red', label='EWK j2PT')
plt.hist(q_j2PT, bins=100, weights=q_weight, range=(200,2000), alpha=0.5, color='blue', label='QCD j2PT')
plt.hist(s_j2PT, bins=100, weights=s_weight, range=(200,2000), alpha=0.5, color='orange', label='Signal j2PT')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('j2Eta & Combined Backgrounds')
plt.hist(s_j2Eta, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j2Eta')
plt.hist(b_j2Eta, bins=100, weights=b_weight, range=(-5,5), alpha=0.5, color='purple', label='Background j2Eta')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('j2Eta & Seperate Backgrounds')
plt.hist(e_j2Eta, bins=100, weights=e_weight, range=(-5,5), alpha=0.5, color='red', label='EWK j2Eta')
plt.hist(q_j2Eta, bins=100, weights=q_weight, range=(-5,5), alpha=0.5, color='blue', label='QCD j2Eta')
plt.hist(s_j2Eta, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j2Eta')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('j2Phi & Combined Backgrounds')
plt.hist(s_j2Phi, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j2Phi')
plt.hist(b_j2Phi, bins=100, weights=b_weight, range=(-5,5), alpha=0.5, color='purple', label='Background j2Phi')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('j2Phi & Seperate Backgrounds')
plt.hist(e_j2Phi, bins=100, weights=e_weight, range=(-5,5), alpha=0.5, color='red', label='EWK j2Phi')
plt.hist(q_j2Phi, bins=100, weights=q_weight, range=(-5,5), alpha=0.5, color='blue', label='QCD j2Phi')
plt.hist(s_j2Phi, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j2Phi')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('j3PT & Combined Backgrounds')
plt.hist(s_j3PT, bins=100, weights=s_weight, alpha=0.5, color='orange', label='Signal j3PT')
plt.hist(b_j3PT, bins=100, weights=b_weight, alpha=0.5, color='purple', label='Background j3PT')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('j3PT & Seperate Backgrounds')
plt.hist(e_j3PT, bins=100, weights=e_weight, alpha=0.5, color='red', label='EWK j3PT')
plt.hist(q_j3PT, bins=100, weights=q_weight, alpha=0.5, color='blue', label='QCD j3PT')
plt.hist(s_j3PT, bins=100, weights=s_weight, alpha=0.5, color='orange', label='Signal j3PT')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('j3Eta & Combined Backgrounds')
plt.hist(s_j3Eta, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j3Eta')
plt.hist(b_j3Eta, bins=100, weights=b_weight, range=(-5,5), alpha=0.5, color='purple', label='Background j3Eta')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('j3Eta & Seperate Backgrounds')
plt.hist(e_j3Eta, bins=100, weights=e_weight, range=(-5,5), alpha=0.5, color='red', label='EWK j3Eta')
plt.hist(q_j3Eta, bins=100, weights=q_weight, range=(-5,5), alpha=0.5, color='blue', label='QCD j3Eta')
plt.hist(s_j3Eta, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j3Eta')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('j3Phi & Combined Backgrounds')
plt.hist(s_j3Phi, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j3Phi')
plt.hist(b_j3Phi, bins=100, weights=b_weight, range=(-5,5), alpha=0.5, color='purple', label='Background j3Phi')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('j3Phi & Seperate Backgrounds')
plt.hist(e_j3Phi, bins=100, weights=e_weight, range=(-5,5), alpha=0.5, color='red', label='EWK j3Phi')
plt.hist(q_j3Phi, bins=100, weights=q_weight, range=(-5,5), alpha=0.5, color='blue', label='QCD j3Phi')
plt.hist(s_j3Phi, bins=100, weights=s_weight, range=(-5,5), alpha=0.5, color='orange', label='Signal j3Phi')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('METPhi & Combined Backgrounds')
plt.hist(s_METPhi, bins=100, weights=s_weight, alpha=0.5, color='orange', label='Signal METPhi')
plt.hist(b_METPhi, bins=100, weights=b_weight, alpha=0.5, color='purple', label='Background METPhi')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('METPhi & Seperate Backgrounds')
plt.hist(e_METPhi, bins=100, weights=e_weight, alpha=0.5, color='red', label='EWK METPhi')
plt.hist(q_METPhi, bins=100, weights=q_weight, alpha=0.5, color='blue', label='QCD METPhi')
plt.hist(s_METPhi, bins=100, weights=s_weight, alpha=0.5, color='orange', label='Signal METPhi')
plt.yscale('log')
plt.legend()
plt.show()
