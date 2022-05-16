#!/usr/bin/env python
# coding: utf-8

# Imports and setup

# In[1]:


get_ipython().run_line_magic('config', 'InteractiveShell.ast_node_interactivity="last_expr_or_assign"')


# In[2]:


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

import uproot

# Checking if a GPU is available, not sure it will run in Jupyter
status = len(tf.config.experimental.list_physical_devices("GPU"))

# If we need a random seed.
seed = 42


# some useful functions

# In[3]:



def plotPR(x, y, t):
    #plt.subplot(411)
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


# This block defines the branch names that we'll pull from the tree. Luckily our current method of gathering data and weighting events does this for us, so this will remain commented out for the time being as it is redundant.

# In[4]:


def Event_Organization(Input_Trees):
    tree_branches = Input_Trees.arrays()
    MET = []
    METPhi = []
    j1PT = []
    j1Eta = []
    j1Phi = []
    j2PT = []
    j2Eta = []
    j2Phi = []
    j3PT = []
    j3Eta = []
    j3Phi = []
    weight = []
    mjj = []
    mjj_u   = []
    for element in tree_branches['MET']:
        MET.append(element)
    for element in tree_branches['METPhi']:
        METPhi.append(element)
    for element in tree_branches['j1PT']:
        j1PT.append(element)
    for element in tree_branches['j1Eta']:
        j1Eta.append(element)
    for element in tree_branches['j1Phi']:
        j1Phi.append(element)
    for element in tree_branches['j2PT']:
        j2PT.append(element)
    for element in tree_branches['j2Eta']:
        j2Eta.append(element)
    for element in tree_branches['j2Phi']:
        j2Phi.append(element)
    for element in tree_branches['j3PT']:
        j3PT.append(element)
    for element in tree_branches['j3Eta']:
        j3Eta.append(element)
    for element in tree_branches['j3Phi']:
        j3Phi.append(element)
    for element in tree_branches['weight']:
        weight.append(element)
    for element in tree_branches['mjjoptimized']:
        mjj.append(element)
    for element in tree_branches['mjj']:
        mjj_u.append(element)
            
    Output = {'MET':MET, 'METPhi':METPhi, 'j1PT':j1PT, 'j1Eta':j1Eta, 'j1Phi':j1Phi, 'j2PT':j2PT, 'j2Eta':j2Eta, 'j2Phi':j2Phi,
         'j3PT':j3PT, 'j3Eta':j3Eta, 'j3Phi':j3Phi, 'weight':weight, 'mjj':mjj, 'mjj_u':mjj_u}
        
    return(Output)


# This block reads in the data from the input files, using the branches specified above.  So we should update this section to use the right input files, etc.  Note there's also some scaling of the samples to get the weights right.

# In[5]:


main_line= '/home/jupyter-blonsbro/'
sig_line = 'Combined Signal Ntuples.root'
ewk_line = 'Combined EWKBackground Ntuples.root'
qcd_line = 'Combined QCDBackground Ntuples.root'

sig_dir = main_line+sig_line
ewk_dir = main_line+ewk_line
qcd_dir = main_line+qcd_line

main_sig = uproot.open(sig_dir)['Signal;1']
main_ewk = uproot.open(ewk_dir)['EWKBackground;1']
main_qcd = uproot.open(qcd_dir)['QCDBackground;1']

Signal = Event_Organization(main_sig)
EWK = Event_Organization(main_ewk)
QCD = Event_Organization(main_qcd)

print('Done!')


# In[6]:


#turning important data into dataframe to be easily read or transferred if need be

list_signal = pd.DataFrame(Signal)
list_signal = list_signal[(list_signal['MET'] > 200) & (list_signal['mjj'] > 1000) & (list_signal['mjj_u'] > 1000) & 
                          (list_signal['j3Eta'] > -500)]
list_ewk = pd.DataFrame(EWK)
list_qcd = pd.DataFrame(QCD)


#determining input neuron count

numBranches = len(list_signal.keys()) - 3

#getting counts for each input

nsig = len(list_signal['MET'])
# nEWKbkg = len(list_ewk['MET'])
# nQCDbkg = len(list_qcd['MET'])


df_background = list_ewk.append(list_qcd, ignore_index=True)

df_background = df_background[(df_background['MET'] > 200) & (df_background['mjj'] > 1000) & (df_background['mjj_u'] > 1000) & 
                          (df_background['j3Eta'] > -500)]

nbkg = len(df_background['MET'])


# The 3 backgrounds are concatenated we shuffle to make sure they are not sorted.
shuffleBackground = shuffle(df_background, random_state=seed)

#1/sum(weights)

scalefactor = sum(list_signal['weight'])/len(list_signal['weight'])

# Signal and shuffle background data.
rawdata = pd.concat([list_signal, shuffleBackground])

#Z is a non-guassian transformed, but equally ordered data set that provides as a clean copy of our original data

Z = rawdata

z = {-999.00:-9.0}
Z = Z.replace(z)

X = Z.drop(["mjj", "weight", "mjj_u"], axis=1)

# Normalized the data with a Gaussian distrubuition with 0 mean and unit variance.
X = sc.fit_transform(X)

# Labeling data with 1's and 0's to distinguish.(1/positve/signal and 0/negative/background)
# Truth Labels.
y = np.concatenate((np.ones(len(list_signal)), np.zeros(len(shuffleBackground))))

# Shuffle full data and split into train/test and validation set.
X_dev, X_eval, y_dev, y_eval, Z_dev, Z_eval = train_test_split(
    X, y, Z, test_size=0.01, random_state=seed, stratify=y
)
X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(
    X_dev, y_dev, Z_dev, test_size=0.2, random_state=seed, stratify=y_dev
)
#Changing all re-ordered data sets into dataframes
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
Z_df = Z_train.append(Z_test, ignore_index=True)


# In[7]:


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


# In[8]:


def compare_train_test(kModel, X_train_df, y_train, X_test_df, y_test, bins=100):
    """
    This creates the signal and background distrubution.
    """
    i = 0
    j = 0
    sig_index = []
    bkg_index = []
    decisions = []
    for X, y in ((X_train_df, y_train), (X_test_df, y_test)):
        # captures indices in X_train and X_test dataframes that correlate to signal and background events
        while i < len(y):
            if y[i] == 1.:
                sig_index.append(j)
            elif y[i] == 0.:
                bkg_index.append(j)
            i += 1
            j += 1
        i = 0
        d1 = model.predict(X[y > 0.5]).ravel()  # signal
        d2 = model.predict(X[y < 0.5]).ravel()  # background
        decisions += [d1, d2]
                
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = array([low, high])
    
    train_s = decisions[0]
    train_b = decisions[1]
    test_s = decisions[2]
    test_b = decisions[3]
    
    #Combining scores for test and training sets of signal and background seperately
    S_scores = np.concatenate((train_s, test_s), axis=None)
    B_scores = np.concatenate((train_b, test_b), axis=None)
    max_s = S_scores.max()
    
    
    #Scoring histograms for training sets
    
    figure2, axes = plt.subplots(3, figsize=(6,10))
    axes[0].hist(
        train_s,
        color="r",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="S (train)",
    )
    
    axes[0].hist(
        train_b,
        color="b",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="B (train)",
    )
    
    #Scoring points for testing sets

    histS, bins = np.histogram(test_s, bins=bins, range=low_high, density=True)
    scale = len(test_s) / sum(histS)
    err = np.sqrt(histS * scale) / scale

    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    axes[0].errorbar(center, histS, yerr=err, fmt="o", c="r", label="S (test)")
    axes[0].set_xlim(0,max_s)

    histB, bins = np.histogram(test_b, bins=bins, range=low_high, density=True)
    scale = len(test_b) / sum(histB)
    
    start = score_testing(train_s, train_b, histS, histB, low_high)
    
    err = np.sqrt(histB * scale) / scale
    axes[0].set_title("Net Learning Feedback")
    axes[0].errorbar(center, histB, yerr=err, fmt="o", c="b", label="B (test)")
    axes[0].vlines(start, 0, 30, colors='green', label='Score Threshold')
    axes[0].legend(loc='upper right')
    
    
    #These lists contain re-seperated, non-gaussian event information
    learned_sig_l = []
    learned_bkg_l = []
    
    for item in sig_index:
        learned_sig_l.append(Z_df.loc[item])
    
    for item in bkg_index:
        learned_bkg_l.append(Z_df.loc[item])
        
    #Creating new indices for following dataframe
    lsig_ind = list(range(len(learned_sig_l)))
    lbkg_ind = list(range(len(learned_bkg_l)))
    
    #changing lists into dataframes
    learned_sig = pd.DataFrame(learned_sig_l, index=lsig_ind)
    learned_bkg = pd.DataFrame(learned_bkg_l, index=lbkg_ind)
    
    #indices in scoring lists with scores greater than or equal too the dominant signal score
    passing_sig = []
    passing_bkg = []
        
    i = 0
    
    while i < len(S_scores):
        if S_scores[i] >= start: #The number here needs to be updated according to scoring graph output
            passing_sig.append(i)
        i += 1    
        
    i = 0
        
    while i < len(B_scores):
        if B_scores[i] >= start:
            passing_bkg.append(i)
        i += 1
    
    #lists of seperate signal and background event information with scores that were kept in the "passing" lists
    worthy_sig_l = []
    worthy_bkg_l = []
    
    for item in passing_sig:
        worthy_sig_l.append(learned_sig.loc[item])
        
    for item in passing_bkg:
        worthy_bkg_l.append(learned_bkg.loc[item])
        
    #Re-indexing dataframes
    
    wsig_ind = list(range(len(worthy_sig_l)))
    wbkg_ind = list(range(len(worthy_bkg_l)))
    
    #Dataframes for seperate signal and background events containing all physical feature information that also had scores
    #above or equal to the passing score

    worthy_sig = pd.DataFrame(worthy_sig_l, index=wsig_ind)
    worthy_bkg = pd.DataFrame(worthy_bkg_l, index=wbkg_ind)
    
    #Optimized Mjj graph
    
    print(len(worthy_bkg['mjj']), len(worthy_bkg['mjj_u']))
    
    axes[1].hist(worthy_bkg['mjj'], bins=100, weights=worthy_bkg['weight'], color='blue', alpha=0.5, label='Background Mjj')
    axes[1].hist(worthy_sig['mjj'], bins=100, weights=worthy_sig['weight'], color='red', alpha=0.5, label='Signal Mjj')
    axes[1].legend()
    axes[1].set_title('Mjj Distribution')
    axes[1].set_yscale('log')
    
    #Base Mjj Graph
    
    axes[2].hist(worthy_bkg['mjj_u'], bins=100, weights=worthy_bkg['weight'], color='blue', alpha=0.5, label="Background")
    axes[2].hist(worthy_sig['mjj_u'], bins=100, weights=worthy_sig['weight'], color='red', alpha=0.5, label="Signal")
    axes[2].legend()
    axes[2].set_title("Unordered Mjj Distribution")
    axes[2].set_xlabel('Mjj')
    axes[2].set_ylabel('Counts')
    axes[2].set_yscale('log')
    
    all_scores = np.append(B_scores, S_scores)
    all_mjj = np.append(learned_bkg['mjj'], learned_sig['mjj'])
    print(len(B_scores), len(S_scores), len(all_mjj), len(all_scores))
    figure2.tight_layout()
    
    Correlations = {'bkg':learned_bkg['mjj'], 'sig':learned_sig['mjj'], 'scoreb':B_scores, 'scores':S_scores, 
                    'combined':all_scores, 'mjj':all_mjj, 'strain':train_s, 'btrain':train_b, 'stest':test_s, 'btest':test_b
                   ,'sind':sig_index, 'bind':bkg_index}
    return(Correlations)

    


# In[9]:


def score_testing(s_scores, b_scores, s_te, b_te, h_range):
    threshold = 1
    step = 0.01
    
    s_tr, bins = np.histogram(s_scores, bins=100, range=h_range, density=True)
    b_tr, bins = np.histogram(b_scores, bins=100, range=h_range, density=True)
    
    score_ratios = []
    starting_vals = []
    
    
    while threshold > 0:
        focus_section = threshold - step
        focus_sig = []
        focus_bkg = []
        for score in s_te:
            if score >= focus_section:
                focus_sig.append(score)
        for score in b_te:
            if score >= focus_section:
                focus_bkg.append(score)
        sig_sum = sum(focus_sig)
        bkg_sum = sum(focus_bkg)
        score_ratios.append(sig_sum/np.sqrt(bkg_sum))
        starting_vals.append(threshold)
        threshold = focus_section
        
    i = 0
    while i < len(score_ratios):
        if score_ratios[i] == max(score_ratios):
            best_start = starting_vals[i]
            break
        i += 1
        
    print('Max: ', max(score_ratios))
    
    return(best_start)
    


# In[10]:


def runNN(LAYER, BATCH, RATE, xtrain, ytrain, xtest, ytest):
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
        xtrain,
        ytrain,
        epochs=numEpochs,
        batch_size=batchSize,
        validation_data=(xtest, ytest),
        verbose=1,
        callbacks=[earlyStopCallBack, checkPointsCallBack]
    )
    
    return model,kModel,startTime,modelName


# In[11]:


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
    sig_to_N = maxs/maxb
    print(
        "\n Score = %6.3f\n Signif = %5.2f\n nsig = %d\n nbkg = %d\n Scored S/N = %d\n"
        % (score, maxsignif, maxs, maxb, sig_to_N)
    )
    runtime = datetime.now() - startTime
    areaUnderCurve = "{:.4f}".format(aucroc)
    maxsignif = "{:5.2f}".format(maxsignif)
    # This is the predicted score. Values range between [0,1]
    y_predicted = model.predict(X_test_df)
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
        "Scored S/N",
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
                    sig_to_N,
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


# In[12]:


def checkTraining(model,kModel,nature):
    # This is the predicted score. Values range between [0,1]
    y_predicted = model.predict(X_test_df)

    # Prediction, fpr,tpr and threshold values for ROC.
    fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
    aucroc = auc(fpr, tpr)
    precision, recall, thresRecall = precision_recall_curve(y_test, y_predicted)
    
    figure, axs = plt.subplots(2, figsize=(6,10))
    axs[0].set_xlabel("Score")
    axs[0].set_ylabel("Distribution")
    axs[0].legend(loc="upper right")
    axs[0].plot(fpr, tpr, "r-", label="ROC (area = %0.6f)" % (aucroc))
    axs[0].plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck")
    axs[0].set_xlim([-0.05, 1.05])
    axs[0].set_ylim([-0.05, 1.05])
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].set_title("Receiver operating characteristic")
    axs[0].legend(loc="lower right")
    axs[0].grid(True)
    


    # AUC
    #plotROC(fpr, tpr, aucroc)

    axs[1].plot(thresRecall, precision[:-1], "b--", label="Precision")
    axs[1].plot(thresRecall, recall[:-1], "g-", label="Recall")
    axs[1].set_ylim([0.00, 1.05])
    axs[1].set_xlabel("Threshold")
    axs[1].set_title("Precision/Recall vs. Threshold Curve")
    axs[1].legend(loc="lower right")
    axs[1].grid(True)

    if nature == 'Real': 
        cor = compare_train_test(kModel, X_train_df, y_train, X_test_df, y_test)
    if nature == 'Test':
        cor = 'Dummy'
    
    '''
    #This plots the important features. Sadly it doesn't work
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
    '''
    return aucroc,cor


# In[13]:


batch = 512

layers = 3

# This runs the training. A for loop can be used to vary the parameters. 
model,kModel,startTime,modelName=runNN(layers,batch,0.5, X_train, y_train, X_test, y_test)


# Run statistics


aucroc,Correlations=checkTraining(model,kModel,'Real')
print("Signal: ", nsig, "\n All Background: ", nbkg, "\n S/N: ", nsig/nbkg)
storeModel(model,startTime,modelName,aucroc)


# ## Histogramming Proof

# In[17]:



#Signal Data

s_MET = list_signal['MET']
s_mjj_o = list_signal['mjj']
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
s_mjj = list_signal['mjj_u']

#Summed Background Data

b_MET = shuffleBackground['MET']
b_mjj_o = shuffleBackground['mjj']
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
b_mjj = shuffleBackground['mjj_u']

#EWK Background Data

e_MET = list_ewk['MET']
e_mjj_o = list_ewk['mjj']
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
e_mjj = list_ewk['mjj_u']

#QCD Background Data

q_MET = list_qcd['MET']
q_mjj_o = list_qcd['mjj']
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
q_mjj = list_qcd['mjj_u']


# In[18]:

#Input data review

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
plt.title('mjj ordered & Combined Backgrounds')
plt.hist(s_mjj_o, bins=100, weights=s_weight, range=(200,12000), alpha=0.5, color='orange', label='Signal mjj')
plt.hist(b_mjj_o, bins=100, weights=b_weight, range=(200,12000), alpha=0.5, color='purple', label='Background mjj')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('mjj ordered & Seperate Backgrounds')
plt.hist(e_mjj_o, bins=100, weights=e_weight, range=(200,12000), alpha=0.5, color='red', label='EWK mjj')
plt.hist(q_mjj_o, bins=100, weights=q_weight, range=(200,12000), alpha=0.5, color='blue', label='QCD mjj')
plt.hist(s_mjj_o, bins=100, weights=s_weight, range=(200,12000), alpha=0.5, color='orange', label='Signal mjj')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(211)
plt.title('mjj random & Combined Backgrounds')
plt.hist(s_mjj, bins=100, weights=s_weight, range=(200,12000), alpha=0.5, color='orange', label='Signal mjj')
plt.hist(b_mjj, bins=100, weights=b_weight, range=(200,12000), alpha=0.5, color='purple', label='Background mjj')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('mjj random & Seperate Backgrounds')
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
plt.hist(s_j3PT, bins=100, weights=s_weight, range=(200,3000), alpha=0.5, color='orange', label='Signal j3PT')
plt.hist(b_j3PT, bins=100, weights=b_weight, alpha=0.5, color='purple', label='Background j3PT')
plt.yscale('log')
plt.legend()
plt.show()

plt.subplot(212)
plt.title('j3PT & Seperate Backgrounds')
plt.hist(e_j3PT, bins=100, weights=e_weight, range=(200,2000), alpha=0.5, color='red', label='EWK j3PT')
plt.hist(q_j3PT, bins=100, weights=q_weight, range=(200,2000), alpha=0.5, color='blue', label='QCD j3PT')
plt.hist(s_j3PT, bins=100, weights=s_weight, range=(200,2000), alpha=0.5, color='orange', label='Signal j3PT')
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


# ## Correlation Review

# In[19]:


sig = Correlations['sig']
scores = Correlations['scores']

bkg = Correlations['bkg']
scoreb = Correlations['scoreb']

# mjj = Correlations['mjj']
# combined = Correlations['combined']

plt.hist2d(bkg, scoreb, bins=(100,100), norm=matplotlib.colors.LogNorm())
plt.title('Background')
plt.xlabel('Mjj')
plt.ylabel('Scores')
plt.colorbar()
plt.show()
plt.hist2d(sig, scores, bins=(100,100), norm=matplotlib.colors.LogNorm())
plt.title('Signal')
plt.xlabel('Mjj')
plt.ylabel('Scores')
plt.colorbar()
plt.show()
        
#File exporting for fitting


file_1 = uproot.recreate('/home/jupyter-zdethlof/mario-mapyde/NN_Feedback/Feedback.root')

data_list= np.concatenate((Correlations['strain'],Correlations['btrain'],Correlations['stest'],Correlations['btest']))
data_frame = pd.DataFrame(data_list)
file_1['data'] = data_frame

file_2 = uproot.recreate('/home/jupyter-zdethlof/mario-mapyde/NN_Feedback/Feedback_sig.root')
file_3 = uproot.recreate('/home/jupyter-zdethlof/mario-mapyde/NN_Feedback/Feedback_bkg.root')
data_splits = np.concatenate((Correlations['strain'], Correlations['stest']))
sig_frame = pd.DataFrame(data_splits)
data_splitb = np.concatenate((Correlations['btrain'], Correlations['btest']))
bkg_frame = pd.DataFrame(data_splitb)
file_2['signal'] = sig_frame
file_3['background'] = bkg_frame


#Below is simply the Check_training module, just cleaned up to only give the score feedback


def ctt_small(kModel, X_train_df, y_train, X_test_df, y_test, drop, bins):
    """
    This creates the signal and background distrubution.
    """
    i = 0
    j = 0
    sig_index = []
    bkg_index = []
    decisions = []
    for X, y in ((X_train_df, y_train), (X_test_df, y_test)):
        # captures indices in X_train and X_test dataframes that correlate to signal and background events
        while i < len(y):
            if y[i] == 1.:
                sig_index.append(j)
            elif y[i] == 0.:
                bkg_index.append(j)
            i += 1
            j += 1
        i = 0
        d1 = model.predict(X[y > 0.5]).ravel()  # signal
        d2 = model.predict(X[y < 0.5]).ravel()  # background
        decisions += [d1, d2]
                
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = array([low, high])
    
    train_s = decisions[0]
    train_b = decisions[1]
    test_s = decisions[2]
    test_b = decisions[3]
    
    #Combining scores for test and training sets of signal and background seperately
    S_scores = np.concatenate((train_s, test_s), axis=None)
    B_scores = np.concatenate((train_b, test_b), axis=None)
    max_s = S_scores.max()
    
    
    #Scoring histograms for training sets
    
    plt.hist(
        train_s,
        color="r",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="S (train)",
    )
    
    plt.hist(
        train_b,
        color="b",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="B (train)",
    )
    
    #Scoring points for testing sets

    histS, bins = np.histogram(test_s, bins=bins, range=low_high, density=True)
    scale = len(test_s) / sum(histS)
    err = np.sqrt(histS * scale) / scale

    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, histS, yerr=err, fmt="o", c="r", label="S (test)")
    plt.xlim(0,1)

    histB, bins = np.histogram(test_b, bins=bins, range=low_high, density=True)
    scale = len(test_b) / sum(histB)
    
    
    err = np.sqrt(histB * scale) / scale
    plt.title(('Net Learning Feedback + ' + drop))
    plt.errorbar(center, histB, yerr=err, fmt="o", c="b", label="B (test)")
    plt.legend(loc='upper right')
    plt.show()
    


# In[22]:

#Plotting the score for NN runs by adding a new feature each run of the Net

K = pd.DataFrame()
numBranches = 1
Areas = {"Base":aucroc}
for item in Z:
    if item != 'weight' and item != 'mjj' and item != 'mjj_u':
        K = pd.concat([K,Z[item]], axis=1)
        M = sc.fit_transform(K)
        X_dev, X_eval, y_dev, y_eval = train_test_split(
            K, y, test_size=0.01, random_state=seed, stratify=y
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_dev, y_dev, test_size=0.2, random_state=seed, stratify=y_dev
        )
        X_test_df = pd.DataFrame(X_test)
        model,kModel,startTime,modelName=runNN(layers,batch,0.5, X_train, y_train, X_test, y_test)
        print(K.keys())
        ctt_small(kModel, X_train, y_train, X_test_df, y_test, item, 100)
    
        numBranches += 1
    

#Testing Feedback regions surrounding the spike


t_srng = Correlations['scores'][(Correlations['scores']>=0.365)*(Correlations['scores']<=0.370)]
t_brng = Correlations['scoreb'][(Correlations['scoreb']>=0.365)*(Correlations['scoreb']<=0.370)]
l_srng = Correlations['scores'][(Correlations['scores']>=0.330)*(Correlations['scores']<=0.350)]
l_brng = Correlations['scoreb'][(Correlations['scoreb']>=0.330)*(Correlations['scoreb']<=0.350)]
h_srng = Correlations['scores'][(Correlations['scores']>=0.370)*(Correlations['scores']<=0.390)]
h_brng = Correlations['scoreb'][(Correlations['scoreb']>=0.370)*(Correlations['scoreb']<=0.390)]

figure3, plot = plt.subplots(3, figsize=(6,16))

plot[0].hist(t_srng, bins=30, density=True, alpha=0.5, color='r')
plot[0].hist(t_brng, bins=30, density=True, alpha=0.5, color='b')

plot[1].hist(l_srng, bins=30, density=True, alpha=0.5, color='r')
plot[1].hist(l_brng, bins=30, density=True, alpha=0.5, color='b')

plot[2].hist(h_srng, bins=30, density=True, alpha=0.5, color='r')
plot[2].hist(h_brng, bins=30, density=True, alpha=0.5, color='b')


# In[ ]:


print(len(t_srng) + len(t_brng), len(l_srng) + len(l_brng), len(h_srng) + len(h_brng))


# In[ ]:


ts_ind = []
us_ind = []
ds_ind = []

tb_ind = []
ub_ind = []
db_ind = []
i = 0
while i < len(scores):
    if scores[i]>= 0.390 and scores[i] <= 0.410:
        ts_ind.append(i)
    if scores[i]>= 0.410 and scores[i] <= 0.430:
        us_ind.append(i)
    if scores[i]>= 0.370 and scores[i] <= 0.390:
        ds_ind.append(i)
    i += 1
    
j = 0
while j < len(scoreb):
    if scoreb[j]>= 0.390 and scoreb[j] <= 0.410:
        tb_ind.append(j)
    if scoreb[j]>= 0.410 and scoreb[j] <= 0.430:
        ub_ind.append(j)
    if scoreb[j]>= 0.370 and scoreb[j] <= 0.390:
        db_ind.append(j)
    j += 1
    


re_ts = []
re_us = []
re_ls = []

re_tb = []
re_ub = []
re_lb = []

    
for val in ts_ind:
    real = Correlations['sind'][val]
    re_ts.append(real)
    
for val in us_ind:
    real = Correlations['sind'][val]
    re_us.append(real)
    
for val in ds_ind:
    real = Correlations['sind'][val]
    re_ls.append(real)
    
for val in tb_ind:
    real = Correlations['bind'][val]
    re_tb.append(real)
    
for val in ub_ind:
    real = Correlations['bind'][val]
    re_ub.append(real)
    
for val in db_ind:
    real = Correlations['bind'][val]
    re_lb.append(real)
    

    
tsig = pd.DataFrame(Z_df.iloc[re_ts])
usig = pd.DataFrame(Z_df.iloc[re_us])
lsig = pd.DataFrame(Z_df.iloc[re_ls])

tbkg = pd.DataFrame(Z_df.iloc[re_tb])
ubkg = pd.DataFrame(Z_df.iloc[re_ub])
lbkg = pd.DataFrame(Z_df.iloc[re_lb])


# In[ ]:


plt.subplot(311)
plt.hist(tsig['MET'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['MET'], color='blue', alpha=0.5, bins=100)
plt.xlim(0,3000)
plt.yscale('log')

plt.subplot(312)
plt.hist(usig['MET'], color='red', alpha=0.5, bins=100)
plt.hist(ubkg['MET'], color='blue', alpha=0.5, bins=100)
plt.xlim(0,3000)
plt.yscale('log')

plt.subplot(313)
plt.hist(lsig['MET'], color='red', alpha=0.5, bins=100)
plt.hist(lbkg['MET'], color='blue', alpha=0.5, bins=100)
plt.xlim(0,3000)
plt.yscale('log')


# In[ ]:


plt.subplot(311)
plt.hist(tsig['METPhi'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['METPhi'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')

plt.subplot(312)
plt.hist(usig['METPhi'], color='red', alpha=0.5, bins=100)
plt.hist(ubkg['METPhi'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')

plt.subplot(313)
plt.hist(lsig['METPhi'], color='red', alpha=0.5, bins=100)
plt.hist(lbkg['METPhi'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')


# In[ ]:


plt.subplot(311)
plt.hist(tsig['j1PT'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['j1PT'], color='blue', alpha=0.5, bins=100)
plt.xlim(0,2000)
plt.yscale('log')

plt.subplot(312)
plt.hist(usig['j1PT'], color='red', alpha=0.5, bins=100)
plt.hist(ubkg['j1PT'], color='blue', alpha=0.5, bins=100)
plt.xlim(0,2000)
plt.yscale('log')

plt.subplot(313)
plt.hist(lsig['j1PT'], color='red', alpha=0.5, bins=100)
plt.hist(lbkg['j1PT'], color='blue', alpha=0.5, bins=100)
plt.xlim(0,2000)
plt.yscale('log')


# In[ ]:


plt.subplot(311)
plt.hist(tsig['j1Eta'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['j1Eta'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')


plt.subplot(312)
plt.hist(usig['j1Eta'], color='red', alpha=0.5, bins=100)
plt.hist(ubkg['j1Eta'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')

plt.subplot(313)
plt.hist(lsig['j1Eta'], color='red', alpha=0.5, bins=100)
plt.hist(lbkg['j1Eta'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')


# In[ ]:


plt.subplot(311)
plt.hist(tsig['j1Phi'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['j1Phi'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')

plt.subplot(312)
plt.hist(usig['j1Phi'], color='red', alpha=0.5, bins=100)
plt.hist(ubkg['j1Phi'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')

plt.subplot(313)
plt.hist(lsig['j1Phi'], color='red', alpha=0.5, bins=100)
plt.hist(lbkg['j1Phi'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')


# In[ ]:


plt.subplot(311)
plt.hist(tsig['j2PT'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['j2PT'], color='blue', alpha=0.5, bins=100)
#plt.xlim(0,3000)
plt.yscale('log')


plt.subplot(312)
plt.hist(usig['j2PT'], color='red', alpha=0.5, bins=100)
plt.hist(ubkg['j2PT'], color='blue', alpha=0.5, bins=100)
#plt.xlim(0,3000)
plt.yscale('log')


plt.subplot(313)
plt.hist(lsig['j2PT'], color='red', alpha=0.5, bins=100)
plt.hist(lbkg['j2PT'], color='blue', alpha=0.5, bins=100)
#plt.xlim(0,3000)
plt.yscale('log')


# In[ ]:


plt.subplot(311)
plt.hist(tsig['j3PT'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['j3PT'], color='blue', alpha=0.5, bins=100)
#plt.xlim(0,3000)
plt.yscale('log')


plt.subplot(312)
plt.hist(usig['j3PT'], color='red', alpha=0.5, bins=100)
plt.hist(ubkg['j3PT'], color='blue', alpha=0.5, bins=100)
#plt.xlim(0,3000)
plt.yscale('log')


plt.subplot(313)
plt.hist(lsig['j3PT'], color='red', alpha=0.5, bins=100)
plt.hist(lbkg['j3PT'], color='blue', alpha=0.5, bins=100)
#plt.xlim(0,3000)
plt.yscale('log')


# In[ ]:


tj2Eb = tbkg['j2Eta'][tbkg['j2Eta'] > -5]
tj2Es = tsig['j2Eta'][tsig['j2Eta'] > -5]

plt.subplot(311)
plt.hist(tj2Eb, color='blue', alpha=0.5, bins=100, label='bkg')
plt.hist(tj2Es, color='red', alpha=0.5, bins=100, label='sig')
#plt.xlim(-1000,-900)
plt.yscale('log')
plt.title('j2Eta: Target Range')
#plt.legend()

uj2Eb = ubkg['j2Eta'][ubkg['j2Eta'] > -5]
uj2Es = usig['j2Eta'][usig['j2Eta'] > -5]

plt.subplot(312)
plt.hist(uj2Eb, color='blue', alpha=0.5, bins=100, label='bkg')
plt.hist(uj2Es, color='red', alpha=0.5, bins=100, label='sig')
#plt.xlim(-6,6)
plt.yscale('log')
plt.title('j2Eta: Upper Range')
#.legend()

lj2Eb = lbkg['j2Eta'][lbkg['j2Eta'] > -5]
lj2Es = lsig['j2Eta'][lsig['j2Eta'] > -5]

plt.subplot(313)
plt.hist(lj2Eb, color='blue', alpha=0.5, bins=100, label='bkg')
plt.hist(lj2Es, color='red', alpha=0.5, bins=100, label='sig')
#plt.xlim(-6,6)
plt.yscale('log')
plt.title('j2Eta: Lower Range')
plt.subplots_adjust(hspace=0.8)
#plt.legend()


# In[ ]:


tj2Ps = tsig['j2Phi'][tsig['j2Phi'] > -5]
tj2Pb = tbkg['j2Phi'][tbkg['j2Phi'] > -5]

plt.subplot(311)
plt.hist(tj2Ps, color='red', alpha=0.5, bins=100)
plt.hist(tj2Pb, color='blue', alpha=0.5, bins=100)
plt.yscale('log')

uj2Ps = usig['j2Phi'][usig['j2Phi'] > -5]
uj2Pb = ubkg['j2Phi'][ubkg['j2Phi'] > -5]

plt.subplot(312)
plt.hist(uj2Ps, color='red', alpha=0.5, bins=100)
plt.hist(uj2Pb, color='blue', alpha=0.5, bins=100)
plt.yscale('log')

lj2Ps = lsig['j2Phi'][lsig['j2Phi'] > -5]
lj2Pb = lbkg['j2Phi'][lbkg['j2Phi'] > -5]


plt.subplot(313)
plt.hist(lj2Ps, color='red', alpha=0.5, bins=100)
plt.hist(lj2Pb, color='blue', alpha=0.5, bins=100)
plt.yscale('log')


# In[ ]:


plt.subplot(311)
plt.hist(tsig['j3Eta'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['j3Eta'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')


plt.subplot(312)
plt.hist(usig['j3Eta'], color='red', alpha=0.5, bins=100)
plt.hist(ubkg['j3Eta'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')


plt.subplot(313)
plt.hist(lsig['j3Eta'], color='red', alpha=0.5, bins=100)
plt.hist(lbkg['j3Eta'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')


# In[ ]:


plt.subplot(311)
plt.hist(tsig['j3Phi'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['j3Phi'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')

plt.subplot(312)
plt.hist(usig['j3Phi'], color='red', alpha=0.5, bins=100)
plt.hist(ubkg['j3Phi'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')

plt.subplot(313)
plt.hist(lsig['j3Phi'], color='red', alpha=0.5, bins=100)
plt.hist(lbkg['j3Phi'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')


# In[ ]:


plt.subplot(311)
plt.hist(tsig['mjj'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['mjj'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')
plt.xlim(0,12000)

plt.subplot(312)
plt.hist(tsig['mjj'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['mjj'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')
plt.xlim(0,12000)

plt.subplot(313)
plt.hist(tsig['mjj'], color='red', alpha=0.5, bins=100)
plt.hist(tbkg['mjj'], color='blue', alpha=0.5, bins=100)
plt.yscale('log')
plt.xlim(0,12000)
