import utils
import numpy as np
from saxpy.znorm import znorm
from saxpy.alphabet import cuts_for_asize
from saxpy.sax import sax_via_window
from saxpy.paa import paa
from saxpy.sax import sax_via_window
from saxpy.sax import ts_to_string
import matplotlib.pyplot as plt
from collections import Counter


def find_ngrams(input_list, n):
    grams = []
    for i in range(2, n + 1):
        grams.append(zip(*[input_list[j:] for j in range(i)]))
    return grams


def find_occurances(input_list):  #return number of occurances per letter sequence and for all possible n's
    occurances = []
    for i in range(len(input_list)):
        occurances.append(Counter(input_list[i]))
    return occurances


def find_probabilities(input_list): #returns probabilities
    probabilities = []
    for i in range(len(input_list)):
        som = sum(input_list[i].itervalues())
        dict = {}
        for key, value in input_list[i].iteritems():
            dict.update({key: (float(value) / float(som))})
        probabilities.append(dict)
    return probabilities


if __name__ == "__main__":
    df_trn1, df_trn2, df_test = utils.import_datasets()
    labels = df_test['ATT_FLAG']

    col = 'P_J422'
    n = 8 #this is de size of the window slices
    s = 3 #number of letters to use for SAX

    #normalize the data
    df_trn1[col] = znorm(df_trn1[col])
    df_test[col] = znorm(df_test[col])

    ######################################################
    #plot the data, first normal then with discrete data
    x = df_test['DATETIME']
    y = df_test[col]
    plt.plot(x, y)
    plt.show()
    #y = list(ts_to_string(df_trn2[col], cuts_for_asize(5)))
    df_test[col] = list(ts_to_string(df_test[col], cuts_for_asize(s)))
    y = [ord(x) for x in df_test[col]]
    x = df_test['DATETIME']
    plt.step(x, y)
    plt.show()



    columns = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU3', 'F_PU4', 'F_PU5', 'F_PU6','F_PU7','F_PU8', 'F_PU9','F_PU10', 'F_PU11', 'F_V2', 'S_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
    predictions = np.zeros((df_test.shape[0]))


    #looping through each sensor
    for col in columns:
        df_trn1[col] = list(ts_to_string(df_trn1[col], cuts_for_asize(s))) #SAX
        df_test[col] = list(ts_to_string(df_test[col], cuts_for_asize(s))) #SAX the test data

        ngramsTrain = find_ngrams(df_trn1[col] , n) #generate ngrams of window slice n

        occurances =  find_occurances(ngramsTrain) #calculating the occurances of each possible slice
        probas = find_probabilities(occurances) #calculating the probabilities of each slice in Train1

        ngramsTest = find_ngrams(df_test[col], n) #generate ngrams of the test data


        counter = n-1
        for r in ngramsTest[n-2]:
            if probas[n-2].get(r) < 0.000000001:
                predictions[counter] = 1
            counter += 1


    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(df_test.shape[0]):
        if (labels[i] == 1 and predictions[i] == 1):
            tp = tp + 1
        if (labels[i] == 0 and predictions[i] == 1):
            fp = fp + 1
        if (labels[i] == 0 and predictions[i] == 0):
            tn = tn + 1
        if (labels[i] == 1 and predictions[i] == 0):
            fn = fn + 1
    print 'tp: ', tp
    print 'fp: ', fp
    print 'fn: ', fn
    print 'tn: ', tn
    print 'precision:', 1.0 * tp / (tp + fp)
    print 'recall:', 1.0 * tp / (tp + fn)
