########################################################################
########################################################################


# General script containing all functions and variables for the thesis
# project : Identification of Diagnoses through Electrocardiograms 
# using Convolutional Neural Networks.

# Author : Miguel Calvo Valente


########################################################################
########################################################################

##### Modules

# Data Loading and Manipulation
import numpy as np
import pandas as pd
import wfdb
import ast
import scipy.signal
from scipy.fft import fft, ifft
from loess.loess_1d import loess_1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import median_filter

# from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal
# from loess.loess_1d import loess_1d
# from statsmodels.nonparametric.smoothers_lowess import lowess

# Data Visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Modeling
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Tensorflow
import tensorflow as tf
import tensorflow_addons as tfa

# Others
import os
import random
import winsound
import itertools


########################################################################
########################################################################

##### Variables

# This is the order of the 12 signals for each file.
signals = ['I', 'II', 'III', 'AVL', 'AVR', 'AVF', 'V1', 'V2','V3','V4','V5', 'V6']
signals_dict = {'I':0, 'II':1, 'III':2, 'AVL':3, 'AVR':4, 'AVF':5, 
                'V1':6, 'V2':7,'V3':8,'V4':9,'V5':10, 'V6':11}

# Path to images folder
imgs_path = 'imgs/'

# General random seed for replicability
random_state = 203129

# N of examples when plotting for each dx
n_sample = 5

# Fixed set of colors to identify the urgency level
urgency_colors = {1: '#EC0D0D', 2:'#EC7D0D', 3:'#E9EC0D', 4:'#0DEC84'}

# N of examples to plot the confusion matrix
n_ex = 500

# Thresholds to have more recall on specific classes
mi_threshold = 0.2
ms_threshold = 0.3
thresholds = {'PTB-XL':               
              {'MI' : 
                {'idx' : 0, 
                 'threshold' : mi_threshold},               
               'MI&STTC' : 
                {'idx' : 2,
                 'threshold' : ms_threshold}
              }
             }

# N of epochs for training models
epochs = 50

# Batch size for training models
batch_size = 64

# Loss function dictionary
loss_function_dict = {
    'mi' : 'binary_crossentropy',
}

# Metrics to track dictionary
metrics_to_track_dict = {
    'mi' : 'accuracy',
}

# Dictionary of neural network elements
element_dict = {
    'ks' : 'Adequate Kernel and Stride',
    'bn' : 'Batch Normalization',
    'l1' : 'L1 Regularization',
    'dr' : 'Dropout',
    'as' : 'Age & Sex',
    'lr' : 'LR Reduction',
    'cw' : 'Class Weights',    
}

# Dictionary of preprocessing steps
preproc_dict = {
    'og'     : 'Original',
    'mf_200' : 'Median Filter (200)',
}

# Dictionary of outlier removal
outlier_dict = {
    'ol_5_25' : 'Remove Outliers (5, 25)',
}

########################################################################
########################################################################

##### Data Loading and Manipulation

def load_labels(path):

    """
    Load the SNOMED CT labels from the path, along with the urgency of the treatment
    according to the criteria supplied by Instituto Nacional de Cardiología Ignacio Chávez.
    
    Inputs:
        path - str; contains relative path to the excel files.
    """
    
    # Load labels given the paths
    most_freq_labels = pd.read_excel(path+'Most_Frequent_Labels-DAG.xlsx', header=6)
    less_freq_labels = pd.read_excel(path+'Less_Frequent_Labels-DAG.xlsx', header=6)

    # The last one is just the sum per columns
    most_freq_labels = most_freq_labels[:-1]
    less_freq_labels = less_freq_labels[:-1]

    # Convert the SNOMED CT Code to integer
    most_freq_labels['SNOMED CT Code'] = most_freq_labels['SNOMED CT Code'].apply(lambda x : int(x))
    less_freq_labels['SNOMED CT Code'] = less_freq_labels['SNOMED CT Code'].apply(lambda x : int(x))

    # Rename the column for the Urgency of the treatment
    most_freq_labels = most_freq_labels.rename(columns={'Here:':'Urgency'})
    less_freq_labels = less_freq_labels.rename(columns={'Here:':'Urgency'})

    # Modify the "empty" case from a "." to a 4
    most_freq_labels['Urgency'] = most_freq_labels['Urgency'].apply(lambda x : 4 if x == '.' else x)
    less_freq_labels['Urgency'] = less_freq_labels['Urgency'].apply(lambda x : 4 if x == '.' else x)
    
    # Add an identifer to know whether it belongs to the "most" or "less" frquent kind
    most_freq_labels['Kind'] = 'most'
    less_freq_labels['Kind'] = 'less'
    
    # Add everything into one DF and set the index to be the SNOMED CT Code
    labels = pd.concat([most_freq_labels, less_freq_labels])
    labels.index =  labels['SNOMED CT Code']
    labels.drop(['SNOMED CT Code'], axis=1, inplace=True)

    return labels


def load_labels_train(path, labels_id):

    """
    Retrieves the labels to train as the model expects them.

    This function will eventually consider many cases, but for 
    now we just consider the one we are working with, which is
    MI vs OTHER.
    """

    if labels_id == 'mi':

        Y_superclasses = pd.read_csv(path + 'Y_superclasses.csv', index_col=0)
        Y_mi = Y_superclasses[['MI']]

        return Y_mi    

def initial_data_preproc(X, Y, X_dtype=np.float16):

    """
    The data we load using the wfdb.rdsamp function still needs some 
    preprocessing, particularly regarding the metadata. THis function
    takes this initial data and transforms the signals to a numpy array, 
    and the metadata to a pandas DataFrame.
    
    Inputs:
        ...
    """

    # Convert X to a np.array
    X = np.array(X, dtype=X_dtype)

    # We create a DF to better portray the metadata information
    Y_df = pd.DataFrame(columns=['age', 'sex', 'dx', 'rx', 'hx', 'sx', 'fs', 'sig_len', 'n_sig', 'units', 'sig_name'])

    # Access each component of the metadata and make an auxiliary DF
    for y in Y:

        y_aux = pd.DataFrame({'age':       [y['comments'][0][5:]],
                              'sex':       [y['comments'][1][5:]],
                              'dx':        [y['comments'][2][4:]], 
                              'rx':        [y['comments'][3][4:]], 
                              'hx':        [y['comments'][4][4:]], 
                              'sx':        [y['comments'][5][4:]],
                              'fs':        [y['fs']], 
                              'sig_len':   [y['sig_len']], 
                              'n_sig':     [y['n_sig']], 
                              'units':     [y['units']], 
                              'sig_name':  [y['sig_name']]})
        
        Y_df = pd.concat([Y_df, y_aux])
    
    Y = Y_df
    Y.reset_index(drop = True, inplace = True)

    return X, Y


def load_data(files, X_dtype=np.float16):

    """
    Read the .mat files and return the signals and the metadata.
    
    Inputs:
        files : list; contains the names of each file that will be loaded        

    Note: 
        !!!
        This function should be later altered such that it accounts to
        the differences in mV, observations per second and total duration.
        !!!

    """

    # Initialize empty containers
    X = []
    Y = pd.DataFrame(columns=['age', 'sex', 'dx', 'rx', 'hx', 'sx',
                              'fs', 'sig_len', 'n_sig', 'units', 'sig_name'])

    # Given the wfdb.rdsamp tuple, pass its values to X and Y, respectively
    for file in files:
        f = wfdb.rdsamp(file)
        # Process signals
        x = f[0]
        x = np.array(x, dtype=X_dtype)
        X.append(x)
        # Process metadata
        y = f[1]
        y = pd.DataFrame({'age':       [y['comments'][0][5:]],
                          'sex':       [y['comments'][1][5:]],
                          'dx':        [y['comments'][2][4:]], 
                          'rx':        [y['comments'][3][4:]], 
                          'hx':        [y['comments'][4][4:]], 
                          'sx':        [y['comments'][5][4:]],
                          'fs':        [y['fs']], 
                          'sig_len':   [y['sig_len']], 
                          'n_sig':     [y['n_sig']], 
                          'units':     [y['units']], 
                          'sig_name':  [y['sig_name']]})
        Y = pd.concat([Y, y])

    # Return the containers with the data
    X = np.array(X, dtype=X_dtype)
    Y.reset_index(drop = True, inplace = True)
    return X, Y


def load_X_and_M(path, X_name, M_name):

    """
    The load_data function works in general for any given files.
    This was necessary to load the whole dataset. However, if data is already
    stored in a more efficient format, then we can load it more quickly
    with this function.
    """

    X = np.load(path + X_name)
    M = pd.read_csv(path + M_name, index_col=0)
    M['age'] = M['age'].apply(lambda x : 'NaN' if np.isnan(x) else x)

    return X, M


def y_superclasses(path_to_load='data/ptbxl_database.csv', path_to_store='data/PTB-XL/Y_superclasses.csv',
                   scp_statements='data/scp_statements.csv'):

    """
    We load the superclasses labels for each register. This is only available for the
    PTB-XL database, since they are the ones that have a separate way to download the data,
    where they classify their data through superclasses, classes and subclasses.
    """

    # Load and convert annotation data
    Y_ptbxl = pd.read_csv(path_to_load, index_col='ecg_id')
    Y_ptbxl.scp_codes = Y_ptbxl.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(scp_statements, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y_ptbxl['diagnostic_superclass'] = Y_ptbxl.scp_codes.apply(aggregate_diagnostic)

    superclasses = set(agg_df.diagnostic_class)

    ## Now, we reformat the results since we want to have a matrix that 
    ## identifies if a register belongs to a certain superclass (note that 
    ## a register can belong to more than 1). This can take some time. 
    ## For this reason we rather save it as a .csv file for later use.

    superclasses_df = pd.DataFrame(columns = superclasses)

    for i in range(len(Y_ptbxl.diagnostic_superclass)):
        
        class_aux = pd.DataFrame(index = [0], columns = superclasses)
        
        for sc in Y_ptbxl.diagnostic_superclass.iloc[i]:
            class_aux.loc[0, sc] = 1
                        
        superclasses_df = pd.concat([superclasses_df, class_aux])
        
    superclasses_df = superclasses_df.fillna(0)
    superclasses_df = superclasses_df.reset_index().drop('index', 1)

    superclasses_df.to_csv(path_to_store)


def y_mi_sttc(Y_superclasses, with_label=False):

    # Copy and keep only the classes of interest
    strat_df = Y_superclasses.copy()        
    strat_df = strat_df[['MI', 'STTC']]

    # Intersection between MI & STTC
    strat_df['MI&STTC'] = (strat_df['MI'] == 1) & (strat_df['STTC'] == 1)
    strat_df['MI&STTC'] = strat_df['MI&STTC'].astype(int)

    # Correct MI and STTC columns given an intersection
    strat_df['MI'] = strat_df['MI'] - strat_df['MI&STTC']
    strat_df['STTC'] = strat_df['STTC'] - strat_df['MI&STTC']

    # Create the OTHER column as the complement
    strat_df['OTHER'] = (strat_df['MI'] == 0) & (strat_df['STTC'] == 0) & (strat_df['MI&STTC'] == 0)
    strat_df['OTHER'] = strat_df['OTHER'].astype(int)

    # Add a label column 
    if with_label:
        strat_df['label'] = strat_df.idxmax(axis=1)
    
    return strat_df


def y_labelling(dx, labels):

    """
    Compute the 0-1 label matrix for a given set of diagnosis. That is we
    create a matrix in which the rows are the registers, the columns are the
    different SNOMED CT Codes, and the values are 1 if said register was diagnosed
    with a code, and 0 otherwise. 

    Note : This function can take a long time to give results. In the future, it is
    advised to rather create a new csv file that has it already computed.

    Inputs:
        dx - pd.Series, contains the diagnosis separated by a ',' for each row
        labels - list, contains all of the available (or desired) labels from
            the SNOMED CT Codes.
    """
    # Split all values of dx and turn them into integers
    Y_aux = dx.apply(lambda x : x.split(','))
    Y_aux = Y_aux.apply(lambda x : [int(v) for v in x])

    # Initialize OHE DataFrame
    Y_ohe = pd.DataFrame(columns = labels)

    # For each column and record, write 1 if there is an intersection,
    # 0 otherwise
    for k in range(len(Y_aux)):
        
        new_row = pd.DataFrame(index = [k], columns = labels)
        
        for c in labels:
            
            new_row.loc[k, c] = int(c in Y_aux[k])
            
        Y_ohe = pd.concat([Y_ohe, new_row])

    Y_ohe = Y_ohe.reset_index().drop('index', 1)

    return Y_ohe


def ud_median_filter(signal, rolling_window=200, correct_first=True):

    """
    Compute the mobile median of a signal and substract it.
    """

    # Compute the rolling median on a given window size
    rolling_median = signal.rolling(rolling_window).median()
    rolling_median = rolling_median.fillna(0)
    # The corrected values are the original minus the median
    corrected_values = pd.DataFrame(signal - rolling_median)
    
    # This approach will leave the first "n = rolling_window" values
    # unchanged. To also correct those first values, we reverse the series
    # and calculate the rolling median for those
    if correct_first:
          
        reversed_row = signal.iloc[::-1].reset_index().iloc[:, 1].copy()
        reversed_rolling_median = reversed_row.rolling(rolling_window).median()
        reversed_rolling_median = reversed_rolling_median.fillna(0)
        reversed_row = reversed_row - reversed_rolling_median
        reversed_row = reversed_row.iloc[::-1].reset_index().iloc[:, 1:]
        reversed_row = reversed_row.iloc[:rolling_window, :]
        # We join the now corrected first values with the others prevously
        # calculated
        corrected_values = corrected_values.iloc[rolling_window:, :] 
        corrected_values = pd.concat([reversed_row, corrected_values])
    
    return corrected_values


def ud_median_filter_ecg(ECG, rolling_window=200, correct_first=True):
    """
    Apply the median_filter function to a whole ECG.
    """

    x = pd.DataFrame()
    aux = pd.DataFrame(ECG)
    for row in range(12):        
        corrected_values = ud_median_filter(aux[row], rolling_window, correct_first)        
        x = pd.concat([x, corrected_values], 1)

    x = np.array(x)
    return x


def sp_median_filter(signal, size=200, mode='reflect'):

    """
    Compute the mobile median of a signal and substract it, using
    scipy's median_filter functionn.
    """

    original_series = signal.copy()
    original_series = np.array(original_series)
    # Arranging the signal
    original_series = original_series.reshape(-1, 1)
    original_series = original_series.astype(float)
    # Calculate the median filter
    median_series = median_filter(original_series, size=size, mode=mode)
    median_series = median_series[:, 0]
    # Return the centered series
    return original_series[:, 0] - median_series
    

def sp_median_filter_ecg(ECG, size=200, mode='reflect'):
    """
    Apply the scipy median_filter function to a whole ECG.
    """

    x = pd.DataFrame()
    aux = pd.DataFrame(ECG)
    for row in range(12):        
        corrected_values = sp_median_filter(aux[row], size=size, mode=mode)    
        corrected_values = pd.DataFrame(corrected_values, columns=[row+1])
        x = pd.concat([x, corrected_values], 1)

    x = np.array(x)
    return x


def tf_median_filter(signal, filter_shape=200, padding='REFLECT', return_np=False, return_mf=False):

    """
    Compute the mobile median of a signal and substract it, using
    tensorflow's median_filter2d function.
    """

    original_series = signal.copy()
    original_series = np.array(original_series)
    # Arranging the signal
    original_series = original_series.reshape(-1, 1)
    original_series = original_series.astype(float)
    # Convert to tensor
    original_series = tf.convert_to_tensor(original_series)
    # Calculate the median filter
    median_series = tfa.image.median_filter2d(original_series, filter_shape=(filter_shape, 1),
                                               padding=padding)
    median_series = median_series[:, 0]
    if return_mf:
        return original_series[:, 0] - median_series, median_series
    # Return the centered series
    return original_series[:, 0] - median_series
    

def tf_median_filter_ecg(ECG, filter_shape=200, padding='REFLECT', return_np=False):
    """
    Apply the tensorflow median_filter2d function to a whole ECG.
    """

    original_series = ECG.copy()
    # Calculate the median filter
    median_series = tfa.image.median_filter2d(original_series, filter_shape=(filter_shape, 1),
                                              padding=padding)    
    # # Return the centered series
    corrected_series = original_series - median_series
    if return_np:
        corrected_series = corrected_series.numpy()

    return corrected_series


def tf_median_filter_ecgs(ECGs, filter_shape=200, padding='REFLECT', return_np=False):
    """
    Apply the tensorflow median_filter2d function to many ECGs.
    """

    original_series = ECGs.copy()
    # This is the median function that will be applied to each ecg
    f = lambda x: tfa.image.median_filter2d(x, 
                                            filter_shape=(filter_shape, 1),
                                            padding=padding) 
    # We apply it to the ECGs
    median_series = tf.map_fn(f, original_series)
    # And correct the series
    corrected_series = original_series - median_series
    if return_np:
        corrected_series = corrected_series.numpy()

    return corrected_series


def calculate_standard_deviations(x, batch_size=50, repeated=False, standarized=True, variance=False):

    """
    Calculate the standard deviations of a signal every batch 
    of size batch_size.
    """
    # Get number of batches
    n_batches = x.shape[0] // batch_size
    # Calculate the index to group by with
    ids = list(range(batch_size)) * n_batches
    ids.sort()
    # Create a df with the index and the signal
    aux = pd.DataFrame([ids, x]).transpose()
    # Group by index and calculate standard deviation
    if variance:
        aux = aux.groupby(0).var()
    else:
        aux = aux.groupby(0).std(ddof=1)
    # Standarize if desired
    if standarized:
        aux = aux / x.std()
    # Repeat stdevs if desired
    if repeated:
        aux = aux.loc[aux.index.repeat(n_batches)]
        
    stdevs = aux[1].values
    return stdevs


def rolling_variance(x, window_size=50, homologate_qrs=None):

    """
    Calculate the rolling variance for the given window_size
    of size batch_size.
    """
    series = pd.DataFrame(x)
    if homologate_qrs is not None:
        xqrs_radius = [[range(qrs - homologate_qrs['fs'] // 50,
                              qrs + homologate_qrs['fs'] // 50)]
                       for qrs in homologate_qrs['qrs']]
        xqrs_radius = list(np.concatenate(xqrs_radius).flat)
        series.loc[xqrs_radius] = series.loc[xqrs_radius].mean()[0]
    series = series.rolling(window_size).var()
    series = series.fillna(0)
    return series


def qrs_rr_and_hr(x, fs, omit_first_rr=False):

    """
    Calculate QRS segments, RR intervals and Heart Rate of a signal.
    """
    xqrs = wfdb.processing.xqrs_detect(x, fs, learn=False, verbose=False)
    if omit_first_rr:
        xqrs = xqrs[1:]
    rr = wfdb.processing.calc_rr(xqrs, fs)
    hr = wfdb.processing.calc_mean_hr(rr, fs)
    return xqrs, rr, hr


def hr_out_of_desv(x, desv=1):        

    """
    Determine if there are registers that have heartbeats beyond a 
    threshold equal to desv.
    """
    median = np.median(x)
    out_of_desv = [(v < median - desv) | (v > median + desv) for v in x]
    n_out_of_desv = sum(out_of_desv)
    return n_out_of_desv


def calculate_median_peaks(x, qr, s, return_points=False):

    """
    Given the QRS complexes, find the maximum and minimum values
    around them, then calculate their medians.
    """
    # We will store the values around each QRS here
    around_qr = dict()
    
    # This generates non-overlapping chunks of the ECG, each centered
    # around a QRS complex
    try:
        for i in range(len(qr[s])):
            if i==0:
                # The first one is a special case since we ommited the first
                # QRS when we calculated them, so we use an interval around it
                # using the distance to the next one
                lower_middle_point = int((qr[s][i]+qr[s][i+1])/2 - qr[s][i])
                upper_middle_point = int((qr[s][i]+qr[s][i+1])/2)            
            elif i==len(qr[s])-1:
                # The last one is also a special case, since we now want to
                # close the chunk at the furthest point to the right
                lower_middle_point = int((qr[s][i]+qr[s][i-1])/2)
                upper_middle_point = len(x)
            else:
                # For all the other ones, we simply use the middle points between
                # QRS complexes as lower and upper bounds
                lower_middle_point = int((qr[s][i]+qr[s][i-1])/2)
                upper_middle_point = int((qr[s][i]+qr[s][i+1])/2)
        
            # Generate the chunk with the bounds
            around_qr[qr[s][i]] = np.array(range(lower_middle_point, upper_middle_point))

        # We prevent index errors for the first chunk, since it could have negative indexes
        # if the lower_middle_point was a negative value.
        filtered = [value > 0 for value in around_qr[qr[s][0]]]
        around_qr[qr[s][0]] = around_qr[qr[s][0]][filtered]

        # For each chunk, find the max and min values
        max_values = dict()
        min_values = dict()
        for i in range(len(qr[s])):
            values = x[around_qr[qr[s][i]]]
            max_values[qr[s][i]] = values.max()
            min_values[qr[s][i]] = values.min()

    # In very rare cases, the function that calculates the QRS complexes can fail to
    # capture many of them. If there are 0, 1, or 2 QRS complexes detected for any signal,
    # then this function produces an IndexError.
    # Even though the following is not the best way to proceed, we will do so in order
    # to have an estiamte for every signal of where the QRS complexes are.
    # Note that if all signals have only 0, 1 or 2 QRS complexes detected, then this
    # function will fail as a whole.
    except IndexError:
        # Check if there are signals with more than 2 QRS
        max_qrs = 0
        s_to_use = None
        for j in range(len(qr)):
            # If there are more than in the previous iteration,
            # we store the amount there are and also the signal which
            # contains them
            if max_qrs < len(qr[j]):        
                max_qrs = len(qr[j])
                s_to_use = j
        # Create a synthetic array that contains repeated QRS locations
        # for the signal to use that was detected
        aux_qr = []
        for j in range(len(qr)):
            aux_qr.append(qr[s_to_use])

        for i in range(len(aux_qr[s])):
            if i==0:
                # The first one is a special case since we ommited the first
                # QRS when we calculated them, so we use an interval around it
                # using the distance to the next one
                lower_middle_point = int((aux_qr[s][i]+aux_qr[s][i+1])/2 - aux_qr[s][i])
                upper_middle_point = int((aux_qr[s][i]+aux_qr[s][i+1])/2)            
            elif i==len(aux_qr[s])-1:
                # The last one is also a special case, since we now want to
                # close the chunk at the furthest point to the right
                lower_middle_point = int((aux_qr[s][i]+aux_qr[s][i-1])/2)
                upper_middle_point = len(x)
            else:
                # For all the other ones, we simply use the middle points between
                # QRS complexes as lower and upper bounds
                lower_middle_point = int((aux_qr[s][i]+aux_qr[s][i-1])/2)
                upper_middle_point = int((aux_qr[s][i]+aux_qr[s][i+1])/2)
        
            # Generate the chunk with the bounds
            around_qr[aux_qr[s][i]] = np.array(range(lower_middle_point, upper_middle_point))
    
        # We prevent index errors for the first chunk, since it could have negative indexes
        # if the lower_middle_point was a negative value.
        filtered = [value > 0 for value in around_qr[aux_qr[s][0]]]
        around_qr[aux_qr[s][0]] = around_qr[aux_qr[s][0]][filtered]

        # For each chunk, find the max and min values
        max_values = dict()
        min_values = dict()
        for i in range(len(aux_qr[s])):
            values = x[around_qr[aux_qr[s][i]]]
            max_values[aux_qr[s][i]] = values.max()
            min_values[aux_qr[s][i]] = values.min()

    # Return the points if desired
    if return_points:
        return pd.Series(max_values), pd.Series(min_values)

    # We calculate the medians of both the max
    # and min values
    max_median = np.median(list(max_values.values()))    
    min_median = np.median(list(min_values.values()))    
    
    # We will use a dislpacement factor to catch whenever there are values 
    # considerably far from the average amplitude of the signal
    var_f = np.power(max_median - min_median, 1/4) / 2
    # var_f = np.power(max_median - min_median, 1/2) / 4
    
    return max_median, min_median, var_f


def values_beyond_median_peaks(x, qrs):

    """
    For an ECG, calculate how many joint points
    of each signal have gone beyond the median peaks
    across time.
    """

    # We will store the values in this array
    beyonds = np.zeros_like(x)
    # For each signal, calculate the median peaks and store whether
    # the observations went beyond them or not
    for s in range(x.shape[1]):
        xx = x[:, s]        
        max_median, min_median, var_f = calculate_median_peaks(xx, qrs, s)
        beyonds[:, s] = (xx > max_median + var_f) | (xx < min_median - var_f)
        
    # Sum across signals to retrieve if they went beyond the threshold
    # jointly
    beyonds_sum = beyonds.sum(1)
    return beyonds_sum


def get_inliers(path, first_rule, second_rule, return_all=False):

    """
    Look for the file which contains the criteria to detect outliers,
    and pull the indexes of the inliers.
    """
    # Read the file with the outliers indexes
    beyond_either = pd.read_csv(path + 'qrs_hr/' + 
                                f'beyond_either_{first_rule}_{second_rule}.csv', index_col=0)
    beyond_either = beyond_either.reset_index()
    beyond_either.columns = ['original_index'] + list(beyond_either.columns[1:])

    # We will only keep the non-outliers (inliers). However,
    # we want to keep the previous indexes in case we need them
    inliers = beyond_either[beyond_either['outlier'] == 0]
    inliers = inliers.reset_index(drop=True).reset_index()
    inliers.columns = ['new_index'] + list(inliers.columns[1:])

    # Get the inliers indexes
    inliers_idx = inliers['original_index'].values
    inliers_idx = list(inliers_idx)

    if return_all:
        return inliers_idx, beyond_either, inliers
    
    return inliers_idx


def predict_mi(x, model, threshold=0.5):

    """
    Predict with the model whether there was an MI or not, 
    for the given threshold.
    """
    
    y_score = model.predict(x)    
    y_pred  = np.array([1 if score > threshold else 0 for score in y_score])
    
    return y_pred, y_score


def calculate_metrics_mi(y_true, y_pred, y_score, first_to_target_recall=True, target_recall=0.9,
                         equal_recall_spec=True):
        
    """
    Calculate all the metrics for the clasification problem of MI vs no MI.
    """

    recall       =  recall_score(y_true, y_pred)
    precision    =  precision_score(y_true, y_pred)
    specificity  =  recall_score(1-y_true, 1-y_pred)
    f1           =  f1_score(y_true, y_pred)
    cf           =  confusion_matrix(y_true, y_pred)
    roc_auc      =  roc_auc_score(y_true, y_score)
    roc_curve_   =  roc_curve(y_true, y_score)
    accuracy     =  accuracy_score(y_true, y_pred)

    # Get location of threshold = 50%
    _, _, thresholds  =  roc_curve_
    t                 =  thresholds - 0.5
    t                 =  t ** 2
    threshold_loc     =  np.argmin(t)
    
    metrics_dict = {'roc_auc' : roc_auc, 'recall' : recall, 'precision' : precision, 
                    'specificity' : specificity, 'f1' : f1, 
                    'confusion_matrix' : cf,  'roc_curve' : roc_curve_, 
                    'accuracy' : accuracy, 'threshold_loc' : threshold_loc}
    
    if first_to_target_recall:
        
        fpr, tpr, thresholds                     =  roc_curve_
        metrics_dict['target_threshold_loc']     =  np.argmax(tpr > target_recall) + 1
        metrics_dict['target_threshold']         =  thresholds[metrics_dict['target_threshold_loc']]
        target_pred                              =  np.array([1 if score > metrics_dict['target_threshold']
                                                              else 0 for score in y_score])
        
        metrics_dict['target_recall']            =  recall_score(y_true, target_pred)
        metrics_dict['target_precision']         =  precision_score(y_true, target_pred)
        metrics_dict['target_specificity']       =  recall_score(1-y_true, 1-target_pred)
        metrics_dict['target_f1']                =  f1_score(y_true, target_pred)
        metrics_dict['target_confusion_matrix']  =  confusion_matrix(y_true, target_pred)
        metrics_dict['target_accuracy']                 =  accuracy_score(y_true, target_pred)
        
    if equal_recall_spec:

        fpr, tpr, thresholds                    =  roc_curve_
        equal                                   =  fpr - (1 - tpr)
        equal                                   =  equal ** 2
        metrics_dict['equal_threshold_loc']     =  np.argmin(equal)
        metrics_dict['equal_threshold']         =  thresholds[metrics_dict['equal_threshold_loc']]
        equal_pred                              =  np.array([1 if score > metrics_dict['equal_threshold']
                                                              else 0 for score in y_score])
        
        metrics_dict['equal_recall']            =  recall_score(y_true, equal_pred)
        metrics_dict['equal_precision']         =  precision_score(y_true, equal_pred)
        metrics_dict['equal_specificity']       =  recall_score(1-y_true, 1-equal_pred)
        metrics_dict['equal_f1']                =  f1_score(y_true, equal_pred)
        metrics_dict['equal_confusion_matrix']  =  confusion_matrix(y_true, equal_pred)
        metrics_dict['equal_accuracy']          =  accuracy_score(y_true, equal_pred)

    return metrics_dict


def divide_x_in_pieces(x, n_pieces, return_q05_q95=True):

    # Define the pieces
    pieces = np.linspace(0, 5000, n_pieces+1)
    pieces = pieces.astype(int)

    # Create a DF
    x_df = pd.DataFrame(x)
    x_df.columns = ['value']

    # Assign a column that tells which registers belong to which piece
    pieces_str = [str(pieces[i])+':'+str(pieces[i+1]) for i in range(n_pieces)]
    pieces_str = np.repeat(pieces_str, 5000 / n_pieces)
    x_df['pieces'] = pieces_str

    if return_q05_q95:
        q_05_95 =  x_df.groupby('pieces').quantile([0.025, 0.975]).reset_index().\
                   pivot(index='pieces', columns='level_1', values='value')
        q95_q95 = np.quantile(q_05_95[0.975], 0.95)
        q05_q05 = np.quantile(q_05_95[0.025], 0.05)

        return x_df, q05_q05, q95_q95

    else:
        return x_df

    


def sample_Y_by_code(Y_df, code, n_sample=5, random_state=203129):

    """
    Keep only the rows in which there is a positive diagnostic i.e. value of 1
    for the given SNOMED CT Code, and sample n_sample values to plot them.
    """
    Y_code = Y_df[code][Y_df[code].apply(lambda x : True if x > 0 else False)]
    Y_code = Y_code.sample(n_sample, random_state = random_state)

    return Y_code


def dx_by_urgency(urgency_level, requirements):

    """
    Filter the diagnoses by their urgency level, and by whether or not the current db
    actually contains registers of those.
    """

    # Filter given the urgenvy level
    urgency = requirements['labels'][requirements['labels']['Urgency'] == urgency_level]

    # Filter given the db at hand
    urgency = urgency[urgency[requirements['current_db']] != 0]

    # Keep important columns
    urgency = urgency[['Diagnostic Description', 'Abbreviation', requirements['current_db'], 'Total', 'Kind']]

    return urgency

    
def dx_by_kind(kind_level, requirements):

    """
    Filter the diagnoses by their kind (Most vs Less Frequent), and by whether or not the current db
    actually contains registers of those.
    """

    # Filter given the urgenvy level
    kind = requirements['labels'][requirements['labels']['Kind'] == kind_level]

    # Filter given the db at hand
    kind = kind[kind[requirements['current_db']] != 0]

    # Keep important columns
    kind = kind[['Diagnostic Description', 'Abbreviation', requirements['current_db'], 'Total', 'Urgency']]

    return kind

def calculating_class_weights(y_true, linear_factor=2, exp_factor=None):

    """

    """

    if exp_factor is None:
        weights = 1 / (y_true.sum(0) / y_true.sum()) / linear_factor
    
    else:
        weights = (1 / (y_true.sum(0) / y_true.sum())) ** exp_factor

    keys = np.arange(0,y_true.shape[1],1)
    weight_dictionary = dict(zip(keys, weights))

    return weight_dictionary
    

def age_and_sex_set(values, indices, mean=None):
    
    """
    Retrieve the age and sex from a list of indices. 
    Manipulate them so that they are integer columns.
    """
    
    # Retrieve the selected values given the indices
    age_n_sex = values.iloc[indices][['age', 'sex']]

    # Calculate the mean if it is not provided
    if mean is None:
        aux = age_n_sex['age'][age_n_sex['age'] != 'NaN']
        aux = aux.astype(int)
        aux = aux.mean().round()
        mean_age = int(aux)

    else:
        mean_age = mean

    # There are some 'NaN' values, so we set them to be the mean
    age_n_sex['age'] = age_n_sex['age'].apply(lambda x : mean_age if x == 'NaN' else int(x))
    
    # Bolleanize the sex column
    age_n_sex['sex'] = age_n_sex['sex'].apply(lambda x : 1 if x == 'Male' else 0)

    #Convert all to np.array
    age_n_sex = np.array(age_n_sex)
    
    # Return the mean to use it for the other sets.
    if mean is None:
        return age_n_sex, mean_age
    
    else:
        return age_n_sex


def fit_loess(x, y, frac=0.2):
    
    still_not_fitted = True
    
    while still_not_fitted:
        
        try:
            _, fitted_y, _ = loess_1d(x, y, degree=2, frac=frac)                        
            return fitted_y
            
        except np.linalg.LinAlgError:
            print(f'frac = {frac} generates LinAlgError')
            if frac > 1:
                print('The fraction cannot be greater than 1.')
            else:
                frac += 0.02
                pass


def smooth_ecg_signal(x, y, std_threshold = 2, n_intervals = 50, outlier_frac = 0.01, 
                      gross_frac = 0.1, use_lowess = True, analysis=False):


    stds = []
    lowess_by_intervals = []

    n_in_each_interval = 5000 // n_intervals
    join_counter = 0

    for k in range(n_intervals):
        
        begin = k * n_in_each_interval
        end =  (k+1) * n_in_each_interval
        
        yy = y[begin:end]
        xx = x[begin:end]
        
        std = np.std(yy)
        stds.append(std)
        
        if std > std_threshold:
            if join_counter > 0:
                
                if use_lowess:
                    fitted_y = lowess(yyy, xxx, frac=gross_frac)[:, 1]
                    lowess_by_intervals.append(fitted_y)            
                    join_counter = 0
                else:
                    fitted_y = fit_loess(xxx, yyy, frac=gross_frac)
                    lowess_by_intervals.append(fitted_y)            
                    join_counter = 0
                        
            fitted_y = lowess(yy, xx, frac=outlier_frac)[:, 1]
            lowess_by_intervals.append(fitted_y)
        
        else:
            if join_counter == 0:            
                yyy = yy
                xxx = xx
                join_counter +=1
            else:
                yyy = np.append(yyy, yy)
                xxx = np.append(xxx, xx)
                join_counter +=1
                
    if join_counter > 0:
        if use_lowess:
            fitted_y = lowess(yyy, xxx, frac=gross_frac)[:, 1]
            lowess_by_intervals.append(fitted_y)                        
        else:
            fitted_y = fit_loess(xxx, yyy, frac=gross_frac)
            lowess_by_intervals.append(fitted_y)

    if analysis == False:
        return  np.concatenate(lowess_by_intervals)
    else:
        return np.concatenate(lowess_by_intervals), \
               [std_threshold,  np.array(stds)]

def clean_ecg_signal(y, smoothed_y, low_q=0.025, upp_q=0.995, apply_quantiles=True, apply_butter=True, Wn=0.2, analysis=False):

    # The new Y is the original minus the smoothed one.
    y_new = y - smoothed_y 
    
    if apply_quantiles:
        y_corrected = y_new.copy()
        lower_q = np.quantile(y_corrected, low_q)
        upper_q = np.quantile(y_corrected, upp_q)
        below_lower_q = y_corrected[y_corrected < lower_q] # Used if analysis==True
        above_upper_q = y_corrected[y_corrected > upper_q] # Used if analysis==True
        y_corrected[y_corrected < lower_q] = lower_q
        y_corrected[y_corrected > upper_q] = upper_q

        if apply_butter:            
            b, a = scipy.signal.butter(4, Wn, 'low', analog = False)
            y_butter = scipy.signal.filtfilt(b, a, y_corrected)

            if analysis == False:
                return y_new, y_corrected, y_butter
            else:
                return y_new, y_corrected, y_butter, \
                       [low_q, lower_q, below_lower_q], \
                       [upp_q, upper_q, above_upper_q]

        else:
            if analysis == False:
                return y_new, y_corrected
            else:
                return y_new, y_corrected, \
                       [low_q, lower_q, below_lower_q], \
                       [upp_q, upper_q, above_upper_q]

    else:
        return y_new


def pad_zeros_before_n_after(values, cut):

    a = values.copy()

    left_cut = int(a.shape[1] * (1 - cut) / 2)
    right_cut = a.shape[1] -  left_cut

    left_zeros = np.zeros((a.shape[0], left_cut, a.shape[2]))
    right_zeros = left_zeros

    a = a[: , left_cut:right_cut, :]
    a = np.concatenate([left_zeros, a, right_zeros], 1)
    
    return a

### !!! (add citation) !!!
# Function from #1 team in Physionet Challenge 
def apply_filter(signal, filter_bandwidth, fs=500):
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                 order=order, frequency=filter_bandwidth, 
                                 sampling_rate=fs)
    return signal

def correct_static_noise_label(x):
    
    """
    Correct the labels from the static_noise column.
    Note however that this function was only used to work with the most common 17 labels,
    and all of the other ones could still have errors, but we aren't actuallñy working with those.
    """
    
    x = str(x)
    
    if x == 'nan':
        return np.nan
    
    x = x.upper()
    x = x.replace(" ", "")
    x = x.split(',')
    x = [v for v in x if v!='']
    
    return ','.join(x)

def has_signal(signal, signals):
    if signal in signals:
        return 1
    else:
        return 0


def get_noise_by_signal(data, row, noise_labels = ['I-AVR', 'I-AVF', 'ALLES', 
                                                   'I-V1', 'I-V2', 'I-AVL', 
                                                   'I,II', 'ALLES,NOISYRECORDING',
                                                   'V1', 'I,III,AVL', 'I,II,AVR', 'V2']):

    a = data[row]

    if a not in noise_labels:

        signals_dict = {signal:np.nan for signal in signals} 

        return pd.DataFrame(signals_dict, index = [row])

    a = a.split(',')

    b = []

    alles_keywords = ['ALLES','ALLES,NOISYRECORDING']  

    if a[0] in alles_keywords:

        b = signals

    else:

        for v in a:        

            if '-' in v:

                v_range = v.split('-')
                v_start = signals.index(v_range[0].upper())
                v_end = signals.index(v_range[1].upper())            

                for k in range(v_start, v_end):

                    b.append(signals[k])

                b.append(v_range[1])

            else:

                if v.upper() in signals:

                    b.append(v)

    signals_dict = {signal:[has_signal(signal, b)] for signal in signals}

    return pd.DataFrame(signals_dict, index = [row])

def add_static_noise(xx, add_noise_threshold=0.8, noise_threhsold=0.4):

    # Run for each of the signals
    for sig in range(0, 11):

        # Get a random number in (0,1). If it is superior
        # than a threshold, then add noise to that signal
        add_noise = np.random.uniform()
        if add_noise > add_noise_threshold:

            # Get a reasonable standard deviation to calculate the gaussian noise
            std = max(np.quantile(xx[:, sig], 0.6), -1 * np.quantile(xx[:, sig], 0.4))
            # Refactor it to overall make it smaller, but sometimes higher
            std = std * np.random.uniform(0.6, 1.2, size=len(xx[:, sig]))

            # Calculate random noise for all the signal
            xx_static_noise = np.random.normal(scale=std, size=len(xx[:, sig]))
            # We calculate random numbers in (0,1)
            xx_static_noise_idx = np.random.uniform(size=len(xx[:, sig]))
            # We only add the gaussian noise whenever these random values are above a threshold
            xx_static_noise = np.multiply(xx_static_noise , xx_static_noise_idx > noise_threhsold)
            # And we add the noise
            xx[:, sig] = xx[:, sig] + xx_static_noise

    return xx

def add_static_noise_to_many(values, add_noise_threshold_limits=(0.2, 0.9), noise_threhsold_limits=(0.2, 0.9)):

    # Run for each register
    for k in range(len(values)):

        # Calculate random values for the threshold, in order to have a diverse set of noise registers
        add_noise_threshold = np.random.uniform(add_noise_threshold_limits[0], add_noise_threshold_limits[1])
        noise_threhsold = np.random.uniform(noise_threhsold_limits[0], noise_threhsold_limits[1])

        # Add noise
        values[k] = add_static_noise(values[k], add_noise_threshold, noise_threhsold)

    return values


########################################################################
########################################################################

##### Data Visualization

def plot_ecg(waves, ylim=(-6,6), figsize=(16,10), metadata=None, dims=[6, 2], show=True, M=None, vlines=None):

    """
    Plot the 12-lead ecg. Note this function assumes we are using the standard 12 leads in the 
    following order:
    
    signals = ['I', 'II', 'III', 'AVL', 'AVR', 'AVF', 'V1', 'V2','V3','V4','V5', 'V6']
    
    Inputs:
        waves - np.array, shape=(n, 12); n is the number of simulations for each individual signal.
        ylim - float; sets the lower & upper limit for every signal.
        
    
    Possible Upgrades:
        Don't immediately assume it has 12 channels, nor the order. Instead pass both as parameters.
        The y limits by themselves are not good enough to represent a more realistic image. We could
        use some overlapping between waves.
        Maybe the patients id and the diagnostic can be added at the top.
    """
    
    fig, axs = plt.subplots(dims[0],dims[1],figsize=figsize, facecolor='w', edgecolor='k')    

    # Adjust the spacing and axs given the dimension chosen. Note that this function is
    # optimized for the [6,2] case.

    if dims == [12,1]:
        fig.subplots_adjust(hspace = -0.11)
        axs = axs.ravel()

    elif dims == [6,2]:
        fig.subplots_adjust(hspace = -0.11, wspace=0.15)
        axs = axs.T.ravel()

    else:
        print("Please enter a valid dimension, either [12,1] or [6,2]")
        return None

    # Plot the waves, setting the appropriate labels and adjusting the limits.

    if isinstance(waves, tuple):
        for k, wave in enumerate(waves):
            alpha = 1 if k == 0 else 0.7
            for channel in range(12):
                axs[channel].plot(range(len(wave[:, channel])), wave[:, channel], alpha=alpha)
                axs[channel].set_ylabel(signals[channel], fontsize=14, rotation=0, labelpad=20)
                axs[channel].set_ylim(ylim[0], ylim[1])
    else:
        for channel in range(12):            
            axs[channel].plot(range(len(waves[:, channel])), waves[:, channel])
            axs[channel].set_ylabel(signals[channel], fontsize=14, rotation=0, labelpad=20)
            axs[channel].set_ylim(ylim[0], ylim[1])

    # Add vertical lines
    if vlines is not None:
        for channel in range(12):
            for vline in vlines:
                axs[channel].axvline(vline, ls='--', color='gray', linewidth=2)

    # Add additional information that will appear on the title, as well as choose to save
    # the plot on a given path.

    if isinstance(metadata, dict):        
        
        metadata_ = get_metadata(metadata["info"], metadata["labels"])

        # Correct very long lists so that there is an "enter" between the dx
        if len(metadata_['dx']) > 5:            
            metadata_['dx'] = '[' + ', '.join(metadata_['dx'][:5]) + ',\n' + ', '.join(metadata_['dx'][5:]) + ']'

        sex_corr = "    " if str(metadata_['sex']) == "Male" else "" # Add spaces if it's Male
        meta_title = "Dimensions:                " + str(metadata_['dimension']) + "\n" \
                   + "Duration (seconds):             " + str(metadata_['duration'])  + "\n" \
                   + "Age:                                        " + str(metadata_['age'])  + "\n"\
                   + "Sex:                                 " + sex_corr + str(metadata_['sex'])  + "\n\n"\
                   + "Dx: " + str(metadata_['dx'])
        plt.suptitle(meta_title, fontsize=16, y = 1.05)

        if isinstance(metadata['path'], str):
            plt.savefig(metadata['path']+'.png', bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()
        


def plot_various_given_code(Y_code, code, requirements, n_sample=5, show=False, verbose=False):

    """"
    Plot and save n_sample ecgs that belong to the same diagnosis (code).

    Inputs:
        Y_code
    """

    for k in range(n_sample):
        
        # This is the actual index to find the needed signals in X
        f_number = Y_code.index[k]        
        
        # Retrieve the signal and metadata
        ecg_file =  wfdb.rdsamp(requirements['path'] + requirements['hea_files'][f_number])
        
        ### Define the paths to store the plot in two folders:
        ### dx : The folder names are the SNOMED CT Codes themselves
        ### dx_desc : The folder names are the descriptions for the respective codes
        
        # dx
        ecg_path = requirements['imgs_path'] + requirements['current_db'] + "/ecg/dx/" + str(code)
        make_dir(ecg_path)        
        
        # dx_desc
        ecg_desc_path = requirements['imgs_path'] + requirements['current_db'] + "/ecg/dx_desc/" \
                                 + requirements['labels'].loc[code]['Diagnostic Description'].replace(' ', '_')
        make_dir(ecg_desc_path)
        
        ### Define the metadata, plot and store in both aforementioned folders
        
        # dx
        metadata = {"info": ecg_file, "labels": requirements['labels'],
                    "path": ecg_path + '/' + requirements['hea_files'][f_number]}
        plot_ecg(ecg_file[0], metadata=metadata, show=show)
        
        # dx_desc
        metadata = {"info": ecg_file, "labels": requirements['labels'],
                    "path": ecg_desc_path + '/' + requirements['hea_files'][f_number]}
        plot_ecg(ecg_file[0], metadata=metadata, show=False) # It suffices to plot 1

        if verbose:
            print("Register number ", f_number, " plotted")


def plot_original_vs_corrected(axs, original, median, corrected, filter_size, padding):
    
    """
    Compare the original ECG vs it's median filter and the corrected series.
    """
    axs[0].plot(original)
    axs[0].plot(median, lw=3)
    axs[0].set_title(f'Original & Median\nfilter={filter_size}, padding={padding}')
    axs[1].plot(corrected)
    axs[1].set_title(f'Corrected')
    
def plot_double_median_filter(original, og_median, corrected, cr_median):

    """
    Plot the 2 step median filter.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.4)

    axs[0].plot(original)
    axs[0].plot(og_median)
    axs[0].set_title('Original & Median')

    axs[1].plot(corrected)
    axs[1].plot(cr_median)
    axs[1].axhline(y = 0, color = 'black', linestyle = '--')
    axs[1].set_title('Corrected & Median of Corrected')

    axs[2].plot(cr_median)
    axs[2].axhline(y = 0, color = 'black', linestyle = '--')
    axs[2].set_title('Differences vs 0')
    

def plot_qrs(signal, xqrs, title=''):

    """
    Plot signal and its QRS segments.

    Based on wfdb.plot_items.
    """
    fig, ax = plt.subplots(figsize=(8, 4))    
    ax.plot(signal)
    ax.scatter(xqrs, signal[xqrs], marker="*", color='red')
    ax.set_title(title, fontsize=14)


def top_n_dx_plot(n, Y_df, requirements, m=0, urgency_colors=urgency_colors, title=None):
    """
    Plot the top n DX while taking into account their urgency level
    """

    # Retrieve the most frequent dx given the Y_df
    top_n_dx = Y_df.sum(0).sort_values(ascending=False)[m:(n+m)]
    n = len(top_n_dx)
    top_n_dx = pd.DataFrame(top_n_dx)
    top_n_dx = top_n_dx.join(requirements['labels'])[[0, 'Diagnostic Description', 'Urgency']]
    top_n_dx.columns = ['counts',  'dx', 'urgency']
    top_n_dx['color'] = top_n_dx['urgency'].apply(lambda x : urgency_colors[x])

    # We create a horizontal barplot in order to better show the dx description
    fig, ax = plt.subplots(figsize=(12,12))
    ax.barh(top_n_dx['dx'].astype(str), top_n_dx['counts'], color=top_n_dx['color'])
    plt.gca().invert_yaxis()

    # We add labels to show the urgency level color
    handles = [Line2D([0], [0], linewidth = 10, color=urgency_colors[k]) for k in range(1,5)]
    labels = ['Urgency 1', 'Urgency 2', 'Urgency 3', 'No Urgency']
    ax.legend(handles = handles, labels = labels, fontsize = 18, frameon=False, loc='lower right')

    # Get the rectangles/bars for each of the dx
    rectangles = [x for x in ax.get_children() if isinstance(x, matplotlib.patches.Rectangle)][:-1]

    # For each, add the text at the end of the bar
    for rectangle in rectangles:
        ax.text(rectangle.get_width() - top_n_dx['counts'].min()/2,
                rectangle.xy[1] + rectangle.get_height()/1.6,
                rectangle.get_width(),
                size=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)  )

    # Final adjustments
    ax.tick_params(axis='y', which='major', labelsize=12)    
    if isinstance(title, str):
        ax.set_title(title, fontsize = 18, y = 1.02)
    else:
        if m != 0:
            print("Beware if you set a value for m != 0, since it avoids the first m values, then \
                   the following title is misleading.")  
        ax.set_title(f'Top {n} most frequent Dx', fontsize = 18, y = 1.02)
        

def compare_2_ecgs(filename_1, filename_2, requirements, ylim_1 = (-5,5),  ylim_2 = (-5,5), show_unique_n_nan=True) :

    # From https://physionet.org/content/challenge-2020/1.0.2/
    example_1 = wfdb.rdsamp(filename_1)
    metadata_1 = {"info": example_1, "labels": requirements['labels'], "path":None}

    # From https://www.kaggle.com/code/bjoernjostein/physionet-challenge-2020/data?select=WFDB
    example_2 = wfdb.rdsamp(filename_2)
    metadata_2 = {"info": example_2, "labels": requirements['labels'], "path":None}

    print("File 1\n", example_1[0])
    print("\nFile 2\n", example_2[0])
    print("\nFile 1\n")
    print_metadata(example_1, requirements['labels'])
    print("\nFile 2\n")
    print_metadata(example_2, requirements['labels'])        

    if show_unique_n_nan:
        # Divide to get the constant factor
        frac = example_1[0]/example_2[0]
        print('\n\nWe can see that the constant is 0.2 (or 5, depending on the order): (Unique values of the divisions)',
            np.unique(frac), "\n")

        # Check that the nan values are caused by 0 divisions
        frac = frac.ravel()
        print('Sum of the squared denominators that caused nan: ', 
            (example_2[0].ravel()[~np.isclose(frac, 0.2)]**2).sum())

    # Plot with different ylim to adjust for the factor
    
    plot_ecg(example_1[0], ylim=ylim_1, figsize=(16,6))
    plot_ecg(example_2[0], ylim=ylim_2, figsize=(16,6))


def plot_model_history(fitted_model, metrics={'titles':['Loss', 'Accuracy'], 'metrics':['loss', 'accuracy']}):


    if len(metrics['titles']) == 1:

        fig, ax = plt.subplots(figsize = (16, 5))    

        # Loss
        loss_df = pd.DataFrame({'train' : fitted_model[metrics['metrics'][0]],
                                'val' : fitted_model['val_'+metrics['metrics'][0]]})
        ax.plot(loss_df)
        ax.set_xlabel('epoch')
        ax.set_title(metrics['titles'][0], fontsize = 18)
        ax.legend(['train', 'val'], frameon=False, fontsize=12)

    elif len(metrics['titles']) == 2:

        fig, axs = plt.subplots(1,2, figsize = (16, 5))    

        # Loss
        loss_df = pd.DataFrame({'train' : fitted_model[metrics['metrics'][0]],
                                'val' : fitted_model['val_'+metrics['metrics'][0]]})
        axs[0].plot(loss_df)
        axs[0].set_xlabel('epoch')
        axs[0].set_title(metrics['titles'][0], fontsize = 18)
        axs[0].legend(['train', 'val'], frameon=False, fontsize=12)

        # Accuracy
        acc_df = pd.DataFrame({'train' : fitted_model[metrics['metrics'][1]],
                            'val' : fitted_model['val_'+metrics['metrics'][1]]})
        axs[1].plot(acc_df)
        axs[1].set_xlabel('epoch')
        axs[1].set_title(metrics['titles'][1], fontsize = 18)
        axs[1].legend(['train', 'val'], frameon=False, fontsize=12)
    


### The following 2 can be generalized but for the moment we leave them as they are for convenience

def plot_confusion_matrix(model, x, y, y_labels, n_ex = None, random_state=203129, thresholds=None, return_cm=False):

    if isinstance(x, tuple):
        deep = x[0]
        wide = x[1]
        

    else:
        deep = x

    if n_ex is None:
        indices = np.arange(0, deep.shape[0], 1)

    else:
        random.seed(random_state)
        indices = random.sample(range(deep.shape[0]), n_ex)

    if thresholds is not None:

        if 'PTB-XL' in thresholds.keys():

            thresholds = thresholds['PTB-XL']

            if isinstance(x, tuple):
                probs  = model.predict((deep[indices], wide[indices]))

            else:
                probs  = model.predict(deep[indices])

            preds = np.copy(probs)

            for k in range(n_ex):                                

                if probs[k, thresholds['MI']['idx']] > thresholds['MI']['threshold']:
                    preds[k, thresholds['MI']['idx']] = 1

                elif probs[k, thresholds['MI&STTC']['idx']] > thresholds['MI&STTC']['threshold']:
                    preds[k, thresholds['MI&STTC']['idx']] = 1
                    # print(probs[k])                
                    
            
            preds = preds.argmax(1)
            preds = preds.astype(int)

    else:
        if isinstance(x, tuple):
                preds  = model.predict((deep[indices], wide[indices]))

        else:
            preds  = model.predict(deep[indices])
        preds = preds.argmax(1)
        preds = preds.astype(int)

    trues = y[indices].argmax(1)

    cm = confusion_matrix(trues, preds)

    if return_cm:
        return cm

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        ((cm.transpose()/cm.sum(1)).transpose()).flatten()]    

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(y.shape[1],y.shape[1])


    fig, ax = plt.subplots(figsize = (8,8))

    ax = sns.heatmap(cm,
                annot=labels, 
                fmt='',
                cmap='Blues',
                annot_kws = {'fontsize':14})

        
    ax.set_title('Confusion Matrix\n', fontsize = 20);
    ax.set_xlabel('\nPredicted Values', labelpad = 18, fontsize = 16)
    ax.set_ylabel('Actual Values', labelpad = 20, fontsize = 16);

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(y_labels, fontsize = 14)
    ax.yaxis.set_ticklabels(y_labels, fontsize = 14, rotation = 45)


def plot_confusion_matrix_mi(cm, title=None):

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        ((cm.transpose()/cm.sum(1)).transpose()).flatten()]    

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)


    fig, ax = plt.subplots(figsize = (8,8))

    ax = sns.heatmap(cm,
                     annot=labels, 
                     fmt='',
                     cmap='Blues',
                     annot_kws = {'fontsize':14})

        
    # ax.set_title('Confusion Matrix\n', fontsize = 20);
    ax.set_xlabel('\nPredicted Values', labelpad = 18, fontsize = 16)
    ax.set_ylabel('Actual Values', labelpad = 20, fontsize = 16);

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['OTHER', 'MI'], fontsize = 14)
    ax.yaxis.set_ticklabels(['OTHER', 'MI'], fontsize = 14, rotation = 45)

    # Title
    if title is not None:
        ax.set_title(title, fontsize=18, y=1.03)


def plot_roc_curve_mi(fpr, tpr, metrics, include_points=None):

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot ROC curve
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], ls='--', color='gray')

    # Add the points for the specific approache
    if include_points is not None:
        for point, color in include_points.items():

            label = f'Recall = {round(100 * metrics[f"{point}recall"], 2)}%'\
                  + f'\nPrecision = {round(100 * metrics[f"{point}precision"], 2)}%'
            ax.scatter(fpr[metrics[f'{point}threshold_loc']], tpr[metrics[f'{point}threshold_loc']],
                       color=color, label=label, s=100)
            
    # Adjustments
    ax.set_xlabel('False Positive Rate', fontsize=16, labelpad=18)
    ax.set_ylabel('True Positive Rate', fontsize=16, labelpad=18)
    # plt.suptitle('ROC', fontsize=20, y = 0.96)
    ax.set_title(F'AUC = {round(metrics["roc_auc"], 2)}', fontsize=14, y=1.02)
    ax.tick_params(labelsize=14)
    plt.legend(loc='lower right', frameon=False, fontsize=14);


def element_attribution_mi(results, element, filters, add_target_and_equal=True):    
    """
    Obtain the attribution of the neural network elements.
    
    Note! This function works on a heavy assumption that
    has to do with the order in which we ran the models:
    Each element was added consecutively. We will be retreiving
    the iterations in which the element appears last in the 
    string.
    
    Thus, we will be calculating "consecutive" attributions, and
    not quite "real" ones. To get the real ones, we would have
    to run many models with many different combinations of the elements,
    and then we could simply take those in which the element appears
    in the string vs not.
    """

    # Obtain all the iterations where the element appears at the end
    element_filter = [idx for idx in filters if element in idx]
    element_filter = [idx for idx in filters if try_to_convert_to_int(idx.split(element+'_')[-1])]

    # Obtain all the iterations where we did not use the element
    no_element_filter = [idx for idx in filters if element not in idx]

    if element != 'ks':
        # However, we will only retain consecutive cotributions,
        # so we only compare them with the immediate previous
        # iterations
        element_list = list(element_dict.keys())
        element_idx = element_list.index(element)
        no_element_filter = [idx for idx in no_element_filter 
                             if element_list[element_idx-1] in idx]

    # Get pairwise combinations to iterate through them
    combinations = itertools.product(element_filter, no_element_filter)

    differences = dict()
    idx = 0

    # For each combination, obtain the difference in the metrics
    for comb in combinations:

        differences[idx] = dict()

        for metric in ['roc_auc', 'target_precision', 'target_specificity', 
                       'target_f1', 'target_accuracy', 'equal_recall', 'equal_precision', 
                       'equal_specificity', 'equal_f1', 'equal_accuracy']:

            differences[idx][metric] = results.loc[comb[0], metric] - results.loc[comb[1], metric]    

        idx += 1

    ax = calculate_and_plot_attribution(differences, add_target_and_equal)
    ax.set_title(f'{element_dict[element]} Attribution', fontsize=16, y=1.04)    


def cut_increase_attribution_mi(results, low, high, filters, add_target_and_equal=True):
    
    lower_cut_filter = [idx for idx in filters if f'ct_{low}' in idx]
    upper_cut_filter = [idx for idx in filters if f'ct_{high}' in idx]

    differences = dict()
    
    for lower_cut, upper_cut in zip(lower_cut_filter, upper_cut_filter):

        idx = lower_cut.split(f'ct_{low}_')[-1]
        differences[idx] = dict()

        for metric in ['roc_auc', 'target_precision', 'target_specificity', 
                       'target_f1', 'target_accuracy', 'equal_recall', 'equal_precision', 
                       'equal_specificity', 'equal_f1', 'equal_accuracy']:

            differences[idx][metric] = results.loc[upper_cut, metric] - results.loc[lower_cut, metric]    

    ax = calculate_and_plot_attribution(differences, add_target_and_equal)
    ax.set_title(f'{int(100 * float(high))}% Cut vs {int(100 * float(low))}% Cut Attribution', fontsize=16, y=1.04)
    


def preproc_attribution_mi(results, pre_1, pre_2, filters, add_target_and_equal=True):

    pre_1_filter = [idx for idx in filters if pre_1 in idx]
    pre_2_filter = [idx for idx in filters if pre_2 in idx]

    differences = dict()
    for pre_1_, pre_2_ in zip(pre_1_filter, pre_2_filter):

        idx = pre_1_.split(f'{pre_1}_')[-1]
        differences[idx] = dict()

        for metric in ['roc_auc', 'target_precision', 'target_specificity', 
                       'target_f1', 'target_accuracy', 'equal_recall', 'equal_precision', 
                       'equal_specificity', 'equal_f1', 'equal_accuracy']:

            differences[idx][metric] = results.loc[pre_2_, metric] - results.loc[pre_1_, metric]    

    ax = calculate_and_plot_attribution(differences, add_target_and_equal)
    ax.set_title(f'{preproc_dict[pre_2]} vs {preproc_dict[pre_1]} Attribution', fontsize=16, y=1.04)
    


def outlier_removal_attribution_mi(results, ol, filters, add_target_and_equal=True):    

    # Obtain all the iterations where the element appears at the end
    element_filter = [idx for idx in filters if ol in idx]    

    # Obtain all the iterations where we did not use the element
    no_element_filter = [idx for idx in filters if ol not in idx]

    # Get pairwise combinations to iterate through them
    combinations = itertools.product(element_filter, no_element_filter)

    differences = dict()
    idx = 0

    # For each combination, obtain the difference in the metrics
    for comb in combinations:

        differences[idx] = dict()

        for metric in ['roc_auc', 'target_precision', 'target_specificity', 
                       'target_f1', 'target_accuracy', 'equal_recall', 'equal_precision', 
                       'equal_specificity', 'equal_f1', 'equal_accuracy']:

            differences[idx][metric] = results.loc[comb[0], metric] - results.loc[comb[1], metric]    

        idx += 1

    ax = calculate_and_plot_attribution(differences, add_target_and_equal)
    ax.set_title(f'{outlier_dict[ol]} vs Original Attribution', fontsize=16, y=1.04)


def outlier_removal_attribution_mi_2(results, ol, filters, add_target_and_equal=True):    

    """
    This function is very similar to outlier_removal_attribution_mi. The only difference
    lies in which observations are being compared. In the previous one, we use a combi-
    natorial approach, meaning we do not rely on having an exact replica of the iteration
    in each of the element_filter and no_element_filter sets.

    On the otherhand, the approach on this function does require such thing to happen.
    Note that the functions for the cut increase attribution and the preproc attribution
    also use this approach.

    In the future, if these functions fail, we only have to change the approach and to
    always use a combinatorial approach. I felt that using such approach would yield not
    the best results and so I only used it in the elements attribution where it was needed.

    Nevertheless, I created these 2 functions to actually see if there were significant
    differences and there actually was only a little difference in values, but the insight
    that could be drawn from both was the same.
    
    """
    # Obtain all the iterations where the element appears at the end
    element_filter = [idx for idx in filters if ol in idx]    

    # Obtain all the iterations where we did not use the element
    no_element_filter = [idx for idx in filters if ol not in idx]

    differences = dict()

    for pre_1_, pre_2_ in zip(no_element_filter, element_filter):

        idx = pre_2_.split(f'{ol}_')[-1]
        differences[idx] = dict()

        for metric in ['roc_auc', 'target_precision', 'target_specificity', 
                       'target_f1', 'target_accuracy', 'equal_recall', 'equal_precision', 
                       'equal_specificity', 'equal_f1', 'equal_accuracy']:

            differences[idx][metric] = results.loc[pre_2_, metric] - results.loc[pre_1_, metric]    


    ax = calculate_and_plot_attribution(differences, add_target_and_equal)
    ax.set_title(f'{outlier_dict[ol]} vs Original Attribution', fontsize=16, y=1.04)
    

def calculate_and_plot_attribution(differences, add_target_and_equal):
    """
    We use all this code in each of the attrbution functions, so we just
    put it all in one function to clean the code
    """
    # Calculate mean and stdev of the differences
    differences = pd.DataFrame(differences).transpose()    
    means = differences.mean(0)
    stds  = differences.std(0) / np.sqrt(len(differences))

    # Plot them to see if they are above 0.
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(x=means.values, y=means.index, xerr=stds.values, ls='none', color='gray')
    ax.scatter(means.values, means.index, color='black', marker='D',zorder=10)
    ax.axvline(0, ls='--', color='black')

    if add_target_and_equal:
        aux_means = means.loc[[idx for idx in means.index if 'target' in idx]]
        aux_stds = stds.loc[[idx for idx in stds.index if 'target' in idx]]

        ax.errorbar(x=aux_means.values, y=aux_means.index, xerr=aux_stds.values, ls='none', color='#E8BB0D')
        ax.scatter(aux_means.values, aux_means.index, color='#B2900C', marker='D',zorder=10)

        aux_means = means.loc[[idx for idx in means.index if 'equal' in idx]]
        aux_stds = stds.loc[[idx for idx in stds.index if 'equal' in idx]]

        ax.errorbar(x=aux_means.values, y=aux_means.index, xerr=aux_stds.values, ls='none', color='#4AD349')
        ax.scatter(aux_means.values, aux_means.index, color='#127965', marker='D',zorder=10)

    ax.tick_params(which='major', labelsize=14)
    plt.gca().invert_yaxis()

    return ax


def plot_spectogram(y, fs=500, nperseg=200, noverlap=100):

    # https://github.com/awerdich/physionet/blob/master/physionet_processing.py

    # Calculate the spectogram
    frequency, time, amplitude = scipy.signal.spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap)

    fig, axs = plt.subplots(1,2, figsize=(12,6))

    # No log 
    plt.sca(axs[0])
    plt.pcolormesh(time, frequency, amplitude, shading='gouraud', cmap='jet')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Without log')

    # With log
    amplitude = abs(amplitude)
    mask = amplitude > 0
    amplitude[mask] = np.log(amplitude[mask])

    plt.sca(axs[1])
    plt.pcolormesh(time, frequency, amplitude, shading='gouraud', cmap='jet')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('With log');


def plot_smoothing_evolution(x, y, y_smoothed, y_new, y_quantiles, y_butter):

    fig, axs = plt.subplots(3,2, figsize=(20,10))

    fig.subplots_adjust(hspace = 0.45, wspace=0.1)

    ylim = (y.min(), y.max())
    axs[0, 0].plot(x, y, alpha = 0.3)
    axs[0, 0].plot(x, y_smoothed, linewidth=3)
    axs[0, 0].set_ylim(ylim)
    axs[0, 0].set_title("Original vs Smoothed (full range)")

    ylim = (y_new.min(), y_new.max())
    # ylim = (-2, 2)
    axs[1, 0].plot(x, y, alpha = 0.3)
    axs[1, 0].plot(x, y_new)
    axs[1, 0].set_ylim(ylim)
    axs[1, 0].set_title("Original vs Corrected")

    ylim = (y_new.min(), y_new.max())
    # ylim = (-2, 2)
    axs[0, 1].plot(x, y, alpha = 0.3)
    axs[0, 1].plot(x, y_smoothed, linewidth=3)
    axs[0, 1].set_ylim(ylim)
    axs[0, 1].set_title("Original vs Smoothed (zoomed)")

    ylim = (y_new.min(), y_new.max())
    # ylim = (-2, 2)
    axs[1, 1].plot(x, y, alpha = 0.3)
    axs[1, 1].plot(x, y_quantiles)
    axs[1, 1].set_ylim(ylim)
    axs[1, 1].set_title("Original vs Corrected + Quantiles")

    ylim = (y_new.min(), y_new.max())
    # ylim = (-2, 2)
    axs[2, 0].plot(x, y, alpha = 0.3)
    axs[2, 0].plot(x, y_butter)
    axs[2, 0].set_ylim(ylim)
    axs[2, 0].set_title("Original vs Corrected + Quantiles + Butter")

    ylim = (y_butter.min(), y_butter.max())
    # ylim = (-2, 2)
    axs[2, 1].plot(x, y_quantiles, alpha = 0.3)
    axs[2, 1].plot(x, y_butter)
    axs[2, 1].set_ylim(ylim)
    axs[2, 1].set_title("Corrected + Quantiles vs Corrected + Quantiles + Butter")

    plt.show()


########################################################################
########################################################################

##### Other

def make_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def get_metadata(info, labels):

    """
    Get the metadata of the provided ecg. This code might look messy, but that's the way
    the .hea files' info is retrieved.

    Inputs:
        info - tuple, this tuple is the one obtained through wfdb.rdsamp() when we read a
            .hea file. The first argument contains the waves, while the second one contains
            metadata such as gender, age, diagnosis, frequency sample, length, units (mV), etc.
        labels - pd.DataFrame, contains the mapping of SNOMED CT Codes to a description
            of the diagnosis.
    """

    metadata = dict()

    metadata['dimension'] = info[0].shape

    try:
        metadata['fs'] = info[1]['fs']
    except IndexError:
        metadata['fs'] = "No Frequency Sample"

    try:        
        metadata['sig_len'] = info[1]['sig_len']
    except IndexError:
        metadata['sig_len'] = "No Signal Length"

    try:
        metadata['duration'] = info[1]['sig_len']/info[1]['fs']
    except IndexError:
        metadata['duration'] = "No Frequency Sample"

    try:
        metadata['age'] = info[1]['comments'][0][5:]
    except IndexError:
        metadata['age'] = "No Age"

    try:
        metadata['sex'] = info[1]['comments'][1][5:]
    except IndexError:  
        metadata['sex'] = "No Sex"
    
    try:
        metadata['dx'] = [labels.loc[int(dx), 'Diagnostic Description'] 
                                    for dx in info[1]['comments'][2][4:].split(',')]
    except IndexError:
        metadata['dx'] = "No Diagnoses"

    return metadata


def print_metadata(info, labels):

    """
    Print the metadata of the provided ecg.

    Inputs:
        Same as get_metadata()
    """

    metadata = get_metadata(info, labels)

    print("The dimensions of this ECG are: ", metadata['dimension'])
    # print("Frequency sample: ", metadata['fs'])
    # print("Singal length: ", metadata['sig_len'])
    print("Duration (seconds): ",  metadata['duration'])
    print("Patient's age: ", metadata['age'])
    print("Patient's sex: ",  metadata['sex'])
    print("Patient's diagnosis: ", metadata['dx'])


def sound_alert(reps=10, song='school'):

    """
    Make a sound alert. Useful when training large models to
    know when it has ended.
    """
    if song == 'school':
        frequencies = [37, 698, 523, 587, 440, 37, 698, 784, 880, 698, 37, 37] * reps
        durations = [1100] * reps * len(frequencies)

        for frequency, duration in zip(frequencies, durations):
            winsound.Beep(frequency, duration)



def get_number_of_parameters(model, print_it=True, path=None):
    """
    Given a model, obtain the amount of parameters it has
    """
    trainable     = model.trainable_variables
    non_trainable = model.non_trainable_variables

    trainable_sum = 0
    non_trainable_sum = 0

    for i in range(len(trainable)):
        s = np.prod(trainable[i].shape)
        trainable_sum += s

    for i in range(len(non_trainable)):
        s = np.prod(non_trainable[i].shape)
        non_trainable_sum += s    
        
    total_sum = trainable_sum + non_trainable_sum
    
    if print_it:
        print('========================================================')
        print('Total params: ',         "{:,}".format(total_sum))
        print('Trainable params: ',     "{:,}".format(trainable_sum))
        print('Non-trainable params: ', "{:,}".format(non_trainable_sum))
        print('========================================================')

    if path is not None:
        n_params = pd.DataFrame([total_sum, trainable_sum, non_trainable_sum])
        n_params.index = ['total', 'trainable', 'non_trainable']
        n_params.columns = ['n']
        n_params.to_csv(path + '/n_params.csv')

    else:    
        return total_sum, trainable_sum, non_trainable_sum
    
def filter_metric(key):
    """
    Filter the merics so that we only retain the
    numerical ones.
    """
    if 'threshold' in key:
        return False
    if 'roc_curve' in key:
        return False
    if 'confusion_matrix' in key:
        return False
    
    return True
    
def try_to_convert_to_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False