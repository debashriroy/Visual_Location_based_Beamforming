from __future__ import division

import csv
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model
from tensorflow.python.keras.metrics import Metric
from tensorflow.keras.layers import Dense,concatenate, Dropout, Conv1D, Conv2D, Flatten, Reshape, Activation, Add, Conv2DTranspose, Subtract, MaxPooling2D, Concatenate
from tensorflow.keras.losses import categorical_crossentropy
#from keras.utils.np_utils import to_categorical
from tensorflow.keras import regularizers
#from tensorflow.losses import mean_squared_error
from tensorflow.keras.optimizers import Adadelta,Adam, SGD, Nadam,Adamax, Adagrad
from tensorflow.python.keras.layers.normalization import BatchNormalization
import sklearn
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from ModelHandler import add_model,load_model_structure, ModelHandler
import numpy as np
import argparse
from sklearn.model_selection import KFold
from tqdm import tqdm
import random
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.custom_metrics import *
import pickle
from sklearn.preprocessing import MinMaxScaler

############################
# Fix the seed
############################
seed = 0 # WAS 0
os.environ['PYTHONHASHSEED']=str(seed)
#tf.set_random_seed(seed)
#tf.random_set_seed()
np.random.seed(seed)
random.seed(seed)

from sklearn.feature_selection import SelectKBest

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False

def open_npz(path,key):
    data = np.load(path)[key]
    return data

def save_npz(path,train_name,train_data,val_name,val_data):
    check_and_create(path)
    np.savez_compressed(path+train_name, train=train_data)
    np.savez_compressed(path+val_name, val=val_data)

def top_2_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=2)
def top_3_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=3)
def top_4_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=4)
def top_6_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=6)
def top_7_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=7)
def top_8_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=8)
def top_9_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=9)
def top_10_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=10)
def top_11_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=11)
def top_12_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=12)
def top_13_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=13)
def top_14_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=14)
def top_15_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=15)
def top_20_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=20)

def top_30_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=30)

def top_50_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=50)



########## FUNCTIONS TO CALCULATE F SCORE OF THE MODEL ###############
from tensorflow.keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
######################################################################


########## FUNCTIONS TO CALCULATE F SCORE FOR TOP-K ACCURACIES ###############
from tensorflow.keras import backend as K
def top_k_recall(Y, k):
	"""
	Y : [nb_samples x k] (k predicted labels/neighbours)
	"""
	s=0
	for i in range(Y.shape[0]): # size(0)
		if 1 in Y[i,:][:k]:
			s += 1
	return s / (1. * Y.shape[0])  # size(0)

def top_k_precision(Y, k):
	"""
	Y : [nb_samples x k] (k predicted labels/neighbours)
	"""
	s = 0
    #print(type(Y))
	for i in range(Y.shape[0]):  # size(0)
		s+=np.sum(Y[i,:][:k]) # s+=Y[i,:][:k].sum().numpy()
	return s / (1.*Y.shape[0]*k)  # size(0)

def f1_score(y_true, y_pred):
    precision = top_k_precision(y_true, y_pred)
    recall = top_k_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
######################################################################


def beamsLogScale(y,thresholdBelowMax):
        y_shape = y.shape   # shape is (#,256)

        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs + 1e-30)
            minValue = np.amax(logOut) - thresholdBelowMax
            zeroedValueIndices = logOut < minValue
            thisOutputs[zeroedValueIndices]=0
            thisOutputs = thisOutputs / sum(thisOutputs)
            y[i,:] = thisOutputs
        return y

def getBeamOutput(output_file):

    thresholdBelowMax = 6
    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]

    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y = beamsLogScale(y,thresholdBelowMax)

    return y,num_classes

def custom_label(output_file, strategy='one_hot' ):
    'This function generates the labels based on input strategies, one hot, reg'

    print("Reading beam outputs...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)

    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]
    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y_non_on_hot =  np.array(y)
    y_shape = y.shape

    if strategy == 'one_hot':
        k = 1           # For one hot encoding we need the best one
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs)
            max_index = logOut.argsort()[-k:][::-1]
            y[i,:] = 0
            y[i,max_index] = 1

    elif strategy == 'reg':
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs)
            y[i,:] = logOut
    else:
        print('Invalid strategy')
    return y_non_on_hot,y,num_classes



def balance_data(beams,modal,variance,dim):
    'This function balances the dataset by generating multiple copies of classes with low Apperance'

    Best_beam=[]    # a list of 9234 elements(0,1,...,256) with the index of best beam
    k = 1           # Augment on best beam
    for count,val in enumerate(beams):   # beams is 9234*256
        Index = val.argsort()[-k:][::-1]
        Best_beam.append(Index[0])

    Apperance = {i:Best_beam.count(i) for i in Best_beam}   #Selected beam:  count apperance of diffrent classes
    print(Apperance)
    Max_apperance = max(Apperance.values())                 # Class with highest apperance

    for i in tqdm(Apperance.keys()):
        ind = [ind for ind, value in enumerate(Best_beam) if value == i]    # Find elements which are equal to i
        randperm = np.random.RandomState(seed).permutation(int(Max_apperance-Apperance[i]))%len(ind)

        ADD_beam = np.empty((len(randperm), 256))
        extension = (len(randperm),)+dim
        ADD_modal = np.empty(extension)
        print('shapes',ADD_beam.shape, ADD_modal.shape)

        for couter,v in enumerate(randperm):
            ADD_beam[couter,:] = beams[ind[v]]
            ADD_modal[couter,:] = modal[ind[v]]+variance*np.random.rand(*dim)
        beams = np.concatenate((beams, ADD_beam), axis=0)
        modal = np.concatenate((modal, ADD_modal), axis=0)

    return beams, modal


def meaure_topk_for_regression(y_true,y_pred,k):
    'Measure top 10 accuracy for regression'
    c = 0
    for i in range(len(y_pred)):
        # shape of each elemnt is (256,)
        A = y_true[i]
        B = y_pred[i]
        top_predictions = B.argsort()[-10:][::-1]
        best = np.argmax(A)
        if best in top_predictions:
             c +=1

    return c/len(y_pred)

# import sys
# #reload(sys)
# sys.setdefaultencoding('utf-8')

# def split_with_new_ratio(train,val,test):
#
#     with open('indeces.pkl','rb') as handle:
#         indeces = pickle.load(handle)
#         handle.close()
#     indeces=indeces.squeeze()
#     all = np.concatenate((train, val , test), axis=0)
#
#     train_new = all[indeces[:int(len(all) * .70)]]
#     val_new = all[indeces[int(len(all)*.70) : int(len(all)*.85)]]
#     test_new = all[indeces[int(len(all)*.85) : ]]
#     return train_new, val_new, test_new

import h5py
# def split_with_new_ratio(train,val,test):
#     with h5py.File('indices.h5', 'r') as hf:
#         indeces = hf['indices'][:]
#     indeces=indeces.squeeze()
#     all = np.concatenate((train, val , test), axis=0)
#
#     train_new = all[indeces[:int(len(all) * .70)]]
#     val_new = all[indeces[int(len(all)*.70) : int(len(all)*.85)]]
#     test_new = all[indeces[int(len(all)*.85) : ]]
#     return train_new, val_new, test_new

def split_with_new_ratio(train,val,test):

    all = np.concatenate((train, val , test), axis=0)

    #np.random.shuffle(all)

    train_new = all[:int(len(all) * .70)]
    val_new = all[int(len(all)*.70) : int(len(all)*.85)]
    test_new = all[int(len(all)*.85) : ]
    return train_new, val_new, test_new


def LOS_label(pathToRawCoord):

    los = np.empty([11194, 1], dtype=bool)
    count = 0
    with open(pathToRawCoord+'CoordVehiclesRxPerScene_s008.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == "V":
                if row[9] == 'LOS=0':
                    los[count] = 0
                else:
                    los[count] = 1
                count+=1

    y_los_test = np.empty([9638, 1], dtype=bool)
    count = 0
    with open(pathToRawCoord+'CoordVehiclesRxPerScene_s009.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == "V":
                if row[9] == 'LOS=0':
                    y_los_test[count] = 0
                else:
                    y_los_test[count] = 1
                count+=1


    y_los_train = los[:9234]
    y_los_val = los[9234:]

    return  y_los_train, y_los_val, y_los_test



parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--data_folder', help='Location of the data directory', type=str, default='E:/beam_selection_NU/baseline_data/')
parser.add_argument('--input', nargs='*', default=['coord'],choices = ['img', 'coord'],
help='Which data to use as input. Select from: img, lidar or coord.')
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=32, type=int,help='Batch size')

parser.add_argument('--epochs', default=100, type = int, help='Specify the epochs to train')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--Aug', type=str2bool, help='Do Augmentaion to balance the dataset or not', default=False)
parser.add_argument('--strategy', type=str ,default='one_hot', help='labeling strategy to use',choices=['baseline','one_hot','reg'])
parser.add_argument('--augmented_folder', help='Location of the augmeneted data', type=str, default='E:/Beam_Selection_ITU/beam_selection/baseline_code_modified/aug_data/')

#parser.add_argument('--fusion_architecture', type=str ,default='mlp', help='Whether to use convolution in fusion network architecture',choices=['mlp','cnn'])
parser.add_argument('--k_fold', type=int, help='K-fold Cross validation', default=0)
parser.add_argument('--val_data_folder', help='Location of the test data directory', type=str, default='E:/beam_selection_NU/baseline_data/')
parser.add_argument('--test_data_folder', help='Location of the test data directory', type=str, default='E:/beam_selection_NU/baseline_data/')
parser.add_argument('--img_version', type=str, help='Which version of image folder to use', default='')

parser.add_argument('--restore_models', type=str2bool, help='Load single modality trained weights', default=False)
parser.add_argument('--new_split', type=str2bool, help='Load single modality trained weights', default=True)
parser.add_argument('--model_folder', help='Location of the trained models folder', type=str,default = 'model_folder/')
parser.add_argument('--image_feature_to_use', type=str ,default='custom', help='feature images to use',choices=['v1','v2','custom'])

parser.add_argument('--rawCoord_folder', help='Location of the raw coordinates data directory', type=str, default = "D:/Raymobtime_Coordinate/")
parser.add_argument('--load_los', help='Load LOS/NLOS information from the raw coordinate data', type=str2bool, default =False)
parser.add_argument('--trainOnly_nLOS',help='Train a model with only LOS or NLOS data',  default='none',choices = ['LOS', 'NLOS', 'none'])

parser.add_argument('--training_strategy',help='Train a model with balanced/seperate data,',  default='none',choices = ['balanced','seperate'])

filepath = 'best_weights.wts.h5'









args = parser.parse_args()
print('Argumen parser inputs', args)

if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

###############################################################################
# Outputs
###############################################################################
###############################################################################
# Outputs
###############################################################################
# Read files
output_train_file = args.data_folder+'beam_output\\beams_output_train.npz'
output_validation_file = args.val_data_folder+'beam_output\\beams_output_validation.npz'
output_test_file = args.test_data_folder+'beam_output\\beams_output_test.npz'


if args.strategy == 'default':
    y_train,num_classes = getBeamOutput(output_train_file)
    y_validation, _ = getBeamOutput(output_validation_file)
    y_test, _ = getBeamOutput(output_test_file)
elif args.strategy == 'one_hot' or args.strategy == 'reg':
    y_train_not_onehot,y_train,num_classes = custom_label(output_train_file,args.strategy)
    y_validation_not_onehot,y_validation, _ = custom_label(output_validation_file,args.strategy)
    y_test_not_onehot,y_test, _ = custom_label(output_test_file,args.strategy)

else:
    print('invalid labeling strategy')

if args.load_los:
    y_los_train, y_los_val, y_los_test = LOS_label(args.rawCoord_folder)
    if args.new_split: y_los_train, y_los_val, y_los_test = split_with_new_ratio(y_los_train, y_los_val, y_los_test)

if args.new_split: y_train, y_validation, y_test = split_with_new_ratio(y_train, y_validation, y_test)
print('label shapes', y_train.shape, y_validation.shape, y_test.shape)

############## LOS AND NLOS ##################################################
if args.trainOnly_nLOS == 'LOS':
    y_train = y_train[np.squeeze(y_los_train == 1)]
    y_validation = y_validation[np.squeeze(y_los_val == 1)]
    y_test = y_test[np.squeeze(y_los_val == 1)]
elif args.trainOnly_nLOS == 'NLOS':
    y_train = y_train[np.squeeze(y_los_train == 0)]
    y_validation = y_validation[np.squeeze(y_los_val == 0)]
    y_test = y_test[np.squeeze(y_los_val == 0)]


###############################################################################
# Inputs
###############################################################################
Initial_labels_train = y_train         # these are same for all modalities
Initial_labels_val = y_validation

if 'coord' in args.input:
    #train
    X_coord_train = open_npz(args.data_folder+'coord_input/coord_train.npz','coordinates')
    #validation
    X_coord_validation = open_npz(args.val_data_folder+'coord_input/coord_validation.npz','coordinates')
    X_coord_test = open_npz(args.test_data_folder + 'coord_input/coord_test.npz', 'coordinates') # added for testing
    if args.trainOnly_nLOS == 'LOS':
        X_coord_train = X_coord_train[np.squeeze(y_los_train == 1), :]
        X_coord_validation = X_coord_validation[np.squeeze(y_los_val == 1), :]
        X_coord_test = X_coord_test[np.squeeze(y_los_test == 1), :]

    elif args.trainOnly_nLOS == 'NLOS':
        X_coord_train = X_coord_train[np.squeeze(y_los_train == 0), :]
        X_coord_validation = X_coord_validation[np.squeeze(y_los_val == 0), :]
        X_coord_test = X_coord_test[np.squeeze(y_los_test == 0), :]

    coord_train_input_shape = X_coord_train.shape

    if args.Aug:
        try:
            #train
            X_coord_train = open_npz(args.augmented_folder+'coord_input/coord_train.npz','train')
            y_train = open_npz(args.augmented_folder+'beam_output/beams_output_train.npz','train')

            #validation
            # X_coord_validation = open_npz(args.augmented_folder+'coord_input/coord_validation.npz','val')
            # y_validation = open_npz(args.augmented_folder+'beam_output/beams_output_validation.npz','val')
        except:
            print('****************Augment coordinates****************')
            y_train, X_coord_train = balance_data(Initial_labels_train,X_coord_train,0.001,(2,))
            # y_validation, X_coord_validation = balance_data(Initial_labels_val,X_coord_validation,0.001,(2,))
            save_npz(args.augmented_folder+'coord_input/','coord_train.npz',X_coord_train,'coord_validation.npz',X_coord_validation)
            # save_npz(args.augmented_folder+'beam_output/','beams_output_train.npz',y_train,'beams_output_validation.npz',y_validation)

    # NORMALIZE THE COORDINATES
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_coord_train = scaler.fit_transform(X_coord_train)
    X_coord_validation = scaler.fit_transform(X_coord_validation)
    X_coord_test = scaler.fit_transform(X_coord_test)

    ### For convolutional input
    X_coord_train = X_coord_train.reshape((X_coord_train.shape[0], X_coord_train.shape[1], 1))
    X_coord_validation = X_coord_validation.reshape((X_coord_validation.shape[0], X_coord_validation.shape[1], 1))
    X_coord_test = X_coord_test.reshape((X_coord_test.shape[0], X_coord_test.shape[1], 1))

    if args.new_split: X_coord_train, X_coord_validation, X_coord_test = split_with_new_ratio(X_coord_train, X_coord_validation,
                                                                           X_coord_test)

    #sklearn.preprocessing.normalize(X_coord_train, norm='l2', *, axis=1, copy=True, return_norm=False)

    print('coordinate shapes', X_coord_train.shape, X_coord_validation.shape, X_coord_test.shape)




if 'img' in args.input:
    ###############################################################################
    resizeFac = 20  # Resize Factor
    nCh = 1  # The number of channels of the image
    if args.image_feature_to_use == 'v1':
        folder = 'image_input'
    elif args.image_feature_to_use == 'v2':
        folder = 'image_v2_input'
    elif args.image_feature_to_use == 'custom':
        folder = 'image_custom_input'

    # train
    X_img_train = open_npz(args.data_folder + folder + '/img_input_train_' + str(resizeFac) + '.npz', 'inputs')
    # validation
    X_img_validation = open_npz(args.val_data_folder + folder + '/img_input_validation_' + str(resizeFac) + '.npz','inputs')
    # test
    X_img_test = open_npz(args.test_data_folder + folder + '/img_input_test_' + str(resizeFac) + '.npz', 'inputs')
    img_train_input_shape = X_img_train.shape
    print('shape first', X_img_test.shape)


    if args.Aug:
        try:
            print('****************Load augmented data****************')
            #train
            X_img_train = open_npz(args.augmented_folder+'image_input/img_input_train_20.npz','train')
            y_train = open_npz(args.augmented_folder+'beam_output/beams_output_train.npz','train')

            #validation
            X_img_validation = open_npz(args.augmented_folder+'image_input/img_input_validation_20.npz','val')
            y_validation = open_npz(args.augmented_folder+'beam_output/beams_output_validation.npz','val')

        except:
            print('****************Augment Image****************')
            y_train, X_img_train = balance_data(Initial_labels_train,X_img_train,0.001,(48, 81, 1))
            y_validation, X_img_validation = balance_data(Initial_labels_val,X_img_validation,0.001,(48, 81, 1))
            print('saving input')
            save_npz(args.augmented_folder+'image_input/','img_input_train_20.npz',X_img_train,'img_input_validation_20.npz',X_img_validation)
            print('saving Outputs')
            save_npz(args.augmented_folder+'beam_output/','beams_output_train.npz',y_train,'beams_output_validation.npz',y_validation)

    if args.image_feature_to_use == 'v1' or args.image_feature_to_use =='v2':
        print('********************Normalize image********************')
        X_img_train = X_img_train.astype('float32') / 255
        X_img_validation = X_img_validation.astype('float32') / 255
        X_img_test = X_img_test.astype('float32') / 255
    elif args.image_feature_to_use == 'custom':
        print('********************Reshape images for convolutional********************')
        X_img_train = X_img_train.reshape((X_img_train.shape[0], X_img_train.shape[1], X_img_train.shape[2],1))/3 # NORMALIZE
        X_img_validation = X_img_validation.reshape((X_img_validation.shape[0], X_img_validation.shape[1],X_img_validation.shape[2], 1))/3 # NORMALIZE
        X_img_test = X_img_test.reshape((X_img_test.shape[0], X_img_test.shape[1], X_img_test.shape[2],1))/3 # NORMALIZE

    if args.new_split: X_img_train, X_img_validation, X_img_test = split_with_new_ratio(X_img_train, X_img_validation, X_img_test)
    print('img shapes', X_img_train.shape, X_img_validation.shape, X_img_test.shape)

if 'lidar' in args.input:
    ###############################################################################
    #train
    print("Normalizing Lidar\n")
    X_lidar_train = open_npz(args.data_folder+'lidar_input/lidar_train.npz','input')/2.0
    #validation
    X_lidar_validation = open_npz(args.val_data_folder+'lidar_input/lidar_validation.npz','input')/2.0
    X_lidar_test = open_npz(args.test_data_folder + 'lidar_input/lidar_test.npz', 'input')/2.0
    lidar_train_input_shape = X_lidar_train.shape

    if args.Aug:
        try:
            X_lidar_train = open_npz(args.augmented_folder+'lidar_input/lidar_train.npz','train')
            y_train = open_npz(args.augmented_folder+'beam_output/beams_output_train.npz','train')

            #validation
            X_lidar_validation = open_npz(args.augmented_folder+'lidar_input/lidar_validation.npz','val')
            y_validation = open_npz(args.augmented_folder+'beam_output/beams_output_validation.npz','val')
        except:
            print('****************Augment Lidar****************')
            y_train, X_lidar_train = balance_data(Initial_labels_train,X_lidar_train,0.001,(20, 200, 10))
            y_validation, X_lidar_validation = balance_data(Initial_labels_val,X_lidar_validation,0.001,(20, 200, 10))
            save_npz(args.augmented_folder+'lidar_input/','lidar_train.npz',X_lidar_train,'lidar_validation.npz',X_lidar_validation)
            save_npz(args.augmented_folder+'beam_output/','beams_output_train.npz',y_train,'beams_output_validation.npz',y_validation)

    if args.new_split: X_lidar_train, X_lidar_validation, X_lidar_test = split_with_new_ratio(X_lidar_train, X_lidar_validation, X_lidar_test)

    ### NORMALIZE THE LIDAR
    # X_lidar_train = (X_lidar_train + 1)/3
    # X_lidar_validation = (X_lidar_validation + 1) / 3
    # X_lidar_test = (X_lidar_test + 1) / 3

    # X_lidar_train = X_lidar_train / 2.0
    # X_lidar_validation = X_lidar_validation / 2.0
    # X_lidar_test = X_lidar_test / 2.0

    print('lidar shapes', X_lidar_train.shape, X_lidar_validation.shape, X_lidar_test.shape)

##############################################################################
# Model configuration
##############################################################################
#multimodal
multimodal = False if len(args.input) == 1 else len(args.input)
fusion = False if len(args.input) == 1 else True
# print(fusion)

validationFraction = 0.2 #from 0 to 1

trainFraction = 0.6
trainValFraction = 0.8

modelHand = ModelHandler()
opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#opt = Nadam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
#opt = Adamax(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
#opt = Adagrad(lr=args.lr, epsilon=1e-07)
#opt = Adadelta(lr=args.lr, rho=0.95, epsilon=1e-07)
#opt= Adam(lr=args.lr)
file_name = 'acc'

if 'coord' in args.input:
    coord_model = modelHand.createArchitecture('coord_mlp',num_classes,coord_train_input_shape[1],'complete',args.strategy, fusion)
if 'img' in args.input:
    if args.image_feature_to_use == 'v1' or args.image_feature_to_use == 'v2':
        model_type = 'light_image_v1_v2'
    elif args.image_feature_to_use == 'custom':
        model_type = 'light_image_custom'

    if args.restore_models:
        img_model = load_model_structure(args.model_folder + 'image_' + args.image_feature_to_use + '_model' + '.json')
        img_model.load_weights(args.model_folder + 'best_weights.img_' + args.image_feature_to_use + '.h5',
                               by_name=True)
        # img_model.trainable = False
    else:
        img_model = modelHand.createArchitecture(model_type, num_classes, [img_train_input_shape[1], img_train_input_shape[2], 1], 'complete', args.strategy, fusion)
        # add_model('image_'+args.image_feature_to_use,img_model,args.model_folder)
        if not os.path.exists(args.model_folder + 'image_' + args.image_feature_to_use + '_model' + '.json'):
            add_model('image_' + args.image_feature_to_use, img_model, args.model_folder)
if 'lidar' in args.input:
    lidar_model = modelHand.createArchitecture('lidar_marcus',num_classes,[lidar_train_input_shape[1],lidar_train_input_shape[2],lidar_train_input_shape[3]],'complete',args.strategy, fusion)

# ADDED TO PERFORM ONE HOT ENCODING
#y_train = to_categorical(y_train, num_classes=num_classes)
#y_validation = to_categorical(y_validation, num_classes=num_classes)


cb_list = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='auto',restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=2, mode='auto',min_lr=args.lr * 0.01)]


if multimodal == 2:
        # LIDAR AND COORDINATE
    if 'coord' in args.input and 'lidar' in args.input:
        file_name = "lidar_coord"
        #combined_model = concatenate([lidar_model.output, coord_model.output])
        x_validation = [X_lidar_validation, X_coord_validation]
        x_train = [X_lidar_train, X_coord_train]
        x_test = [X_lidar_test, X_coord_test]

        merged_layer = tensorflow.keras.layers.multiply([lidar_model.output, coord_model.output]) # multiply works best
        channel = 16
        dropProb = 0.25  # 0.25

        # merged_layer = tensorflow.keras.layers.Conv2DTranspose(4 * channel, 9, strides=2,  # change stride to 3
        #                                                        # kernel size: (7,2) # 3*channel
        #                                                        padding='same',
        #                                                        kernel_initializer='he_normal',
        #                                                        activation='relu', name="trans0_coord_img")(merged_layer) # added

        merged_layer = tensorflow.keras.layers.Conv2DTranspose(3 * channel, 7, strides=2,
                                                               # kernel size: (7,2) # 3*channel
                                                               padding='same',
                                                               kernel_initializer='he_normal',
                                                               activation='relu', name="trans1_lidar_coord")(
            merged_layer)

        merged_layer = tensorflow.keras.layers.Conv2DTranspose(2 * channel, 5, strides=2,  # change stride to 2
                                                               # kernel size: (5,2) 2*channel
                                                               padding='same',
                                                               kernel_initializer='he_normal',
                                                               activation='relu', name="trans2_lidar_coord")(
            merged_layer)

        merged_layer = tensorflow.keras.layers.Conv2DTranspose(channel, 3, strides=1,  # kernel size: (3,1) stride = 1
                                                               padding='same',
                                                               kernel_initializer='he_normal',
                                                               activation='relu', name="trans3_lidar_coord")(
            merged_layer)

        # merged_layer = tensorflow.keras.layers.Conv2DTranspose(channel, 3, strides=1,  # kernel size: (3,1) stride = 1
        #                                                        padding='same',
        #                                                        kernel_initializer='he_normal',
        #                                                        activation='relu', name="trans4_coord_img")(merged_layer)

        merged_layer = Conv2D(channel, (5, 5), padding="SAME", activation='relu', name='conv1_lidar_coord')(
            merged_layer)
        merged_layer = Conv2D(channel, (5, 5), padding="SAME", activation='relu', name='conv2_lidar_coord')(
            merged_layer)
        merged_layer = MaxPooling2D(pool_size=(2, 2), name='maxpool1_lidar_coord')(merged_layer)
        merged_layer = Dropout(dropProb, name='lidar_coord_dropout1')(merged_layer)

        merged_layer = Conv2D(2 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_coord_conv4')(
            merged_layer)
        merged_layer = Conv2D(2 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_coord_conv5')(
            merged_layer)
        merged_layer = MaxPooling2D(pool_size=(2, 2), name='lidar_coord_maxpool2')(merged_layer)
        merged_layer = Dropout(dropProb, name='lidar_coord_dropout2')(merged_layer)

        ################################################################################
        # merged_layer = Conv2D(3 * channel, (3, 3), padding="SAME", activation='relu')(merged_layer)
        # merged_layer = Conv2D(3 * channel, (3, 3), padding="SAME", activation='relu')(merged_layer)
        # merged_layer = MaxPooling2D(pool_size=(2, 2))(merged_layer)
        # merged_layer = Dropout(dropProb)(merged_layer)
        ####################################################################################

        merged_layer = Conv2D(4 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_coord_conv6')(
            merged_layer)
        merged_layer = Conv2D(4 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_coord_conv7')(
            merged_layer)
        merged_layer = MaxPooling2D(pool_size=(1, 2), name='lidar_coord_maxpool3')(merged_layer)
        merged_layer = Dropout(dropProb, name='lidar_coord_dropout3')(merged_layer)

        ###################################################################
        # merged_layer = Conv2D(8 * channel, (3, 3), padding="SAME", activation='relu')(merged_layer)
        # merged_layer = Conv2D(8 * channel, (3, 3), padding="SAME", activation='relu')(merged_layer)
        # merged_layer = MaxPooling2D(pool_size=(1, 2))(merged_layer)
        # merged_layer = Dropout(dropProb)(merged_layer)

        ###########################################################################

        merged_layer = Conv2D(4 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_coord_conv8')(
            merged_layer)
        merged_layer = Conv2D(2 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_coord_conv9')(
            merged_layer)

        z = Flatten()(merged_layer)
        z = Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                      name="lidar_coord_dense1")(z)
        z = Dropout(0.2, name='lidar_coord_dropout4')(z)  # 0.25 is similar ... could try more values
        z = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                      name="lidar_coord_dense2")(z)
        z = Dropout(0.2, name='lidar_coord_dropout5')(z)
        z = Dense(num_classes, activation="softmax")(z)

        model = Model(inputs=[lidar_model.input, coord_model.input], outputs=z)

        # IMAGE AND COORDINATE
    elif 'coord' in args.input and 'img' in args.input:
        file_name = "img_coord"

        x_validation = [X_img_validation, X_coord_validation]
        x_test = [X_img_test, X_coord_test]
        x_train = [X_img_train, X_coord_train]

        #merged_layer = concatenate([img_model.output, coord_model.output])
        print("coord..", coord_model.output.shape)
        print("img..", img_model.output.shape)
        merged_layer = tensorflow.keras.layers.multiply([img_model.output, coord_model.output]) #multiply
        print("lidar reshaped..", merged_layer.shape)
        #merged_layer = tensorflow.keras.layers.subtract([coord_model.output, img_model.output])
        channel = 16
        dropProb = 0.25 # 0.25

        # merged_layer = tensorflow.keras.layers.Conv2DTranspose(4 * channel, 9, strides=2,  # change stride to 3
        #                                                        # kernel size: (7,2) # 3*channel
        #                                                        padding='same',
        #                                                        kernel_initializer='he_normal',
        #                                                        activation='relu', name="trans0_coord_img")(merged_layer) # added

        merged_layer = tensorflow.keras.layers.Conv2DTranspose(3*channel, 7, strides=2,
                                                               # kernel size: (7,2) # 3*channel
                                                               padding='same',
                                                               kernel_initializer='he_normal',
                                                               activation='relu', name="trans1_coord_img")(merged_layer)

        merged_layer = tensorflow.keras.layers.Conv2DTranspose(2 * channel, 5, strides=2, # change stride to 2
                                                               # kernel size: (5,2) 2*channel
                                                               padding='same',
                                                               kernel_initializer='he_normal',
                                                               activation='relu', name="trans2_coord_img")(merged_layer)

        merged_layer = tensorflow.keras.layers.Conv2DTranspose(channel, 3, strides=1,  # kernel size: (3,1) stride = 1
                                                               padding='same',
                                                               kernel_initializer='he_normal',
                                                               activation='relu', name="trans3_coord_img")(merged_layer)

        # merged_layer = tensorflow.keras.layers.Conv2DTranspose(channel, 3, strides=1,  # kernel size: (3,1) stride = 1
        #                                                        padding='same',
        #                                                        kernel_initializer='he_normal',
        #                                                        activation='relu', name="trans4_coord_img")(merged_layer)


        merged_layer = Conv2D(channel, (5, 5), padding="SAME", activation='relu', name='conv1_coord_img')(merged_layer)
        merged_layer = Conv2D(channel, (5, 5), padding="SAME", activation='relu', name='conv2_coord_img')(merged_layer)
        merged_layer = MaxPooling2D(pool_size=(2, 2), name='maxpool1_coord_img')(merged_layer)
        merged_layer = Dropout(dropProb, name='coord_img_dropout1')(merged_layer)

        merged_layer = Conv2D(2 * channel, (3, 3), padding="SAME", activation='relu', name='coord_img_conv4')(merged_layer)
        merged_layer = Conv2D(2 * channel, (3, 3), padding="SAME", activation='relu', name='coord_img_conv5')(merged_layer)
        merged_layer = MaxPooling2D(pool_size=(2, 2), name='coord_img_maxpool2')(merged_layer)
        merged_layer = Dropout(dropProb, name='coord_img_dropout2')(merged_layer)

        ################################################################################
        # merged_layer = Conv2D(3 * channel, (3, 3), padding="SAME", activation='relu')(merged_layer)
        # merged_layer = Conv2D(3 * channel, (3, 3), padding="SAME", activation='relu')(merged_layer)
        # merged_layer = MaxPooling2D(pool_size=(2, 2))(merged_layer)
        # merged_layer = Dropout(dropProb)(merged_layer)
        ####################################################################################

        merged_layer = Conv2D(4 * channel, (3, 3), padding="SAME", activation='relu', name='coord_img_conv6')(merged_layer)
        merged_layer = Conv2D(4 * channel, (3, 3), padding="SAME", activation='relu', name='coord_img_conv7')(merged_layer)
        merged_layer = MaxPooling2D(pool_size=(1, 2), name='coord_img_maxpool3')(merged_layer)
        merged_layer = Dropout(dropProb, name='coord_img_dropout3')(merged_layer)


        ###################################################################
        # merged_layer = Conv2D(8 * channel, (3, 3), padding="SAME", activation='relu')(merged_layer)
        # merged_layer = Conv2D(8 * channel, (3, 3), padding="SAME", activation='relu')(merged_layer)
        # merged_layer = MaxPooling2D(pool_size=(1, 2))(merged_layer)
        # merged_layer = Dropout(dropProb)(merged_layer)

        ###########################################################################


        merged_layer = Conv2D(4 * channel, (3, 3), padding="SAME", activation='relu', name='coord_img_conv8')(merged_layer)
        merged_layer = Conv2D(2 * channel, (3, 3), padding="SAME", activation='relu', name='coord_img_conv9')(merged_layer)

        z = Flatten()(merged_layer)
        z = Dense(num_classes * 4, activation="relu")(z)  # activity_regularizer=regularizers.l2(1e-4)
        z = Dropout(0.5)(z)
        #z = Dense(2 * num_classes, activation='relu', name='coord_img_dense1')(z)
        #z = Dense(2 * num_classes, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),name='coord_img_dense1')(z)
        #z = Dropout(0.5)(z)
        z = Dense(num_classes, activation="softmax")(z)

        model = Model(inputs=[img_model.input, coord_model.input], outputs=z)


    # IMPLEMENTED K-FOLD CROSS VALIDATION
    else:
        file_name = 'lidar_img'
        x_test = [X_lidar_test, X_img_test]
        combined_model = tensorflow.keras.layers.subtract([lidar_model.output, img_model.output])

        # merged_layer = Conv2DTranspose(num_classes, 3, strides=2,
        #                                padding='same',
        #                                kernel_initializer='he_normal',
        #                                activation='relu', name="trans1_radar_lb_rf")(combined_model)


        z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(combined_model)  # USE THIS AND THE NEXT PART OF CODE OF mlp IMPLEMENTATION

        z = Dropout(0.5)(z)
        z = Dense(num_classes, activation="softmax")(z)

        model = Model(inputs=[lidar_model.input, img_model.input], outputs=z)
        # model.compile(loss=categorical_crossentropy,optimizer=opt,
        #               metrics=[metrics.categorical_accuracy, top_2_accuracy,
        #                        metrics.top_k_categorical_accuracy, top_10_accuracy,
        #                        top_50_accuracy, precision_m, recall_m, f1_m])
        # model.summary()


        # CONCATINATING TRAIN AND VALIDATION AND PERFORM K FOLD CROSS VALIDATION
        k_folds = args.k_fold
        cvscores = []
        testscores = []
        # K fold cross Validation
        if args.k_fold >0:
            while k_folds >0:
                print("Starting ", k_folds, "th Fold........")
                x_lidar_data = np.concatenate([X_lidar_train, X_lidar_validation], axis=0)
                x_img_data = np.concatenate([X_img_train, X_img_validation], axis=0)
                y_data = np.concatenate([y_train, y_validation], axis=0)
                x_lidar_data, x_img_data, y_data = sklearn.utils.shuffle(x_lidar_data, x_img_data, y_data)
                trainLength = int(x_lidar_data.shape[0]*(1-validationFraction))
                print(trainLength)
                X_lidar_train, X_lidar_validation = x_lidar_data[0:trainLength, :], x_lidar_data[trainLength:x_lidar_data.shape[0], :]
                X_img_train, X_img_validation = x_img_data[0:trainLength, :], x_img_data[trainLength:x_img_data.shape[0], :]
                y_train, y_validation = y_data[0:trainLength, :], y_data[trainLength:y_data.shape[0], :]

                # TRAINING THE MODEL
                x_validation = [X_lidar_validation, X_img_validation]
                x_train = [X_lidar_train, X_img_train]
                hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=cb_list)

                # PRINTING TEST RESULTS
                print('***************Validating the model************')
                scores = model.evaluate(x_validation, y_validation)
                print(model.metrics_names, scores)
                # print("%s: %.2f%%" % (model.metrics_names[4], scores[4] * 100))
                cvscores.append(scores[4] * 100)
                # K.clear_session()
                k_folds = k_folds - 1

                print('***************Testing the model************')
                scores = model.evaluate(x_test, y_test)
                print(model.metrics_names, scores)
                testscores.append(scores[4] * 100)

        else:
            # TRAINING THE MODEL
            x_validation = [X_lidar_validation, X_img_validation]
            x_train = [X_lidar_train,X_img_train]

# LIDAR AND COORDINATE AND IMAGE
elif multimodal == 3:

    file_name = "lidar_img_coord"
    # combined_model = concatenate([lidar_model.output, img_model.output, coord_model.output], axis=1)
    x_test = [X_lidar_test, X_img_test, X_coord_test]


    print("lidar..", lidar_model.output.shape)
    print("coord..", coord_model.output.shape)
    print("img..", img_model.output.shape)
    # np.savetxt('lidar_data.txt', lidar_model.output)
    # np.savetxt('img_model.txt', img_model.output)
    # np.savetxt('coord_model.txt', coord_model.output)

    combined_model = merged_layer = tensorflow.keras.layers.multiply([lidar_model.output, coord_model.output]) #subtract better for balanced data

    print("merged layer.", merged_layer)
    if args.image_feature_to_use == "custom":
        img_reshaped = Reshape((img_model.output.shape[2], img_model.output.shape[1], img_model.output.shape[3]))(img_model.output)
        print("before cropping..", img_reshaped.shape)
        crop = (img_reshaped.shape[1] - lidar_model.output.shape[1])//2
        print("the crop size: ", crop)
        img_reshaped = tensorflow.keras.layers.Cropping2D(cropping=((crop, crop), (0, 0)))(img_reshaped)
        print("after cropping..", img_reshaped.shape)
    else:
        #print("before upsample..", img_reshaped.shape)
        img_reshaped = tensorflow.keras.layers.UpSampling2D(size=5)(img_model.output)
        merged_layer = tensorflow.keras.layers.UpSampling2D(size=2)(merged_layer)
        print("after upsample..", img_reshaped.shape, merged_layer.shape)
        crop = (img_reshaped.shape[1] - merged_layer.shape[1]) // 2
        img_reshaped = tensorflow.keras.layers.Cropping2D(cropping=((crop, crop), (0, 0)))(img_reshaped)

    merged_layer = tensorflow.keras.layers.subtract([merged_layer, img_reshaped])
    channel = 16
    dropProb = 0.25  # 0.25

    # merged_layer = tensorflow.keras.layers.Conv2DTranspose(4 * channel, 9, strides=2,  # change stride to 3
    #                                                        # kernel size: (7,2) # 3*channel
    #                                                        padding='same',
    #                                                        kernel_initializer='he_normal',
    #                                                        activation='relu', name="trans0_coord_img")(merged_layer) # added

    merged_layer = tensorflow.keras.layers.Conv2DTranspose(3 * channel, 7, strides=2,
                                                           # kernel size: (7,2) # 3*channel
                                                           padding='same',
                                                           kernel_initializer='he_normal',
                                                           activation='relu', name="trans1_lidar_img_coord")(merged_layer)

    merged_layer = tensorflow.keras.layers.Conv2DTranspose(2 * channel, 5, strides=2,  # change stride to 2
                                                           # kernel size: (5,2) 2*channel
                                                           padding='same',
                                                           kernel_initializer='he_normal',
                                                           activation='relu', name="trans2_lidar_img_coord")(merged_layer)

    merged_layer = tensorflow.keras.layers.Conv2DTranspose(channel, 3, strides=2,  # kernel size: (3,1) stride = 1
                                                           padding='same',
                                                           kernel_initializer='he_normal',
                                                           activation='relu', name="trans3_lidar_img_coord")(merged_layer)

    merged_layer = tensorflow.keras.layers.Conv2DTranspose(channel, 3, strides=1,  # kernel size: (3,1) stride = 1
                                                           padding='same',
                                                           kernel_initializer='he_normal',
                                                           activation='relu', name="trans4_coord_img")(merged_layer)


    #####################################################################################################################
    merged_layer = Conv2D(channel, (5, 5), padding="SAME", activation='relu', name='conv1_lidar_img_coord')(merged_layer)
    merged_layer = Conv2D(channel, (5, 5), padding="SAME", activation='relu', name='conv2_lidar_img_coord')(merged_layer)
    merged_layer = MaxPooling2D(pool_size=(2, 2), name='maxpool1_coord_img')(merged_layer)
    merged_layer = Dropout(dropProb, name='lidar_img_coord_dropout1')(merged_layer)

    merged_layer = Conv2D(2 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv4')(merged_layer)
    merged_layer = Conv2D(2 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv5')(merged_layer)
    merged_layer = MaxPooling2D(pool_size=(2, 2), name='lidar_img_coord_maxpool2')(merged_layer)
    merged_layer = Dropout(dropProb, name='lidar_img_coord_dropout2')(merged_layer)


    merged_layer = Conv2D(4 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv6')(merged_layer)
    merged_layer = Conv2D(4 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv7')(merged_layer)
    merged_layer = MaxPooling2D(pool_size=(1, 2), name='lidar_img_coord_maxpool3')(merged_layer)
    merged_layer = Dropout(dropProb, name='lidar_img_coord_dropout3')(merged_layer)


    merged_layer = Conv2D(4 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv8')(merged_layer)
    merged_layer = Conv2D(2 * channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv9')(merged_layer)
    #
    # z = Flatten()(merged_layer)
    # z = Dense(num_classes * 4, activation="relu")(z)  # activity_regularizer=regularizers.l2(1e-4)
    # z = Dropout(0.5)(z)
    ######################################################################################################################
    # a = merged_layer = Conv2D(channel, kernel_size=(3, 3),
    #                    activation='relu', padding="SAME", name='lidar_img_coord_conv1')(merged_layer)
    # merged_layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv2')(merged_layer)
    # merged_layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv3')(merged_layer)  # + a
    # merged_layer = Add(name='lidar_img_coord_add1')([merged_layer, a])  # DR
    # merged_layer = MaxPooling2D(pool_size=(2, 2), name='lidar_img_coord_maxpool1')(merged_layer)
    # b = merged_layer = Dropout(dropProb, name='lidar_img_coord_dropout1')(merged_layer)
    #
    # merged_layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv4')(merged_layer)
    # merged_layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv5')(merged_layer)  # + b
    # merged_layer = Add(name='lidar_img_coord_add2')([merged_layer, b])  # DR
    # merged_layer = MaxPooling2D(pool_size=(2, 2), name='lidar_img_coord_maxpool2')(merged_layer)
    # c = merged_layer = Dropout(dropProb, name='lidar_img_coord_dropout2')(merged_layer)
    #
    # merged_layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv6')(merged_layer)
    # merged_layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv7')(merged_layer)  # + c
    # merged_layer = Add(name='lidar_img_coord_add3')([merged_layer, c])  # DR
    # merged_layer = MaxPooling2D(pool_size=(1, 2), name='lidar_img_coord_maxpool3')(merged_layer)
    # d = merged_layer = Dropout(dropProb, name='lidar_img_coord_dropout3')(merged_layer)
    #
    # # if add this layer, need 35 epochs to converge
    # # layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    # # layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) + d
    # # layer = MaxPooling2D(pool_size=(1, 2))(layer)
    # # e = layer = Dropout(dropProb)(layer)
    #
    # out = merged_layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv8')(merged_layer)
    # merged_layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_img_coord_conv9')(merged_layer)  # + d
    # # layer = Add(name='lidar_add4')([layer, d])  # DR
    ###################################################################################################################################

    merged_layer = Flatten(name='lidar_img_coord_flatten')(merged_layer)
    z = Dense(num_classes * 4, activation="relu",  name="lidar_img_coord_dense1")(merged_layer)  # activity_regularizer=regularizers.l2(1e-4)
    z = Dropout(0.5)(z)
    z = Dense(num_classes*2, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name="lidar_img_coord_dense2")(z)
    z = Dropout(0.2, name='lidar_img_coord_dropout4')(z)  # 0.25 is similar ... could try more values
    z = Dense(num_classes, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name="lidar_img_coord_dense3")(z)
    z = Dropout(0.2, name='lidar_img_coord_dropout5')(z)  #


    z = Dense(num_classes, activation="softmax")(z)

    model = Model(inputs=[lidar_model.input, img_model.input, coord_model.input], outputs=z)
    viz_model = Model(inputs=[lidar_model.input, img_model.input, coord_model.input], outputs=combined_model)  ## very big

    #model.summary()


    # CONCATINATING TRAIN AND VALIDATION AND PERFORM K FOLD CROSS VALIDATION
    k_folds = args.k_fold
    cvscores = []
    testscores = []
    # K fold cross Validation
    if args.k_fold > 0:
        while k_folds > 0:
            print("Starting ", k_folds, "th Fold........")
            # CONCATINATING TRAIN AND VALIDATION AND PERFORM K FOLD CROSS VALIDATION
            x_lidar_data = np.concatenate([X_lidar_train, X_lidar_validation], axis=0)
            x_img_data = np.concatenate([X_img_train, X_img_validation], axis=0)
            x_coord_data = np.concatenate([X_coord_train, X_coord_validation], axis=0)
            y_data = np.concatenate([y_train, y_validation], axis=0)
            x_lidar_data, x_img_data, x_coord_data, y_data = sklearn.utils.shuffle(x_lidar_data, x_img_data, x_coord_data, y_data)
            trainLength = int(x_lidar_data.shape[0] * (1 - validationFraction))
            print(trainLength)
            X_lidar_train, X_lidar_validation = x_lidar_data[0:trainLength, :], x_lidar_data[trainLength:x_lidar_data.shape[0],:]
            X_img_train, X_img_validation = x_img_data[0:trainLength, :], x_img_data[trainLength:x_img_data.shape[0], :]
            X_coord_train, X_coord_validation = x_coord_data[0:trainLength, :], x_coord_data[trainLength:x_coord_data.shape[0], :]
            y_train, y_validation = y_data[0:trainLength, :], y_data[trainLength:y_data.shape[0], :]

            # TRAINING THE MODEL
            x_validation = [X_lidar_validation, X_img_validation, X_coord_validation]
            x_train = [X_lidar_train, X_img_train, X_coord_train]
            hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=cb_list)

            # PRINTING TEST RESULTS
            print('***************Validating the model************')
            scores = model.evaluate(x_validation, y_validation)
            print(model.metrics_names, scores)
            # print("%s: %.2f%%" % (model.metrics_names[4], scores[4] * 100))
            cvscores.append(scores[4] * 100)
            #K.clear_session()
            k_folds = k_folds - 1

            print('***************Testing the model************')
            scores = model.evaluate(x_test, y_test)
            print(model.metrics_names, scores)
            # print("%s: %.2f%%" % (model.metrics_names[4], scores[4] * 100))
            testscores.append(scores[4] * 100)
    else:
        # TRAINING THE MODEL
        x_validation = [X_lidar_validation, X_img_validation, X_coord_validation]
        x_train = [X_lidar_train,X_img_train,X_coord_train]
else:
    if 'coord' in args.input:
        file_name = 'coord'
        if args.strategy == 'reg':
            model = coord_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_2_accuracy,top_10_accuracy,top_50_accuracy])
            model.summary()

            hist = model.fit(X_coord_train,y_train,
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses', hist.history['loss'])

            print('*****************Testing***********************')
            scores = model.evaluate(X_coord_validation, y_validation)
            print(model.metrics_names,scores)

            print('*****************Manual measuring, its same as using top_2_accuracy***********************')
            preds = model.predict(X_coord_validation)
            top_10_coord = meaure_topk_for_regression(y_validation,preds,10)
            print('top 10 accuracy for coord is:',top_10_coord)

        if args.strategy == 'one_hot':
            model = coord_model
            x_validation = X_coord_validation
            x_test = X_coord_test
            x_train = X_coord_train

        if args.load_los and args.trainOnly_nLOS == 'none':
            acc_los, acc_nlos = los_accuracy(model, X_coord_validation, y_validation, y_los_val, 10)

    elif 'img' in args.input:
        file_name = 'img'
        if args.strategy == 'reg':
            model = img_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_2_accuracy,top_10_accuracy,top_50_accuracy])
            model.summary()

            hist = model.fit(X_img_train,y_train,
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses', hist.history['loss'])

            print('*****************Testing***********************')
            scores = model.evaluate(X_img_validation, y_validation)
            print(model.metrics_names,scores)


            print('*****************Manual measuring, its same as using top_2_accuracy***********************')
            preds = model.predict(X_img_validation)
            top_10_img = meaure_topk_for_regression(y_validation,preds,10)
            print('top 10 accuracy for image is:',top_10_img)

        if args.strategy == 'one_hot':
            model = img_model


            x_validation = X_img_validation
            x_test= X_img_test
            x_train= X_img_train

        if args.load_los and args.trainOnly_nLOS == 'none':
            acc_los, acc_nlos = los_accuracy(model, X_coord_validation, y_validation, y_los_val, 10)


    else: #LIDAR
        file_name = "lidar"
        x_validation = X_lidar_validation
        x_train = X_lidar_train
        x_test = X_lidar_test
        if args.strategy == 'reg':
            model = lidar_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_2_accuracy,top_10_accuracy,top_50_accuracy])
            model.summary()

            hist = model.fit(x_train,y_train,
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses', hist.history['loss'])

            print('*****************Validating***********************')
            scores = model.evaluate(x_validation, y_validation)
            print(model.metrics_names,scores)

            print('*****************Manual measuring, its same as using top_2_accuracy***********************')
            preds = model.predict(X_lidar_validation)
            top_10_lidar = meaure_topk_for_regression(y_validation,preds,10)
            print('top 10 accuracy for lidar is:',top_10_lidar)


        if args.strategy == 'one_hot':
            model = lidar_model
            x_validation = X_lidar_validation
            x_test = X_lidar_test
            x_train = X_lidar_train

        if args.load_los and args.trainOnly_nLOS == 'none':
            acc_los, acc_nlos = los_accuracy(model, X_coord_validation, y_validation, y_los_val, 10)


# WHEN k-FOL CROSS VALIDATION - THEY ARE TRAINED EARLIER
if args.k_fold > 0:
    print("Validation Scores")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print(cvscores)

    print("Test Scores")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(testscores), np.std(testscores)))
    print(testscores)

#FINAL TRAINING
else:
    # model.compile(loss=categorical_crossentropy,
    #               optimizer=opt,
    #               metrics=[metrics.categorical_accuracy, top_2_accuracy,
    #                        metrics.top_k_categorical_accuracy, top_10_accuracy, top_15_accuracy, top_30_accuracy,
    #                        top_50_accuracy, precision_m, recall_m, f1_m])

    # model.compile(loss=categorical_crossentropy,
    #               optimizer=opt,
    #               metrics=[metrics.categorical_accuracy, top_2_accuracy,
    #                        metrics.top_k_categorical_accuracy, top_10_accuracy, top_30_accuracy,
    #                        top_50_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]) # USING KERAS'S PRECION AND RECALL METRICS

    # model.compile(loss=categorical_crossentropy,
    #               optimizer=opt,
    #               metrics=[metrics.categorical_accuracy, top_2_accuracy,top_3_accuracy, top_4_accuracy,
    #                        metrics.top_k_categorical_accuracy, top_6_accuracy, top_7_accuracy, top_8_accuracy, top_9_accuracy,
    #                        top_10_accuracy, top_11_accuracy, top_12_accuracy, top_13_accuracy, top_14_accuracy, top_15_accuracy,
    #                        tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=[metrics.categorical_accuracy, top_2_accuracy, top_3_accuracy, top_4_accuracy,
                           metrics.top_k_categorical_accuracy, top_8_accuracy,
                           top_10_accuracy, top_12_accuracy, top_15_accuracy, top_20_accuracy, top_30_accuracy, top_50_accuracy,
                           tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    #model.summary()

    hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,
                     batch_size=args.bs, callbacks=cb_list, shuffle=args.shuffle)

    # hist = model.fit(x_train, y_train, validation_split = 0.2, epochs=args.epochs,
    #                  batch_size=args.bs, callbacks=cb_list, shuffle=args.shuffle) # for the setting of  Train and validate on s009 and test on s008: (not using s008 validation data)

    print('***************Validating model************')
    scores = model.evaluate(x_validation, y_validation)
    print(model.metrics_names, scores)

    print('***************Testing the model************')
    scores = model.evaluate(x_test, y_test)
    print(model.metrics_names, scores)

    # CALCUALTING THE WEIGHTED F1-SCORE
    y_val_hat = model.predict(x_validation, batch_size=args.bs)
    y_test_hat = model.predict(x_test, batch_size=args.bs)

    import pickle
    ytest_pickle = open("ytest.pkl", 'wb')
    pickle.dump(y_test, ytest_pickle)
    ytest_pickle.close()

    ytest_hat_pickle = open("ytest_hat.pkl", 'wb')
    pickle.dump(y_test_hat, ytest_hat_pickle)
    ytest_hat_pickle.close()

    from sklearn.metrics import f1_score

    print("The Weighted F1 score in validation set: ", f1_score(y_validation, np.around(y_val_hat), average='weighted'))
    print("The Weighted F1 score in test set: ", f1_score(y_test, np.around(y_test_hat), average='weighted'))
    # END CALCUALTING THE WEIGHTED F1-SCORE

    if args.load_los and args.trainOnly_nLOS == 'none':
        for k in range(20):
            acc_los, acc_nlos = los_accuracy(model, x_train, y_train, y_los_train, k)
            acc_los, acc_nlos = los_accuracy(model, x_validation, y_validation, y_los_val, k)
            acc_los, acc_nlos = los_accuracy(model, x_test, y_test, y_los_test, k)


    # FOR PRINTING OUT THE FUSION LAYER FEATURES
    # check_out = viz_model.predict(x_test)
    # print(check_out[0])

    #np.savetxt(file_name+'_feature_data.txt', check_out[0])

    # print("**********************Calculating top-k precision and recall *****************")
    # test_Y_hat = model.predict(x_test)
    # for k in([1, 2, 5, 10, 30, 50]):
    #     print("\nThe top-", k, " percision is. ", top_k_precision(test_Y_hat, k))
    #     print("The top-", k, " recall is. ", top_k_recall(test_Y_hat, k))

K.clear_session()

### SET PLOTTING PARAMETERS #########
params = {'legend.fontsize': 'small',
         'axes.labelsize': 'x-large',
         'axes.titlesize': 'x-large',
         'xtick.labelsize': 'x-large',
         'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)


#Show Accuracy Curves
fig = plt.figure()
plt.title('Training Performance')
plt.plot(hist.epoch, hist.history['categorical_accuracy'], label='Top 1 Training Accuracy', linewidth=2.0, c='b')
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'], label='Top 1 Validation Accuracy', linewidth=2.0, c='b',linestyle='--')
plt.plot(hist.epoch, hist.history['top_2_accuracy'], label='Top 2 Training Accuracy', linewidth=2.0, c='r')
plt.plot(hist.epoch, hist.history['val_top_2_accuracy'], label='Top 2 Validation Accuracy', linewidth=2.0, c='r',linestyle='--')
plt.plot(hist.epoch, hist.history['top_10_accuracy'], label='Top 10 Training Accuracy', linewidth=2.0, c='m')
plt.plot(hist.epoch, hist.history['val_top_10_accuracy'], label='Top 10 Validation Accuracy', linewidth=2.0, c='m',linestyle='--')
# plt.plot(hist.epoch, hist.history['top_50_accuracy'], label='Top 50 Training Accuracy', linewidth=2.0, c='g')
# plt.plot(hist.epoch, hist.history['val_top_50_accuracy'], label='Top 50 Validation Accuracy', linewidth=2.0, c='g',linestyle='--')

plt.ylabel('Accuracy(%)')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
fig.savefig(file_name+'_acc.png')  # save the figure to file
plt.close(fig)


# plt.show()
np.set_printoptions(threshold=np.inf)
classes = []
for i in range(num_classes):
	classes.append(str(i))

# Plot confusion matrix
#y_val_hat = np.around(model.predict(x_validation, batch_size=args.bs))

#print(y_val_hat)

# PRINTING THE TRUE AND PREDICTED LABELS ACCURACY
# TRUE VALIDATION LEBELS
num_zero = 0
num_nonzero = 0
yval_dict = {}
yval_dict = dict.fromkeys(range(0, num_classes), 0)

for i in range(0, len(y_validation)):
    if 1 in list(y_validation[i, :]):
        j = list(y_validation[i, :]).index(1)
        num_nonzero = num_nonzero + 1
    else:
        j = 0
        num_zero = num_zero + 1
    k = int(np.argmax(y_validation[i, :]))
    yval_dict[k] = yval_dict[k] + 1

# print("Num of zeros in true labels", num_zero, " and non zeros ", num_nonzero)
# print(yval_dict.values())
# print(yval_dict.items())
fig = plt.figure()
plt.bar(yval_dict.keys(), yval_dict.values(), color='green')
plt.xlabel("Labels")
plt.ylabel("Frequncy")
#plt.title("Energy output from various fuel sources")

#plt.xticks(x_pos, x)
plt.tight_layout()
fig.savefig(file_name+'_val.png')  # save the figure to file
plt.close(fig)

# PREDICTED VALIDATION LABELS
num_zero = 0
num_nonzero = 0
yval_hat_dict = {}
yval_hat_dict = dict.fromkeys(range(0, num_classes), 0)
#yval_dict.update()
#print(yval_hat_dict)
for i in range(0, len(y_val_hat)):
    if 1 in list(y_val_hat[i, :]):
        j = list(y_val_hat[i, :]).index(1)
        num_nonzero = num_nonzero + 1
    else:
        j = 0
        num_zero = num_zero + 1
    k = int(np.argmax(y_val_hat[i, :]))
    yval_hat_dict[k] = yval_hat_dict[k] + 1

# print("Num of zeros in predicted labels ", num_zero, " and non zeros ", num_nonzero)
# print(yval_hat_dict.values())
# print(yval_hat_dict.items())

fig = plt.figure()
plt.bar(yval_hat_dict.keys(), yval_hat_dict.values(), color='green')
plt.xlabel("Labels")
plt.ylabel("Frequncy")
#plt.title("Energy output from various fuel sources")

#plt.xticks(x_pos, x)
plt.tight_layout()
fig.savefig(file_name+'_val_pred.png')  # save the figure to file
plt.close(fig)


# TRUE TEST LEBELS
num_zero = 0
num_nonzero = 0
ytest_dict = {}
ytest_dict = dict.fromkeys(range(0, num_classes), 0)

for i in range(0, len(y_test)):
    if 1 in list(y_test[i, :]):
        j = list(y_test[i, :]).index(1)
        num_nonzero = num_nonzero + 1
    else:
        j = 0
        num_zero = num_zero + 1
    k = int(np.argmax(y_test[i, :]))
    ytest_dict[k] = ytest_dict[k] + 1

# print("Num of zeros in true labels", num_zero, " and non zeros ", num_nonzero)
# print(yval_dict.values())
# print(yval_dict.items())
fig = plt.figure()
plt.bar(ytest_dict.keys(), ytest_dict.values(), color='green')
plt.xlabel("Labels")
plt.ylabel("Frequncy")
#plt.title("Energy output from various fuel sources")

#plt.xticks(x_pos, x)
plt.tight_layout()
fig.savefig(file_name+'_test.png')  # save the figure to file
plt.close(fig)


# PREDICTED TEST LABELS
num_zero = 0
num_nonzero = 0
ytest_hat_dict = {}
ytest_hat_dict = dict.fromkeys(range(0, num_classes), 0)
#yval_dict.update()
#print(yval_hat_dict)
for i in range(0, len(y_test_hat)):
    if 1 in list(y_test_hat[i, :]):
        j = list(y_test_hat[i, :]).index(1)
        num_nonzero = num_nonzero + 1
    else:
        j = 0
        num_zero = num_zero + 1
    k = int(np.argmax(y_test_hat[i, :]))
    ytest_hat_dict[k] = ytest_hat_dict[k] + 1

# print("Num of zeros in predicted labels ", num_zero, " and non zeros ", num_nonzero)
# print(ytest_hat_dict.values())
# print(ytest_hat_dict.items())

fig = plt.figure()
plt.bar(ytest_hat_dict.keys(), ytest_hat_dict.values(), color='green')
plt.xlabel("Labels")
plt.ylabel("Frequncy")
#plt.title("Energy output from various fuel sources")

#plt.xticks(x_pos, x)
plt.tight_layout()
fig.savefig(file_name+'_test_pred.png')  # save the figure to file
plt.close(fig)


