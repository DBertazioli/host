import numpy as np
import keras
from keras import backend as K
import matplotlib 
from matplotlib import pyplot as plt
#METRICS
def recall(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = tp / (possible_positives + K.epsilon())
        return recall
def precision(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = tp / (predicted_positives + K.epsilon())
        return precision
    
def f1(y_true, y_pred):
  
    def recall(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = tp / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = tp / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#LOSSES
def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def tot_loss(y_true, y_pred):
  #return (f1_loss(y_true, y_pred)*keras.losses.binary_crossentropy(y_true, y_pred))/(f1_loss(y_true, y_pred)+keras.losses.binary_crossentropy(y_true, y_pred))
  return (f1_loss(y_true, y_pred)+keras.losses.binary_crossentropy(y_true, y_pred))/2


#CALLBACKS
from IPython.display import clear_output

class Better_verbose(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        
        if len(self.losses)%10==0:
          print(logs)
          
class TrainingPlot(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.f1=[]
        self.val_f1=[]
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.f1.append(logs.get('f1'))
        self.val_f1.append(logs.get('val_f1'))
        
        if len(self.losses)%10==0:
            
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            #%matplotlib inline
            plt.style.use("seaborn")
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.plot(N, self.f1, label = "f1")
            plt.plot(N, self.val_f1, label = "val_f1")
            plt.ylim(0,1)
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()

