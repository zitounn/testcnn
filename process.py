import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from  numpy import genfromtxt


"""
Load xray images, lung segmentation and dependent variables(xcord,ycord,side
and label) shuffle, with sklearn, and split into train, valid and test set. 

Create a tf dataset for the three different datasets, batch, shuffle and 
use the repeat function on the datasets. Create a generic iterator with 
the right parameters and a get.next() iterator. 

Initlize the iterator on the datasets. 

Create the computational graph and use the correct get.next() element
as the input. (start with very basic NN and see if you can get low biase) 

use tf.session and use the right iterator for training, and validation. 

"""

def data_processing(csv_file):

    """
    Function that reads the csv function, shuffles it and shapes it into
    a matrix suitable for tensorflow dataframe. Return dataset with
    filename in the postiion of the images.
    """
    pd_frame = pd.DataFrame.from_csv(csv_file)

    pd_frame_shuffled = shuffle(pd_frame)
    xcord = pd_frame_shuffled['x-cord'].tolist()
    ycord = pd_frame_shuffled['y-cord'].tolist()
    side = pd_frame_shuffled['side'].tolist()
    label  = pd_frame_shuffled['label'].tolist()
    file_name = pd_frame_shuffled['FileName'].tolist()


    tf_dataset = tf.data.Dataset.from_tensor_slices((file_name,ycord, xcord, side, label))



    return (tf_dataset)
 

def _parser_function(filename, xcord, ycord, side, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    return image_decoded, xcord, ycord, side, label



tf_dataset = data_processing('out.csv')

print(tf_dataset.output_shapes)
print(tf_dataset.output_types)




#iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
#                                                       train_dataset.output_shapes)
#next_element = iterator.get_next()









