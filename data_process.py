import pandas as pd
import os
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np




def sorted_files_cord():
    """
    Function that returns a sorted list of the names of the jpg files
    and the correstponding labels.  


    return:
            sorted_list: string  list of names of the xray files
            label: int list of 0 and 1. 1 for xray with nodules. 

    """

    cwd = os.getcwd()

    list = []
    for file in os.listdir(cwd+'/train'):
            if file.endswith('mask.jpg'):
                list.append(file[0:8])


    sorted_list = sorted(list, key= lambda x: int(x[5:8]))

    label = np.zeros(len(list))
    counter = 0
    for file in sorted_list:
            
            if file[2:5] == 'CLN':
                    label[counter] = 1  
            counter = counter +1
    return sorted_list, label
        
def preprocess_all(file_list, labels):
        """
        Take the list of all the images, read them in
        one by one and ouput a cvs file with the coordinates
        of the lower left corner, the side length, the label and  name
        of the file.
        
        return :
                : numpy matrix(5x234) with header | x-cord | y-cord | side | label | filename

        """     
        cwd = os.getcwd()

#       if not os.path.isdir(cwd+'/test_folder'):
                
#               os.makedirs(cwd+'/test_folder')
                                                        
#       else: return
        
        count = 0 
        csv_matrix = np.zeros((5,len(file_list)))
        x = np.zeros((234), dtype=[('x-cord','f4'), ('y-cord', 'f4'), ('side','f4'),('label','i4'),('FileName', 'U12')])
        for seg_image in file_list:
                                                
                path = os.path.abspath('train/'+seg_image+'_mask.jpg')
                img = misc.imread(path)


                if labels[count] == 1:
                    center, rang = center_side(img)
                    cord, side = corner_side(center,rang)

                    x[count] = (cord[0], cord[1], side, labels[count], seg_image)
                    
                else:
                    
                    x[count] = (-1,-1,-1, labels[count], seg_image)
                
                count = count +1
                print(count)

        return x 



#Function to calculate center and side of square.
def center_side(image):
        
        """
        Function takes in images and returns the coordinates
        of the center and side length of the square.
        
        input: numpy 2D matrix of an image with a disk segmentation

        return: a list with a list of integers with the
        centers of x and y as the first element and the length of the
        square as the second element. 
        
        """
        
        nonzero_list = np.nonzero(image)
        
        list1 = nonzero_list[0]
        list2 = nonzero_list[1]         

        sorted1 = np.sort(list1)
        min1 = sorted1[0]
        max1 = sorted1[-1]
        
        sorted2 = np.sort(list2)
        min2 = sorted2[0]
        max2 = sorted2[-1]
        
        center1, range1 = min1 + (max1-min1)/2,max1-min1
        
        
        center2, range2 = min2 + (max2-min2)/2,max2-min2
        
        center, range = [center1, center2],[range1, range2] 
        
        return center, range
 
def corner_side(center,side):
    
        """
        takes in image and return the right lower side coordinates
        and the average side length for a cube

        reurn:
                [min_max, min_y]: x and why coorindates
                avg_side: average side length



        """

        
        avg_side = (side[0]+side[1])/2
        center_x = center[0]    
        center_y = center[1]
        min_x = center_x - avg_side
        min_y = center_y - avg_side

        return [min_x, min_y], avg_side




#Creates the border of a square aroun a segmentation.
def create_border(img,center,side):
        """
        Takes in the original image the center and side parameters
        and returns a numpy matrix with a border around the 
        segmentation

        Input:  img: The original image with a segmentation
                center: the center position of the square
                side: The length of the side of the square
        
        return: returns a numpy matrix of the square.

        """     
        avg_side = (side[0]+side[1])/2
        center_x = center[0]
        center_y = center[1]
        
        max_x = center_x + avg_side
        max_y = center_y + avg_side
        min_x = center_x - avg_side
        min_y = center_y - avg_side

        border_img = np.zeros(img.shape)
        
        for x_cord in range(1,img.shape[0]):
                for y_cord in range(1,img.shape[1]):

                        if x_cord > min_x and x_cord < max_x and y_cord > min_y and y_cord < max_y:
                                
                                if x_cord < min_x+3 or x_cord > max_x-3 or y_cord < min_y+3 or y_cord > max_y-3:        
                                        border_img[x_cord, y_cord] = 1  
        return border_img


file_list, labels = sorted_files_cord()


csv_matrix = preprocess_all(file_list, labels)

panda_label = pd.DataFrame(csv_matrix)

panda_label.to_csv('out.csv')

#Test border plotting
"""
path = os.path.abspath('train/JPCLN030_mask.jpg')

img = misc.imread(path)

nonzero_list = np.nonzero(img)
        
     
cent,si = center_side(img)
print(cent,si)


im2 = create_border(img,cent,si)

plt.imshow(img)
plt.show()
plt.imshow(im2)
plt.show()
"""


