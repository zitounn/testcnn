import os
import numpy as np

cwd = os.getcwd()

list = []
for file in os.listdir(cwd+'/train'):
	if file.endswith('mask.jpg'):
            list.append(file[0:8]+'.jpg')

#		print (int(file[5:8]))

sorted_list = sorted(list, key= lambda x: int(x[5:8]))

label = np.zeros(len(list))
counter = 0
for file in sorted_list:
	
	if file[2:5] == 'CLN':
		label[counter] = 1	
	counter = counter +1
#print (label)
print (len(list))


#print (sorted(list, key= lambda x: int(x[5:8])))
#print ( len(list))
#print (label[0])
