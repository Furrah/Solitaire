import numpy as np 
from Solitaire import LoadBoard,GameCoordinates

#from collections import defaultdict

import sys



indata = open('training_dataV2.txt','r').read() # pull in game data , each game is separated by a blank line 

# write to file function works a bit differently in windows and OSX so reading data back in requires OS specific splitting of data 
if sys.platform.startswith('darwin'):
	indata = indata.split('\r\n')

elif sys.platform.startswith('win'):
	indata = indata.split('\n\n')


board_data = []
label = []

#first 33 elements are the board , last 3 elements are the move coordinates and direction
#curerently each line is delimited and stored in a corresponding array 
#future adaption should create 3d array . X x Y = state of board, Z = game number   
for eachline in indata:


	if len(eachline)>0:

		data = eachline.split(',')

		tempData = []
		tempLabel = []
		for j in range(0,33):
			tempData.append(data[j])
			
		for j in range(33,36):
			tempLabel.append(data[j])

		board_data.append(tempData)
		label.append(tempLabel)


board_data = np.array(board_data,dtype = int) #training game board data
label = np.array(label,dtype = int)           #training game label data 



Board = LoadBoard("board.txt") # load board from file 
Coords = GameCoordinates(Board) #find all possible coorindates with a direction where a move can be made. 



Coords = tuple(Coords) #convert all game moves into array of tuples 



#we need to convert all possible game moves into a number ie (1,2,3) = 1 ,(2,3,4) = 2 and so on
#this is to convert the one hot output of the NN into an actual move that will be used to change the state of the board 


coord_to_hot = {}
hot_to_coord = {}



for i in range(0,len(Coords)):
	coord_to_hot[tuple(Coords[i])] = np.eye(73)[i]


for i in range(0,len(Coords)):
	hot_to_coord[i] = Coords[i] 


print coord_to_hot[tuple(Coords[20])]
print hot_to_coord[20]

print coord_to_hot[tuple([2,4,3])]




# label_to_hot = [] # one hot needs to work the other way round as well. here we take 

# for value in label:
# 	label_to_hot.append(coord_to_hot[tuple(value)])

# one_hot = np.eye(73)[label_to_hot]





