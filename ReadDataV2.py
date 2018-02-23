import numpy as np 
from Solitaire import LoadBoard,GameCoordinates
from collections import defaultdict
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

	elif len(eachline) == 0:
		

board_data = np.array(board_data,dtype = int) #training game board data
label = np.array(label,dtype = int)           #training game label data 