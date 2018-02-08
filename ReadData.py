import numpy as np 
from Solitaire import LoadBoard,GameCoordinates
indata = open('training_dataV2.txt','r').read()


indata = indata.split('\r\n')


board_data = []
label = []


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

board_data = np.array(board_data,dtype = int)
label = np.array(label,dtype = int)


Board = LoadBoard("board.txt") # load board from file 
Coords = GameCoordinates(Board) #find all possible coorindates with a direction where a move can be made. 

values = [i for i in range(73)]

test_dict = dict(zip(values,Coords))
test_dict2 = dict(zip(Coords,values))



#one_hot = np.eye(73)[test_dict2[label]]


