import numpy as np 

def create_batched_training_data(file):
	raw_data  = open(file,'r').read() # pull in game data , each game is separated by a blank line 


	split_data_by_line = raw_data.split('\r\n')

	all_board_data = []
	all_label_data = []
	this_game_board_data = []
	this_game_label = []

	for eachline  in split_data_by_line:
		if len(eachline) > 0:

			data = eachline.split(',')
			tempData = []
			tempLabel = []
			for j in range(0,33):
				tempData.append(data[j])
				
			for j in range(33,36):
				tempLabel.append(data[j])

			this_game_board_data.append(tempData)
			this_game_label.append(tempLabel)	
		if len(eachline) == 0:
			all_board_data.append(this_game_board_data)
			all_label_data.append(this_game_label)
			this_game_board_data = []
			this_game_label = []

	return all_board_data , all_label_data

create_batched_training_data('training_dataV2.txt') 


