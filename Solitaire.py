from copy import deepcopy
from enum import Enum
import random 
#import matplotlib.pyplot as plt 
from scipy import stats
import tensorflow as tf 





'''
enumeration for easy reading of moves 
'''
class direction(Enum):
	Right = 0
	Left = 1
	Down = 2
	Up = 3


'''
Game Coordinates 
returns an array of all coordiates on the board that have a move associated with them
ie. the center of the board can move in all four directions 
[3, 3, 1], [3, 3, 0], [3, 3, 3], [3, 3, 2]
there are 73 possible moves of the game 
these will be made into one-hot outputs
'''
def GameCoordinates(BoardArray):
	height= 7#len(BoardLines)
	width = 7#len(BoardLines)


	gamemoves = []

	for vertical in range(0,height):
		for horizontal in range(0,width):

			if (horizontal - 2) >= 0  and BoardArray[vertical][horizontal] != 2:			
				if (BoardArray[vertical][horizontal-2] !=2):
					gamemoves.append([vertical,horizontal,direction.Left.value])


			if (horizontal + 2) <= 6 and BoardArray[vertical][horizontal] != 2:
				if (BoardArray[vertical][horizontal+2] !=2):
					gamemoves.append([vertical,horizontal,direction.Right.value])

			if (vertical - 2) >0  and BoardArray[vertical][horizontal] != 2:
				if (BoardArray[vertical-2][horizontal] !=2):			
					gamemoves.append([vertical,horizontal,direction.Up.value])


			if (vertical + 2) <= 6 and BoardArray[vertical][horizontal] != 2:
				if (BoardArray[vertical+2][horizontal] !=2):
					gamemoves.append([vertical,horizontal,direction.Down.value])

	return gamemoves


'''
Find moves 
returns an array of all legal Available moves of a board 
'''
def FindMoves(Coordinates, GameBoard,training_on = False):

	AvailableMoves = []
	
	for p in Coordinates:
		
		if p[1] + 2 <= 6:
			if GameBoard[p[0]	][p[1]+1] != 2 and GameBoard[p[0]][p[1]+2] != 2: 
				if GameBoard[p[0]][p[1]] == 1 and GameBoard[p[0]][p[1]+1] == 1 and GameBoard[p[0]][p[1]+2] == 0: #move right
					if training_on:
						AvailableMoves.append([p[0],p[1],direction.Right.value])

					if not training_on:
						return True

		if p[1] - 2 >= 0:
			if GameBoard[p[0]	][p[1]-1] != 2 and GameBoard[p[0]][p[1]-2] != 2: 
				if GameBoard[p[0]][p[1]] == 1 and GameBoard[p[0]][p[1]-1] == 1 and GameBoard[p[0]][p[1]-2] == 0: #move left
					if training_on:
						AvailableMoves.append([p[0],p[1],direction.Left.value])
					if not training_on:
						return True

		if p[0] + 2 <= 6:
			if GameBoard[p[0]+1	][p[1]	] != 2 and GameBoard[p[0]+2][p[1]] != 2: 
				if GameBoard[p[0]][p[1]] == 1 and GameBoard[p[0]+1][p[1]] == 1 and GameBoard[p[0]+2][p[1]] == 0: #move down
					if training_on:
						AvailableMoves.append([p[0],p[1],direction.Down.value])
					if not training_on:
						return True

		if p[0] - 2 >= 0:
			if GameBoard[p[0]-1][p[1]	] != 2 and GameBoard[p[0]	-2][p[1]] != 2: 
				if GameBoard[p[0]][p[1]] == 1 and GameBoard[p[0]-1][p[1]] == 1 and GameBoard[p[0]-2][p[1]] == 0: #move up
					if training_on:
						AvailableMoves.append([p[0],p[1],direction.Up.value])

					if not training_on:
						return True
	return AvailableMoves


'''
loads board from file and returns as a 2d array 
'''
def LoadBoard(File):
	BoardFile = open(File).read()

	BoardFileLines = BoardFile.split("\n")

	BoardArray = [[] for i in range(len(BoardFileLines))]
	j = 0

	for eachline in BoardFileLines:

		data = eachline.split(",")
		
		for i in range(0,len(data)):
			pass
			BoardArray[j].append(int(data[i]))
		j+=1


	return BoardArray


#if there are marbles left on the board , the score is the distance each marble is away from the center  
def ScoreTrainingGame(Board):
	Score = 0

	for row in range(0,len(Board[0])):
		for column in range(0,7):
			if Board[row][column] ==1:
				Score += (abs(row - 3) + abs(column -3))

	return Score 

#once a move has been selected, TakeMove changes the state of the board to reflect chosen move.				
def TakeMove(Board,Move):

	y = Move[0]
	x = Move[1]
	d = Move[2]

	if d== direction.Right.value:
		Board[y][x]  = 0
		Board[y][x+1]= 0
		Board[y][x+2]= 1
		
	elif d == direction.Left.value:
		Board[y][x]  = 0
		Board[y][x-1]= 0
		Board[y][x-2]= 1

	elif d == direction.Up.value:
		Board[y][x]  = 0	
		Board[y-1][x]= 0
		Board[y-2][x]= 1

	elif d == direction.Down.value:
		Board[y][x]  = 0	
		Board[y+1][x]= 0
		Board[y+2][x]= 1

	return Board

#ReshapeBoard takes the 2d board and turns it into a 1d array
def ReshapeBoard(Board):

	flattenedBoard = []
	for eachline in Board:
		for element in eachline:
			if element == 0 or element ==1:
				flattenedBoard.append(element)

	return flattenedBoard

#during training score selection a frequency distribution was required to see what the average game score was 
def PlotFrequencyDistribution(Scores):
	frequency = stats.itemfreq(Scores)
	plt.plot(frequency[:,0],frequency[:,1])
	plt.xlabel('Score')
	plt.ylabel('Frequency')
	plt.show()

#writes the board and move data to file as [x,x,x...x],[y,y,y] where x is the state of a position on the board and y is the move and direction
def WriteDataToFile(Boards,Moves,infile):

	with open(infile,'a') as f:
		for i in range(0,len(Moves)):

			f.write('%s,%s'%(Boards[i],Moves[i]))
			f.write('\r\n')			

		f.write('\r\n')	
#writes data as continues stream of board state and move with direciton
#each new line is the next state of the board 
#each game is seperated by an empty line 
def WriteDataToFileV2(Boards,Moves,infile):
	with open(infile,'a') as f:

		for j in range(0,len(Boards)):
			for element in Boards[j]:
				f.write('%s'%element)
				f.write(',')

			for i in range(0,len(Moves[j])):

				if i != len(Moves[j])-1:

					f.write('%s'%Moves[j][i])
					f.write(',')
				else:
					f.write('%s'%Moves[j][i])
			f.write('\r\n')
		f.write('\r\n')

#the start of a game is defined by an empty line followed by a line of data 
#each game is stored as a 2d array (rows being each time step and column being state of board + move with direction)
#each game is then stored in a separate array that holds all games for use in batch processing 
def create_batched_training_data(file):
	raw_data  = open(file,'r').read() # pull in game data , each game is separated by a blank line 


	split_data_by_line = raw_data.split('\r\n') #split raw data by line 

	all_board_data = [] # contains all games board data 
	all_label_data = [] # contains all games label data ( move + direction)
	this_game_board_data = [] # contains this game board states 
	this_game_label = [] # contains this game labels 

	for eachline  in split_data_by_line: 
		if len(eachline) > 0: #if the length of the line is greater than zero then we are in a game 

			data = eachline.split(',') # the first 33 elements are the board , the last 3 are the move + direction 
			tempData = [] # store this turns board state as a temp value to be added to the overall game array 
			tempLabel = []
			for j in range(0,33): # first 33 elements are board state 
				tempData.append(data[j])
				
			for j in range(33,36): # last 3 elements are labels 
				tempLabel.append(data[j])

			this_game_board_data.append(tempData) # build up 2d array of game 
			this_game_label.append(tempLabel)	

		if len(eachline) == 0:	# if a new line is found the game has ended. add game and labels to the overall game array that stores all games.
			all_board_data.append(this_game_board_data)
			all_label_data.append(this_game_label)
			this_game_board_data = [] # reset current game array 
			this_game_label = []

	return all_board_data , all_label_data



def training_games():
	Board = LoadBoard("board.txt") # load board from file 
	Coords = GameCoordinates(Board) #find all possible coorindates with a direction where a move can be made. 
	CurrentBoard = deepcopy(Board) #take a copy of the game board in its starting state. 

	Scores = [] #used for identifying frequency of scores 

	for game in range(0,100000):	# start loop here 

		ThisGameMoves = [] #hold an array of all the moves used in a game. write to file if the game was good
		ThisGameBoard = [] #hold an array of the state of the board per turn. write to file if the game was good

		CurrentBoard = deepcopy(Board) #
		while True:
			AvailableMoves = FindMoves(Coords,CurrentBoard,True) #find all avaliable moves on the board in its current state 

			if len(AvailableMoves) == 0: # if no moves are left count up score 
				score = ScoreTrainingGame(CurrentBoard)
				Scores.append(score)

				if score <= 8:#if the score is less than 10 it is concidered a good game! 

					WriteDataToFileV2(ThisGameBoard,ThisGameMoves,'training_dataV2.txt')
				break

			RandomMove = random.randint(0,len(AvailableMoves)-1)#select a random move from the avaliable moves 

			ThisGameMoves.append(AvailableMoves[RandomMove]) #append the move taken this turn 
			
			#append the board in its current state before a move is made and reshape for input into NN
			ThisGameBoard.append(ReshapeBoard(CurrentBoard)) 

			TakeMove(CurrentBoard,AvailableMoves[RandomMove]) #make a move! 

		if game % 5000 == 0:
			print (game) 
	#PlotFrequencyDistribution(Scores) #plot the freuency distribution of score of all games played 



#training_games()
games, labels = create_batched_training_data('training_dataV2.txt') 

















