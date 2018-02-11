from copy import deepcopy
from enum import Enum
import random 
#import matplotlib.pyplot as plt 
from scipy import stats






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

def ReshapeBoard(Board):

	flattenedBoard = []
	for eachline in Board:
		for element in eachline:
			if element == 0 or element ==1:
				flattenedBoard.append(element)

	return flattenedBoard

def PlotFrequencyDistribution(Scores):
	frequency = stats.itemfreq(Scores)
	plt.plot(frequency[:,0],frequency[:,1])
	plt.xlabel('Score')
	plt.ylabel('Frequency')
	plt.show()


def WriteDataToFile(Boards,Moves,infile):

	with open(infile,'a') as f:
		for i in range(0,len(Moves)):

			f.write('%s,%s'%(Boards[i],Moves[i]))
			f.write('\r\n')			

		f.write('\r\n')	

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



def training_games():
	Board = LoadBoard("board.txt") # load board from file 
	Coords = GameCoordinates(Board) #find all possible coorindates with a direction where a move can be made. 
	CurrentBoard = deepcopy(Board) #take a copy of the game board in its starting state. 

	Scores = [] #used for identifying frequency of scores 

	for game in range(0,1000):	# start loop here 

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
















