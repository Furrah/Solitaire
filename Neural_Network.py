import Solitaire
import tensorflow as tf 
import numpy as np 

#training_games()
games, labels = Solitaire.create_batched_training_data('training_dataV2.txt') 

Coord_to_one_hot , One_hot_to_Coord = Solitaire.label_to_one_hot()

input_layer_size = 33
classes = 73
epochs = 1000

x = tf.placeholder('float',[None,input_layer_size])
y = tf.placeholder('float',[None,classes])

def Fully_Connected_Layer(inputs,channels_in ,channels_out, NameScope = '',activation = True):

	with tf.name_scope(NameScope):
		hidden_layer = {'Weights': tf.Variable(tf.random_normal([channels_in,channels_out]),'float'),
		'Biases' :tf.Variable(tf.random_normal([channels_out]),'float')} 

        action = tf.add(tf.matmul(inputs,hidden_layer['Weights']),hidden_layer['Biases'])


        action = tf.nn.relu(action)
        return action 


def Neural_Network(data):
	fc1 = Fully_Connected_Layer(games[0],input_layer_size,200,'hidden_layer_1')
	fc2 = Fully_Connected_Layer(fc1,200,500,'hidden_layer_2')
	fc3 = Fully_Connected_Layer(fc2,500,classes,'hidden_layer_3',False)

	return fc3



def train_network(x):
	prediction = Neural_Network(x)

	cost = tf.nn.softmax_cross_entropy_with_logits(logits = prediction ,labels = y )
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())		

		for epoch in range(epochs):

			for each_game,game_label in zip(games,labels):

				each_game = np.array(each_game)

				game_label = np.array(game_label)




				_, c = sess.run([optimizer, cost], feed_dict={x: each_game, y: game_label})
				epoch_loss += c
		
		print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)



		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x: test_X, y:test_Y}))	




#train_network(x)

for each_game,game_label in zip(games,labels):

	each_game = np.array(each_game)

	one_hot_label = []
	for label in game_label:

		print label
		one_hot_label.append(Coord_to_one_hot[tuple(label)])
	game_label = np.array(one_hot_label)



# for lab in labels[0]:
# 	print Coord_to_one_hot[tuple(lab)]
# 	print lab





