import Solitaire
import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn

#training_games()
games, labels = Solitaire.create_batched_training_data('training_dataV3.txt') 

train_X, test_X, train_Y, test_Y = train_test_split(games,labels,test_size=0.33, random_state = 42)


Coord_to_one_hot , One_hot_to_Coord = Solitaire.label_to_one_hot()

input_layer_size = 33
classes = 76
epochs = 10000

time_steps = 33
n_input = 1

for each_game in train_X:
    each_game = np.array(each_game)
    print each_game.shape
    each_game = each_game.reshape((len(each_game),time_steps,n_input))

    print each_game.shape



#x = tf.placeholder('float',[None,input_layer_size])
x = tf.placeholder('float',[None,time_steps,n_input])
y = tf.placeholder('float',[None,classes])

def Fully_Connected_Layer(inputs,channels_in ,channels_out, NameScope = '',activation = True):

    with tf.name_scope(NameScope):
        hidden_layer = {'Weights': tf.Variable(tf.random_normal([channels_in,channels_out]),'float',name = 'W'),
        'Biases' :tf.Variable(tf.random_normal([channels_out]),'float', name = 'B')} 

        tf.summary.histogram("weights", hidden_layer['Weights'])
        tf.summary.histogram("biases", hidden_layer['Biases'])

        action = tf.add(tf.matmul(inputs,hidden_layer['Weights']),hidden_layer['Biases'])

        if activation:
            action = tf.nn.sigmoid(action)
        return action 


def Recurrent_Layer(inputs):
    num_units = 128
    n_classes = 76
    t_steps = 33

    out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
    out_bias=tf.Variable(tf.random_normal([n_classes]))

    inputs=tf.unstack(inputs  ,t_steps,1)
    lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
    outputs,_=rnn.static_rnn(lstm_layer,inputs,dtype="float")

    outputs = tf.matmul(outputs[-1],out_weights)+out_bias

    return outputs 



def Neural_Network(data):
    fc1 = Fully_Connected_Layer(data,input_layer_size,200,'hidden_layer_1',True)
    fc2 = Fully_Connected_Layer(fc1,200,500,'hidden_layer_2',True)
    fc3 = Fully_Connected_Layer(fc2,500,classes,'hidden_layer_3',False)

    return fc3



def train_network(x):

    #prediction = Neural_Network(x)
    prediction = Recurrent_Layer(x)

    with tf.name_scope('xent'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction ,labels = y ))
        tf.summary.scalar('xent',cost)


    with tf.name_scope('train'):
        optimiser = tf.train.AdamOptimizer().minimize(cost)



    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    merged_summary  = tf.summary.merge_all()    
    writer = tf.summary.FileWriter('/Users/Joe/Projects/Solitaire')


    with tf.Session() as sess:
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())     

        for epoch in range(epochs):



            epoch_loss = 0
            scalar_output = True 

            for each_game,game_label in zip(train_X,train_Y):

                #print len(each_game), len(game_label)

                each_game = np.array(each_game)

                each_game = each_game.reshape((len(each_game),time_steps,n_input))

                one_hot_label = []
                for label in game_label:


                    one_hot_label.append(Coord_to_one_hot[tuple(label)])
                game_label = np.array(one_hot_label)

                _, c = sess.run([optimiser, cost], feed_dict={x: each_game, y: game_label})
                epoch_loss += c

                if epoch % 10 and scalar_output == True:
                    scalar_output = False
                    s = sess.run(merged_summary, feed_dict= {x : each_game, y:game_label})

    
                    writer.add_summary(s,epoch)

        
            print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)



        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        

        sum_of_accuracys = 0.0
        for each_game,game_label in zip(test_X,test_Y):

            each_game = np.array(each_game)



            ylabels = []
            for label in game_label:


                ylabels.append(Coord_to_one_hot[tuple(label)])
            ylabels =  np.array(ylabels)

            output = accuracy.eval({x:each_game , y: ylabels})

            sum_of_accuracys += output
#           print('Accuracy:',accuracy.eval({x:each_game , y: ylabels}))
            print ('Accuracy', output)

        print ('Average Accuracy', (sum_of_accuracys/len(test_X)))


train_network(x)





