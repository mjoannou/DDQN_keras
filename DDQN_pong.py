#!/usr/bin/env python
from __future__ import print_function

import sys
import random
import numpy as np
from collections import deque

from keras import initializers
from keras.initializers import normal, identity
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import Adam

import gym
import cv2
import csv

gamma = 0.99 # decay rate of past observations
start_learning = 5000 # timesteps to observe before training
img_width , img_height = 80, 80
img_channels = 4

learning_rate = 0.0001
clip_value = 1 #gradient clipping, all paramter gradients will be clipped to be between (-1) - (+1)
target_model_update_frequency = 1000

initial_epsilon = 1 #initial value of epsilon
anneal_interval = 50000. #number of steps to anneal epsilon
final_epsilon = 0.1 #final value of epsilon

memory_size = 120000 #size of experience replay
batch_size = 32

save_points = [50000, 100000, 150000, 200000, 250000, 400000, 500000, 750000, 1000000]

def buildmodel(actions):
    model = Sequential()
    model.add(Conv2D(32, (8,8), strides=(4, 4), padding='same', activation='relu', input_shape=(img_width,img_height,img_channels))) #32 (8x8) filters
    model.add(Conv2D(64, (4,4), strides=(2, 2), padding='same', activation='relu')) #64 (4x4) filters
    model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same', activation='relu')) #64 (3x3) filters
    model.add(MaxPooling2D(pool_size=(2,2)))    #max pooling with kernel (2x2)
    model.add(Dropout(0.2)) #20% dropout
    model.add(Flatten())    #flatten for fully-connected layer
    model.add(Dense(512, activation='relu'))    #fully-connected layer, 512 neurons
    model.add(Dense(actions))   #fully connected layer with number of neurons equal to number of actions
   
    adam = Adam(lr=learning_rate, clipvalue=clip_value)
    model.compile(loss='mse',optimizer=adam)
    return model

def create_csv_file(filename):
    with open(filename, 'w', newline = '') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["TIMESTEP", "ACTION VALUES"])

def create_scores_file(filename):
    with open(filename, 'w', newline = '') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["TIMESTEP", "SCORES", "EPSILON"])

def read_data_into_csv(filename, time_step, action_values):  #score=total_reward, episode number
    with open(filename, 'a', newline = '') as csvfile:  #open csvfile (or create it) in append mode (do not overwrite)
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([time_step, action_values])

def read_data_into_scores(filename, time_step, score, epsilon):  #score=total_reward, episode number
    with open(filename, 'a', newline = '') as csvfile:  #open csvfile (or create it) in append mode (do not overwrite)
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([time_step, score, epsilon])

def preprocess_image(image):
    processed_observation = image[35:195] # crop to 160x160
    processed_observation = processed_observation[::2, ::2] #take every other pixel value (downsample) to 80x80
    processed_observation = processed_observation[:, :, 0]  #remove green and blue channels
    processed_observation[processed_observation == 144] = 0 #set all values of 144 (the background value) to black
    processed_observation[processed_observation != 0] = 255 #set all other values to white
    #cv2.namedWindow("image", cv2.WINDOW_NORMAL)    #opencv can be used to view processed image
    #cv2.imshow("image", processed_observation)
    return processed_observation

def main():
    create_csv_file("DDQN_pong_values") #create a csv file for action value predictions
    create_scores_file("DDQN_pong_scores")  #create a csv file for scores
    env = gym.make('Pong-v0')   #create OpenAI Gym environment
    gym_state = env.reset() #reset environment, return observation
    actions = env.action_space.n
    

    model = buildmodel(actions)    #build the primary model
    model.save_weights("random_seed_pong.h5", overwrite=True)   #store the initial random weights for reproducibility
    #model.load_weights("DDQN_pong.h5")
    target_model = buildmodel(actions) #build the target model
    #target_model.load_weights("DDQN_pong.h5")

    D = deque(maxlen=memory_size) #replay memory

    small_img = preprocess_image(gym_state) #crop image and turn into binary image
    gym_stack = np.stack((small_img, small_img, small_img, small_img), axis=2)  #set first observation stack to 4xsame image
    gym_stack = gym_stack.reshape(1, gym_stack.shape[0], gym_stack.shape[1], gym_stack.shape[2])    #add dimension as required by keras

    epsilon = initial_epsilon  #initialise epsilon

    step = 0 #initialise step counter
    done = False    #initialise terminal state boolean to false
    game_score = 0 #initialise game score to 0
    
    while (True):
        if done == True:
            read_data_into_scores("DDQN_pong_scores", step-1, game_score, epsilon)  #save episode data into scores file
            game_score = 0 #reset the score
            gym_state = env.reset() #reset the environment and return an observation
            
            small_img = preprocess_image(gym_state) #crop image and turn into binary image
            gym_stack = np.stack((small_img, small_img, small_img, small_img), axis=2)  #set first observation stack to 4xsame image. shape=(80,80,4)
            gym_stack = gym_stack.reshape(1, gym_stack.shape[0], gym_stack.shape[1], gym_stack.shape[2])   #add dimension as required by keras

        env.render()    #render the environment
        loss = 0 #set the 
        gym_action_index = 0
        reward = 0
        
        if random.random() <= epsilon:
            print("Random Action Selected")
            gym_action_index = random.randrange(actions)
        else:
            #zero-centre and normalise the pixel values
            prediction_stack = np.array(gym_stack, np.float32)  #keras expects float32, cannot be int for normalisation

            prediction_stack -= 128 #centre around 0 before giving to conv net
            prediction_stack /= 128 #normalise before giving to conv net
                
            gym_q = model.predict(prediction_stack) #get value predictions of each possible action
            read_data_into_csv("DDQN_pong_values", step, gym_q)  #store action values in csv file
            gym_action_index = np.argmax(gym_q)    #select index of largest value

        gym_next_state, reward, done, _ = env.step(gym_action_index) #carry out action
        game_score += reward #add reward to score for this episode

        small_next_img = preprocess_image(gym_next_state)   #preprocess the resultant observation stack
        small_next_img = small_next_img.reshape(1, small_next_img.shape[0], small_next_img.shape[1], 1) #shape (1,80,80,1) before adding to stack
        gym_next_stack = np.append(small_next_img, gym_stack[:, :, :, :3], axis=3)  #resultant observation stack

        D.append((gym_stack, gym_action_index, reward, gym_next_stack, done))   # add experience to replay memory

        if step > start_learning:   #if dedicated exploration period is over
            minibatch = random.sample(D, batch_size) #sample from the experience replay
            
            gym_inputs = np.zeros((batch_size, gym_stack.shape[1], gym_stack.shape[2], gym_stack.shape[3]))

            targets = np.zeros((gym_inputs.shape[0], actions))  #initialise target array to be trained on
            future_targets_1 = np.zeros((gym_inputs.shape[0], actions)) #current model future prediction, used to find max value index
            future_targets_2 = np.zeros((gym_inputs.shape[0], actions)) #target model future prediction, used to find value at said index

            for i in range(0, len(minibatch)):  #for each sample in the minibatch
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]

                prediction_stack_t = np.array(state_t, np.float32)  #store stack in array of floats
                prediction_stack_t -= 127.5     #zero-centre pixels
                prediction_stack_t /= 127.5     #normalise pixels

                prediction_stack_t1 = np.array(state_t1, np.float32)    #store stack in array of floats
                prediction_stack_t1 -= 127.5    #zero-centre pixels
                prediction_stack_t1 /= 127.5    #normalise pixels

                gym_inputs[i:i+1] = prediction_stack_t #set input sample to observation stack of the experience
                targets[i] = model.predict(prediction_stack_t)  #get current prediction fo all possible actions
                
                future_targets_1[i] = model.predict(prediction_stack_t1)    #get prediction of future action values using primary model
                
                future_targets_2[i] = target_model.predict(prediction_stack_t1)    #get prediction of future action values using target model
                
                best_future_act_index = np.argmax(future_targets_1[i])  #select index of highest value action from primary model
                best_future_act_val = future_targets_2[i][best_future_act_index]    #select value from target_model prediction at said index

                if done:    #if the experience is the last in the episode
                    targets[i, action_t] = reward_t #value of action is equal to reward
                else:
                    targets[i, action_t] = reward_t + gamma * best_future_act_val #value of action is equal to discounted value

            loss += model.train_on_batch(gym_inputs, targets)   #train the primary model on the batch samples

            if epsilon > final_epsilon: #if epsilon has not been annealed to final value
                epsilon -= (initial_epsilon - final_epsilon) / anneal_interval  #decrement epsilon

            if step % target_model_update_frequency == 0:   #if it is time to update the target model
                target_model.set_weights(model.get_weights())   #copy weights from primary model to target model

        gym_stack = gym_next_stack #set the resultant observation of the previous step to the initial observation of the next step
        step += 1 #increment the step counter

        if step % 1000 == 0:    #every 1000 steps
            model.save_weights("DDQN_pong.h5", overwrite=True)  #save weights

        if step in save_points: #if current step is a save points
            model.save_weights("DDQN_pong_{}.h5".format(step), overwrite=True) #save weights in step-specific file

        print("Step: {}    Epsilon: {}    Action: {}    Loss: {}".format(step, epsilon, gym_action_index, loss))

if __name__ == "__main__":
    main()
