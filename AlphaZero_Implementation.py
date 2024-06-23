#Implementation of the AlphaZero Algorithm
#By Victor Habiyambere
#Started: June 9th, 2024
#2024-06-09

#2024-05-15

#Break-up your implementation in mangeable steps
#Let's import the library because we will obviously use it

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT,RIGHT_ONLY,SIMPLE_MOVEMENT
import gym
from math import *
from chessboard import display

#Now, I will proceed to import as many useful deep learning libraries within this program
#I will also be conscious of the fact that I will need to analyze my experiments with this program
#effectively, therefore a good set of visualization tools will be very useful

#deep learning libraries
import torch
from IPython.display import clear_output
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
from torch.nn.utils import parameters_to_vector
from multiprocessing import Process
from multiprocessing import set_start_method
#Data-structure libraries:
from collections import deque
import numpy as np
#Randomness Library:
from random import shuffle
import random
import copy
import sys


#Plotting library
#I can use the plotting library in order to visualize the probabilities of certain actions of the AI(Artificial Intelligence)
import matplotlib.pyplot as plt
import chess

#Image-processing library:
from skimage.transform import resize

#----------------------->In-case I want to introduce intrinsic motivation to the AI Algorithm 
#Fixed Net architecture
class target_net(nn.Module):
    def __init__(self):
        super(target_net,self).__init__()
        #3 Layer Deep Neural Network(it's weights are frozen)
        self.conv1 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.linear = nn.Linear(196,512)
    def forward(self,x):
        encoded_board = utilities.encode_info(x)
        encoded_board = torch.Tensor(encoded_board).reshape(1,8,8).to('cuda')
        x = encoded_board
        x = F.normalize(x,dim=0)
        y = F.selu(self.conv1(x))
        y = F.selu(self.conv2(y))
        y = F.selu(self.conv3(y))
        y = self.linear(y.flatten())
        return y

#Predictor Model 
class predictor_net(nn.Module):
    def __init__(self):
        super(predictor_net,self).__init__()
        #3 Layer Deep Neural Network(it's weights are not frozen)
        self.conv1 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.linear = nn.Linear(196,512)
        self.linear2 = nn.Linear(512,512)
        self.linear3 = nn.Linear(512,512)
        
    def forward(self,x):
        encoded_board = utilities.encode_info(x)
        encoded_board = torch.Tensor(encoded_board).reshape(1,8,8).to('cuda')
        x = encoded_board
        x = F.normalize(x,dim=0)
        y = F.selu(self.conv1(x))
        y = F.selu(self.conv2(y))
        y = F.selu(self.conv3(y))
        y = F.selu(self.linear(y.flatten()))
        y = F.selu(self.linear2(y))
        y = self.linear3(y)
        return y

def get_loss(obs,predictor,target):
    pred = predictor
    trg = target
    input_1 = obs
    target_output = trg(input_1)
    estimated_output = pred(input_1)
    lossfn = nn.MSELoss()
    loss = lossfn(estimated_output,target_output.detach())
    return loss

class decision_maker(nn.Module):
    #Initialization
    def __init__(self,aspace=12):
        super(decision_maker,self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.linear1 = nn.Linear(256,512)
        #Pick a piece(that is available)
        self.pick = nn.Linear(512,64)
        #Pick a location for the piece(that is available)
        self.location = nn.Linear(512,64)
        self.action = None
        self.action2 = None
        self.aspace = aspace
        self._initialize_weights()

    #Output a probability distribution over all primitive actions
    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.selu(self.conv1(x))
        y = F.selu(self.conv2(y))
        y = F.selu(self.conv3(y))
        y = F.selu(self.conv4(y))
        #Use the sigmoid function
        y = F.selu(self.linear1(y.flatten()))
        y1 = y
        y = F.softmax(self.pick(y1.flatten()))
        y2 = F.softmax(self.location(y1.flatten()))
        #Piece selected
        self.action = np.random.choice(a=range(64),p=y.cpu().detach().numpy())
        #Location selected
        self.action2 = np.random.choice(a=range(64),p=y2.cpu().detach().numpy())
        return y,y2

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('selu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

#Only 1 single critic 
class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size = 1, padding = 1)
        self.linear1 = nn.Linear(256,512)
        #Outputs either true or false 
        self.linear = nn.Linear(512,1)
        
    def forward(self,x):
        #Pass the input to the neural-network
        x = F.normalize(x,dim=0)
        y = F.selu(self.conv1(x))
        y = F.selu(self.conv2(y))
        y = F.selu(self.conv3(y))
        y = F.selu(self.conv4(y))
        y = F.selu(self.linear1(y.flatten()))
        #Use the sigmoid function
        y = F.sigmoid(self.linear(y.flatten()))
        return y

#Typical Advantage Actor Critic, but with planning enabled
class ActorCritic(nn.Module):
    
    def __init__(self):
        super(ActorCritic,self).__init__()
        self.action = None
        self.action2 = None
        self.Critic = Critic().to('cuda').share_memory()
        self.Critic2 = Critic().to('cuda').share_memory()
        self.critic = 0
        self.critic2 = 0
        self.AC = decision_maker(aspace=64).to('cuda').share_memory()
        self.optimizer1 = optim.Adam(self.Critic.parameters(),lr=1e-3)
        self.optimizer2 = optim.Adam(self.Critic2.parameters(),lr=1e-3)
        self.optimizer3 = optim.Adam(list(self.AC.parameters()),lr=1e-4)
        self.Value_Sums = [0 for x in range(64)]
        self.Value_Sums2 = [0 for x in range(64)]
        self.Action_Sample = [0 for x in range(64)]
        self.Action_Sample2 = [0 for x in range(64)]
        self.C = float(2 ** 0.5)
        self.prob1 = 0
        self.prob2 = 0
        self.best_node = None
        self.N_visits = 0
        self.planning = True
        self.ucb_scores = None
        self.ucb_scores2 = None
        self.First_Time = True
        self.steps = 0
        self.maxsteps = 5

    def get_ucb(self,value,visits,samples,prior):
        if samples == 0:
            return float('inf')
        else:
            ucb = (value/samples) + prior * sqrt(2) * sqrt(visits)/(samples+1)
            return ucb
    
    def forward(self,x):
        #Now, time to implement the planning stage...
        env1 = x.copy()
        #If planning is enabled:
        if self.planning == True and self.steps < self.maxsteps and not env1.is_checkmate() and not env1.is_stalemate() and not env1.is_insufficient_material() and not env1.is_seventyfive_moves() and not env1.is_fivefold_repetition() and not env1.can_claim_draw():
            encoded_board = utilities.encode_info(env1)
            encoded_board = torch.Tensor(encoded_board).reshape(1,8,8).to('cuda')
            probs,probs2 = self.AC(encoded_board)
            self.prob1 = probs
            self.prob2 = probs2
            critic = self.Critic(encoded_board)
            critic2 = self.Critic2(encoded_board)
            self.critic = critic
            self.critic2 = critic2
            ucb_scores = []
            ucb_scores2 = []
            #Use the priors
            for i in range(64):
                self.Value_Sums[i] += critic.item() + critic2.item()
                ucb_score = self.get_ucb(self.Value_Sums[i],self.N_visits,self.Action_Sample[i],probs[i].item())
                ucb_scores.append(ucb_score)
            for i in range(64):
                self.Value_Sums2[i] += critic.item() + critic2.item()
                ucb_score = self.get_ucb(self.Value_Sums2[i],self.N_visits,self.Action_Sample2[i],probs2[i].item())
                ucb_scores2.append(ucb_score)
            self.ucb_scores = ucb_scores
            self.ucb_scores2 = ucb_scores2
            legal = False
            does_exist = False
            does_exist2 = False
            legal_moves = list(env1.legal_moves)
            while not legal:
                action = np.argmax(np.array(ucb_scores)) + 1
                action2 = np.argmax(np.array(ucb_scores2)) + 1
                curr_location = utilities.encode_legalmove(action)
                next_location = utilities.encode_legalmove(action2)
                combined_move = curr_location + next_location
                for move in legal_moves:
                    move = str(move)
                    if move == combined_move:
                        legal = True
                        break
                    if move[:2] == curr_location:
                        does_exist = True
                    elif move[2:] == next_location:
                        does_exist2 = True
                if does_exist == False:
                    ucb_scores[action-1] = -float('inf')
                elif does_exist2 == False:
                    ucb_scores2[action2-1] = -float('inf')
                if does_exist == True and does_exist2 == True:
                    ucb_scores[action-1] = -float('inf')
                does_exist = False
                does_exist2 = False
            #Whatever is left, is a legal move
            self.Action_Sample[action-1] += 1
            self.Action_Sample2[action2-1] += 1
            self.N_visits += 1
            self.steps += 1
            if self.First_Time == True:
                self.action = action
                self.action2 = action2
                self.First_Time = False
            #Play the move
            env1.push_uci(combined_move)
            self.forward(env1)
        else:
            self.planning = False
            self.steps = 0
        return self.prob1,self.prob2,self.critic,self.critic2
            
    #Now, change the loss function
    def loss(self,prob1,old_prob,adv,eps=0.2):
        prob_ratio = prob1/old_prob
        comp1 = prob_ratio * adv
        comp2 = torch.clip(prob_ratio,1-eps,1+eps)*adv
        loss = min(comp1,comp2)
        return loss

#Utility function in order to encode the chess board and also decode the chessboard
#Utility function in order to encode the chess board and also decode the chessboard
class utilities:

    #Encoder function
    def encode_info(chessboard):
        piece_encoded = []
        pieces_selected = []
        found_piece = 0
        counta = 1
        for i in range(64):
            piece = chessboard.piece_at(i)
            if str(piece) not in pieces_selected:
                pieces_selected.append(str(piece))
            for p in pieces_selected:
                if p == str(piece):
                    piece_encoded.append(counta)
                    break
                else:
                    counta += 1
            counta = 1
        return piece_encoded

    #Skeleton function
    def get_skeleton(chessboard):
        cnter = 0
        piece_encoded = []
        pieces_selected = []
        found_piece = 0
        counta = 1
        total_count = 13
        for i in range(64):
            piece = chessboard.piece_at(i)
            if str(piece) not in pieces_selected:
                cnter += 1
                pieces_selected.append(str(piece))
        return pieces_selected
    
    #Decoder function
    def decode_info(encoded):
        board1 = chess.Board()
        skeleton = utilities.get_skeleton(board1)
        decoded = []
        for v in encoded:
            decoded.append(skeleton[v-1])
        return decoded

    def decode_legalmove(move):
        dict1 = { 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5, 'f' : 6, 'g' : 7, 'h' : 8}
        relative_location = int(dict1[move[0]]) * int(move[1])
        relative_location2 = int(dict1[move[2]]) * int(move[3])
        return relative_location,relative_location2

    def decode_legalpos(pos):
        dict1 = { 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5, 'f' : 6, 'g' : 7, 'h' : 8}
        relative_location = int(dict1[pos[0]]) * int(pos[1])
        return relative_location

    def encode_legalmove(decoded):
        subtract = 9
        amount = 1
        dict1 = { 1 : 'a', 2 : 'b', 3 : 'c', 4 : 'd', 5 : 'e', 6 : 'f', 7 : 'g' , 8 : 'h' }
        last_stand = 0
        letter_pos = 0
        encoded_pos = ""
        times = 1
        if decoded == 0:
            return -1
        while decoded >= 0:
            last_stand = decoded
            decoded -= 8
            if decoded <= 0:
                break
            else:
                times += 1
        while last_stand >= 0:
            last_stand -= 1
            if last_stand < 0:
                break
            else:
                letter_pos += 1
        encoded_pos = dict1[letter_pos] + str(times)
        return encoded_pos
        
    def moves_for_piece(legal_moves,board,selected_piece,turn1):
        moves_for_piece = []
        initial_position = []
        pieces = utilities.get_skeleton(board)
        if turn1 == True:
            actual_piece = pieces[selected_piece]
        elif turn1 == False:
            actual_piece = pieces[selected_piece + 7]
        for move in legal_moves:
            pos1,pos2 = utilities.decode_legalmove(move.uci())
            piece1 = board.piece_at(pos1-1)
            if str(piece1) == str(actual_piece):
                initial_position.append(pos1)
                moves_for_piece.append(pos2)
        return initial_position,moves_for_piece

#**Clean Up the Traning Function
#**In the process of cleaning it up
def train(epochs,AC,pred,target,Counter):
    
    env = chess.Board()

    gamma = 0.9 #C
    gamma2 = 0.99
    lifespan = 0
    maxlength = 15
    high_score = 0
    actor_loss = 0
    critic_loss = 0
    epoch_counter = 0
    episode_length = 0
    actor_loss = 0
    critic_loss = 0
    net_reward = 0
    
    critics = []
    critics2 = []
    rewards = []
    prob_actions = []
    target_outputs = []
    estimated_outputs = []
    losses_ = []
    rewards2 = []
    old_prob = 1
    curr_prob = 1
    curr_probs = []
    curr_probs2 = []
    old_probs = []
    old_probs2 = []
    entropies = []
    entropies2 = []
    controller_losses = []
    curiosity_spikes = []
    extrinsic_returns = []
    intrinsic_returns = []
    ppo_losses = []
    rn_losses = []
    expected_rewards = []
    dynamic_outputs = []
    representations = []
    dynamics_losses = []
    action_probs = []
    action_probs2 = []
    actions = []
    best_nodes = []
    losses = []
    
    updated_parameters = False
    next_time = False

    done = False
    lossfn = nn.MSELoss()
    lossfn2 = nn.CrossEntropyLoss()
    
    curiosity = optim.Adam(pred.parameters(),lr=1e-4)
    #prev_lives = env.ale.lives()
    while epoch_counter != epochs:
        #Get the probabilities and critic value from the Actor Critic
        probs,probs2,critic_,critic_2 = AC(env)
        action1 = AC.action
        action2 = AC.action2
        curr_location = utilities.encode_legalmove(action1)
        next_location = utilities.encode_legalmove(action2)
        combined_move = curr_location + next_location
        env.push_uci(combined_move)
        display.start(env.fen())
        
        AC.planning = True
        AC.First_Time = True
        
        critics.append(critic_)
        critics2.append(critic_2)
        
        curr_prob = probs[action1-1]
        action_probs.append(probs)
        curr_prob2 = probs2[action2-1]
        action_probs2.append(probs2)
        
        curr_probs.append(curr_prob)
        curr_probs2.append(curr_prob2)
        
        entropy = Categorical(probs).entropy()
        entropies.append(entropy)

        entropy2 = Categorical(probs2).entropy()
        entropies2.append(entropy2)
        
        loss = get_loss(env,pred,target)
        i_reward = loss
        
        episode_length += 1
        lifespan += 1

        if env.is_checkmate() or env.is_stalemate() or env.is_insufficient_material() or env.is_seventyfive_moves() or env.is_fivefold_repetition() or env.can_claim_draw():
            done = True

        elif not done:
            losses.append(float(i_reward))
            
        if done:

            if env.turn == True and env.is_checkmate():
                for i in range(len(critics2)):
                    rewards.append(0.05)
            elif env.turn == False and env.is_checkmate():
                for i in range(len(critics2)):
                    rewards.append(+1)
            if env.is_stalemate() or env.is_insufficient_material() or env.is_seventyfive_moves() or env.is_fivefold_repetition() or env.can_claim_draw():
                for i in range(len(critics2)):
                    rewards.append(0.05)
            
            losses.append(0)
            epoch_counter += 1
            Counter.value += 1
            episode_length = 0
            ppo_loss = 0
            i_loss = torch.Tensor([0])
            e_loss = torch.Tensor([0])
            i_ = len(critics) - 1
            loss2 = 0
            expected_return = 0
            expected_return2 = 0
            Return = 0
            Return2 = 0
            b = 0
            env.reset()
            
            if updated_parameters == False:
                for i in range(len(curr_probs)):
                    old_probs.append(curr_probs[i])
                for i in range(len(curr_probs)):
                    old_probs2.append(curr_probs2[i])

            #Compute the PPO Los,the Intrinsic Loss and the Extrinsic Loss
            
            for critic1 in reversed(critics2):
                
                #Calculate the actual net return
                Return = Return * gamma2 ** b + torch.Tensor([rewards[i_]]).to('cuda')
                Return2 = Return2 * gamma ** b + torch.Tensor([losses[i_]]).to('cuda')
                
                #Calculate the cumulative expected return
                expected_return = expected_return * gamma2 ** b + critic1
                expected_return2 = expected_return2 * gamma ** b + critics2[i_]
                
                Action_Vector = torch.zeros(64)
                Action_Vector[action1-1] = True
                Action_Vector2 = torch.zeros(64)
                Action_Vector2[action2-1] = True
                ppo_loss += lossfn2(action_probs[i_].cpu(),Action_Vector.cpu())
                ppo_loss += lossfn2(action_probs2[i_].cpu(),Action_Vector2.cpu())
                
                e_loss += lossfn(expected_return.cpu(),Return.cpu()).detach()
                i_loss += lossfn(expected_return2.cpu(),Return2.cpu()).detach()
                
                i_ -= 1
                b += 1

            i_ = 0

            losses = F.normalize(torch.Tensor(losses),dim=0)
            AC.Action_Sample = [0 for x in range(64)]
            AC.Action_Sample2 = [0 for x in range(64)]
            AC.Value_Sum = [0 for x in range(64)]
            AC.Value_Sum2 = [0 for x in range(64)]
            AC.N_visits = 0
            
            for loss_ in losses:
                loss_.requires_grad = True
                loss2 += loss_

            old_probs = curr_probs
            old_probs2 = curr_probs2
            
            ppo_losses.append(ppo_loss.item())

            extrinsic_returns.append(Return)
            intrinsic_returns.append(Return2)
            
            plt.plot(ppo_losses)
            plt.savefig("PPO Loss.png")
            plt.clf()
            
            #Optimization Stage:
            AC.optimizer3.zero_grad()
            ppo_loss.backward()
            AC.optimizer3.step()
            
            curiosity.zero_grad()
            loss2.backward()
            curiosity.step()

            AC.optimizer1.zero_grad()
            e_loss.requires_grad = True
            e_loss.backward()
            AC.optimizer1.step()

            AC.optimizer2.zero_grad()
            i_loss.requires_grad = True
            i_loss.backward()
            AC.optimizer2.step()

            #Reset Stage:
            curr_probs.clear()
            curr_probs2.clear()
            
            critics.clear()
            critics2.clear()
            
            losses = []
            entropies.clear()
            entropies2.clear()
            action_probs.clear()
            action_probs2.clear()
            net_reward = 0
            updated_parameters = True
            done = False
            
            print("Epoch:" + str(epoch_counter))

def test(OptionCritic1,world,level):
    env = gym_super_mario_bros.make('SuperMarioBros-'+str(world)+'-'+str(level)+'-v1')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    img = env.reset()
    img2 = downscale_obs(img)
    dynamic_movement = deque([img2,img2,img2],maxlen=3)
    dynamic_inpt = torch.Tensor(np.array(dynamic_movement))
    net_reward = 0
    while True:
        probs,critic,critic2 = OptionCritic1(dynamic_inpt.to('cuda'))
        #Choose best Action
        action1 = np.argmax(probs.cpu().detach().numpy())
        img, reward, done, info = env.step(action1)
        env.render()
        net_reward += reward
        if done:
            img = env.reset()
            img2 = downscale_obs(img)
            dynamic_movement = deque([img2,img2,img2],maxlen=3)
            dynamic_inpt = torch.Tensor(np.array(dynamic_movement))
            net_reward = 0
        dynamic_movement.append(downscale_obs(img))
        dynamic_inpt = torch.Tensor(np.array(dynamic_movement))

#-------------------------------------->In-case I want to do multiprocessing with the AlphaZero AI
processes = []
if __name__ == "__main__":
    epochs = 5000
    AC = ActorCritic().to('cuda')
    pred = predictor_net().to('cuda')
    target = target_net().to('cuda')
    AC.share_memory()
    pred.share_memory()
    target.share_memory()
    Counter = mp.Value('f',0)
    train(epochs,AC,pred,target,Counter)
