#This is an attempt at implementing the AlphaZero paper
#My goal is to reproduce as close as possible the experimental results produced by the paper
#AlphaZero Implementation
#2024-05-15

#Break-up your implementation in mangeable steps
#First-thing : I need a Chess Simulator
#Okay, I installed a chess simulator called python-chess
#Let's import the library because we will obviously use it

import chess

#Okay, I've imported the chess library
#Now, I will proceed to import as many useful deep learning libraries within this program
#I will also be conscious of the fact that I will need to analyze my experiments with this program
#effectively, therefore a good set of visualization tools will be very useful

#deep learning libraries
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
#Data-structure libraries:
from collections import deque
import numpy as np
#Randomness Library:
from random import shuffle

#Plotting library
#I can use the plotting library in order to visualize the probabilities of certain actions of the AI(Artificial Intelligence)
import matplotlib.pyplot as plt

#Image-processing library:
from skimage.transform import resize

#----------------------->In-case I want to introduce intrinsic motivation to the AI Algorithm 
#Fixed Net architecture
class target_net(nn.Module):
    def __init__(self):
        super(target_net,self).__init__()
        #3 Layer Deep Neural Network
        self.conv1 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.linear = nn.Linear(3072,512)
    def forward(self,x):
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
        self.conv1 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.linear = nn.Linear(3072,512)
        self.linear2 = nn.Linear(512,512)
        self.linear3 = nn.Linear(512,512)
        
    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.selu(self.conv1(x))
        y = F.selu(self.conv2(y))
        y = F.selu(self.conv3(y))
        y = F.selu(self.linear(y.flatten()))
        y = F.selu(self.linear2(y))
        y = self.linear3(y)
        return y

def train_predictor(obs,predictor,target):
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
        self.conv1 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.linear1 = nn.Linear(3072,512)
        self.linear = nn.Linear(512,aspace)
        self.action = None
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
        y = F.softmax(self.linear(y.flatten()))
        self.action = np.random.choice(a=range(self.aspace),p=y.cpu().detach().numpy())
        return y

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
        self.conv1 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.linear1 = nn.Linear(3072,512)
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

#Generalized Option Module, with K options, and a critic function
class ActorCritic(nn.Module):
    
    def __init__(self,aspace=12,k_options=5):
        super(ActorCritic,self).__init__()
        self.action = None
        self.aspace = aspace
        self.k_options = k_options
        self.Critic = Critic().to('cuda').share_memory()
        self.Critic2 = Critic().to('cuda').share_memory()
        self.optimizer1 = optim.Adam(self.Critic.parameters(),lr=1e-3)
        self.optimizer2 = optim.Adam(self.Critic2.parameters(),lr=1e-3)
        self.Dynamics_net = dynamics_net().to('cuda').share_memory()
        self.
        self.Value_Sums = [0 for x in range(self.aspace)]
        self.N_visits = 0
        self.Action_Sample = [0 for x in range(self.aspace)]
        self.C = float(2 ** 0.5)
        self.best_node = None
        self.optimizer4 = optim.Adam(self.Controller.parameters(),lr=1e-4)
        self.planning = True
        self.ucb_scores = None

    def get_ucb(self,value,visits,samples,c):
        if samples == 0 or visits == 0:
            return float('inf')
        else:
            ucb = (value/samples) + c * sqrt((log(visits)/samples))
            return ucb

    #Options keep running once selected, until they get terminated
    #Other options get selected when the previous options gets terminated
    def forward(self,x,planning = True):
        #Now, time to implement the planning stage...

        #If planning is enabled:
        if self.planning == True:
            if self.running == False:
                x = self.Repr(x).reshape(3,32,32).detach()
                self.representation = x
                probs2 = self.Controller(x)
                self.decision = self.Controller.action
                #Now, we have selected an option, time to use it
                self.Option_ = self.Options[self.decision]
                self.optimizer3 = optim.Adam(list(self.Option_.parameters())+list(self.Repr.parameters())+list(self.MasterPlanner.parameters()),lr=1e-4)
                probs_ = self.Option_.Decider(x)
                #Initially, the Controller can override the initial termination
                self.running = True
                self.action = self.Option_.Decider.action
                critic = self.Critic(x)
                critic2 = self.Critic2(x)
                #Next hidden state prediction
                expected_rewards = []
                self.action = self.Option_.Decider.action
                self.Initial_Probs = probs_
                ucb_scores = []
                for i in range(self.aspace):
                    action_vector = torch.zeros(32,32)
                    action_vector[0][i] = True
                    reward_inpt_ = deque([self.representation[0].cpu(),self.representation[1].cpu(),self.representation[2].cpu(),action_vector],maxlen=4)
                    reward_inpt = torch.Tensor(np.array(reward_inpt_))
                    #Expected extrinsic and intrinsic reward
                    expected_rwrd = self.Reward_Net(reward_inpt.to('cuda'))
                    self.Value_Sums[i] += expected_rwrd[0].item() + expected_rwrd[1].item()
                    ucb_score = self.get_ucb(self.Value_Sums[i],self.N_visits,self.Action_Sample[i],self.C)
                    ucb_scores.append(ucb_score)
                    expected_rewards.append(expected_rwrd)
                self.ucb_scores = ucb_scores
                self.Initial_Controller_Probs = probs2
                self.Initial_Critics_E.append(critic)
                self.Initial_Critics_I.append(critic2)
                action = np.argmax(np.array(ucb_scores))
                self.Action_Sample[action] += 1
                self.N_visits += 1
                #Copy this initial set of decisions
                #After processing all the expected rewards, it's time to compute the nodes running sums(How good the imagined line is)
                action_vector = torch.zeros(32,32)
                action_vector[0][action] = True
                dynamics_inpt_ = deque([self.representation[0].cpu(),self.representation[1].cpu(),self.representation[2].cpu(),action_vector],maxlen=4)
                dynamics_inpt = torch.Tensor(np.array(dynamics_inpt_))
                dynamics_inpt[-1] = action_vector
                x_next = self.Dynamics_net(dynamics_inpt.to('cuda'))
                self.steps += 1
                self.forward(x_next.reshape(3,32,32),self.planning)
            else:
                x = self.Repr(x).reshape(3,32,32).detach().to('cuda')
                self.representation = x
                probs = self.Option_(x)
                probs_1 = self.Option_.Decider(x)
                self.MasterPlanner(x)
                end = self.MasterPlanner.action
                critic = self.Critic(x)
                critic2 = self.Critic2(x)
                if self.Option_.action == None and end == 1:
                    #select a new option
                    probs2 = self.Controller(x)
                    self.decision = self.Controller.action
                    #Now, we have selected an option, time to use it
                    self.Option_ = self.Options[self.decision]
                    self.optimizer3 = optim.Adam(list(self.Option_.parameters())+list(self.Repr.parameters())+list(self.MasterPlanner.parameters()),lr=1e-4)
                    probs = self.Option_.Decider(x)
                    if self.run_again == True:
                        self.Initial_Probs = probs
                        self.Initial_Controller_Probs = probs2
                        self.run_again = False
                    #Initially, the Controller can override the initial termination
                    self.action = self.Option_.Decider.action
                    expected_rewards = []
                    ucb_scores = []
                    for i in range(self.aspace):
                        action_vector = torch.zeros(32,32)
                        action_vector[0][i] = True
                        reward_inpt_ = deque([self.representation[0].cpu(),self.representation[1].cpu(),self.representation[2].cpu(),action_vector],maxlen=4)
                        reward_inpt = torch.Tensor(np.array(reward_inpt_))
                        #Expected extrinsic and intrinsic reward
                        expected_rwrd = self.Reward_Net(reward_inpt.to('cuda'))
                        self.Value_Sums[i] += expected_rwrd[0].item() + expected_rwrd[1].item()
                        ucb_score = self.get_ucb(self.Value_Sums[i],self.N_visits,self.Action_Sample[i],self.C)
                        ucb_scores.append(ucb_score)
                        expected_rewards.append(expected_rwrd)
                    self.ucb_scores = ucb_scores
                    action = np.argmax(np.array(ucb_scores))
                    #Copy this initial set of decisions
                    #After processing all the expected rewards, it's time to compute the nodes running sums(How good the imagined line is)
                    self.Action_Sample[action] += 1
                    self.N_visits += 1
                    action_vector = torch.zeros(32,32)
                    action_vector[0][action] = True
                    dynamics_inpt_ = deque([self.representation[0].cpu(),self.representation[1].cpu(),self.representation[2].cpu(),action_vector],maxlen=4)
                    dynamics_inpt = torch.Tensor(np.array(dynamics_inpt_))
                    dynamics_inpt[-1] = action_vector
                    x_next = self.Dynamics_net(dynamics_inpt.to('cuda'))
                    self.steps += 1
                    self.forward(x_next.reshape(3,32,32),self.planning)

                elif self.Option_.action != None and end == 1:
                    self.action = self.Option_.action
                    expected_rewards = []
                    ucb_scores = []
                    if self.run_again == True:
                        self.Initial_Probs = probs
                        self.run_again = False
                    for i in range(self.aspace):
                        action_vector = torch.zeros(32,32)
                        action_vector[0][i] = True
                        reward_inpt_ = deque([self.representation[0].cpu(),self.representation[1].cpu(),self.representation[2].cpu(),action_vector],maxlen=4)
                        reward_inpt = torch.Tensor(np.array(reward_inpt_))
                        #Expected extrinsic and intrinsic reward
                        expected_rwrd = self.Reward_Net(reward_inpt.to('cuda'))
                        self.Value_Sums[i] += expected_rwrd[0].item() + expected_rwrd[1].item()
                        ucb_score = self.get_ucb(self.Value_Sums[i],self.N_visits,self.Action_Sample[i],self.C)
                        ucb_scores.append(ucb_score)
                        expected_rewards.append(expected_rwrd)
                    self.ucb_scores = ucb_scores
                    action = np.argmax(np.array(ucb_scores))
                    self.N_visits += 1
                    #Copy this initial set of decisions
                    #After processing all the expected rewards, it's time to compute the nodes running sums(How good the imagined line is)
                    self.Action_Sample[action] += 1
                    action_vector = torch.zeros(32,32)
                    action_vector[0][action] = True
                    dynamics_inpt_ = deque([self.representation[0].cpu(),self.representation[1].cpu(),self.representation[2].cpu(),action_vector],maxlen=4)
                    dynamics_inpt = torch.Tensor(np.array(dynamics_inpt_))
                    dynamics_inpt[-1] = action_vector
                    x_next = self.Dynamics_net(dynamics_inpt.to('cuda'))
                    self.steps += 1
                    self.forward(x_next.reshape(3,32,32),self.planning)
                else:
                    self.best_node = np.argmax(np.array(self.ucb_scores))
                    best_node = np.random.choice(a=range(self.aspace),p=self.Initial_Probs.cpu().detach().numpy())
                    best_value = self.Value_Sums[self.best_node]
                    #Select the best action
                    best_action = best_node
                    self.action = best_node
                    #Select the best action's probability
                    probs = self.Initial_Probs
                    self.probs__ = probs
                    #Select the Critic of that state:
                    critic = self.Initial_Critics_E[0]
                    critic2 = self.Initial_Critics_I[0]
                    self.critic_ = critic
                    self.critic__ = critic2
                    self.Controller_probs = self.Initial_Controller_Probs
                    self.planning = False
                    
        if self.planning == False:
            self.Value_Sums = [0 for x in range(self.aspace)]
            self.steps = 0
            self.N_visits = 0
            self.Action_Sample = [0 for x in range(self.aspace)]
            self.run_again = True
            return self.probs__,self.critic_,self.critic__
            
    #Now, change the loss function
    def loss(self,prob1,old_prob,adv,eps=0.2):
        prob_ratio = prob1/old_prob
        comp1 = prob_ratio * adv
        comp2 = torch.clip(prob_ratio,1-eps,1+eps)*adv
        loss = min(comp1,comp2)
        return loss

board = chess.Board()

#Utility function in order to encode the chess board and also decode the chessboard
class utilities:

    #Encoder function
    def encode_info(chessboard):
        piece_encoded = []
        pieces_selected = []
        found_piece = 0
        counta = 1
        for i in range(63):
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
        for i in range(63):
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
    actions = []
    best_nodes = []
    losses = []
    
    updated_parameters = False
    next_time = False
    
    lossfn = nn.MSELoss()
    lossfn2 = nn.CrossEntropyLoss()
    
    curiosity = optim.Adam(pred.parameters(),lr=1e-4)
    
    #prev_lives = env.ale.lives()
    while epoch_counter != epochs:
        #Get the probabilities and critic value from the Actor Critic
        probs,critic_,critic_2 = AC(dynamic_inpt.to('cuda'),True)
        best_nodes.append(OC.best_node)
        OC.planning = True
        actions.append(OC.action)
        #Get the next hidden state prediction:
        representations.append(OC.Repr(dynamic_inpt.to('cuda')).reshape(3,32,32).detach())
        
        #Action is already selected
        action1 = OC.action
        #Execute the action
        img, reward, done, info = env.step(action1)
        env.render()
        
        action_vector = torch.zeros(32,32)
        action_vector[0][action1] = True
        #Just going to borrow the dynamic inpt data :
        rn_input[3] = action_vector
        rn_input[0] = OC.Repr(dynamic_inpt.to('cuda')).reshape(3,32,32)[0].detach().cpu()
        rn_input[1] = OC.Repr(dynamic_inpt.to('cuda')).reshape(3,32,32)[1].detach().cpu()
        rn_input[2] = OC.Repr(dynamic_inpt.to('cuda')).reshape(3,32,32)[2].detach().cpu()
        rn_inpt = torch.Tensor(np.array(rn_input))
        expected_rew = OC.Reward_net(rn_inpt.to('cuda'))
        expected_rewards.append(expected_rew)

        dynamics_inpt[3] = action_vector
        dynamics_inpt[0] = OC.Repr(dynamic_inpt.to('cuda')).reshape(3,32,32)[0].detach().cpu()
        dynamics_inpt[1] = OC.Repr(dynamic_inpt.to('cuda')).reshape(3,32,32)[1].detach().cpu()
        dynamics_inpt[2] = OC.Repr(dynamic_inpt.to('cuda')).reshape(3,32,32)[2].detach().cpu()
        dynamic_output = OC.Dynamics_net(dynamics_inpt.to('cuda'))
        dynamic_outputs.append(dynamic_output)
        
        actions_taken.append(action1)
        
        #plt.bar(support,critic_[action1].detach().numpy())
        #plt.savefig("Actions(Extrinsic)(SuperMarioBros)/" + str(env.get_action_meanings()[action1]) + ".png")
        #plt.clf()
        #plt.bar(support2,critic_2[action1].detach().numpy())
        #plt.savefig("Actions(Intrinsic)(SuperMarioBros)/" + str(env.get_action_meanings()[action1]) + ".png")
        #plt.clf()
        
        critics.append(critic_)
        critics2.append(critic_2)
        
        curr_prob = probs[action1]
        curr_probs.append(curr_prob)

        #Controller:
        curr_prob2 = OC.Controller_probs[OC.decision]
        curr_probs2.append(curr_prob2)
        
        entropy = Categorical(probs).entropy()
        entropies.append(entropy)
        
        entropy2 = Categorical(OC.Controller_probs).entropy()
        entropies2.append(entropy2)
        
        state = dynamic_inpt
        prob_actions.append(probs[action1])
        state = dynamic_inpt
        dynamic_movement.append(downscale_obs(img))
        dynamic_inpt = torch.Tensor(np.array(dynamic_movement))
        state2 = dynamic_inpt
        
        loss = train_predictor(dynamic_inpt,pred,target)
        i_reward = loss
        
        episode_length += 1
        lifespan += 1

        if episode_length == maxlength:
            done = True
    
        elif not done: #and not env.ale.lives() < prev_lives and reward != 0:
            rewards.append(torch.sigmoid(torch.Tensor([reward])))
            losses.append(float(i_reward))
            net_reward += i_reward.item()
            net_reward += reward
            
        if done: #or env.ale.lives() < prev_lives:
            rewards.append(torch.sigmoid(torch.Tensor([-1])))
            losses.append(0)
            net_reward += (-done)
            epoch_counter += 1
            Counter.value += 1
            #if done:
            img = env.reset()
            img2 = downscale_obs(img)
            dynamic_movement = deque([img2,img2,img2],maxlen=3)
            dynamic_inpt = torch.Tensor(np.array(dynamic_movement))
            episode_length = 0
            ppo_loss = 0
            controller_loss = 0
            rn_loss = 0
            mcts_loss = 0
            dynamics_loss = 0
            i_loss = torch.Tensor([0])
            e_loss = torch.Tensor([0])
            i_ = len(critics) - 1
            loss2 = 0
            expected_return = 0
            expected_return2 = 0
            Return = 0
            Return2 = 0
            b = 0
            #prev_lives = env.ale.lives()
            if updated_parameters == False:
                for i in range(len(curr_probs)):
                    old_probs.append(curr_probs[i])
                for i in range(len(curr_probs2)):
                    old_probs2.append(curr_probs2[i])

            #Compute the PPO Los,the Intrinsic Loss and the Extrinsic Loss
            
            for critic1 in reversed(critics):
                #Turned off extrinsic rewards...
                #Calculate the actual net return
                Return = Return * gamma2 ** b + rewards[i_]  
                Return2 = Return2 * gamma ** b + torch.Tensor([losses[i_]]) 
                
                #Calculate the cumulative expected return
                expected_return = expected_return * gamma2 ** b + expected_rewards[i_][0]
                expected_return2 = expected_return2 * gamma ** b + expected_rewards[i_][1]
                
                adv = (expected_return.cpu() - Return) + (expected_return2.detach().cpu() - Return2)
                
                controller_loss += OC.loss(curr_probs2[i_],old_probs2[i_],adv.to('cuda')).detach() - 0.01 * entropies2[i_].detach()
                action_vector = torch.zeros(32,32)
                action_vector[0][actions[i_]] = True
                action_vector2 = torch.zeros(32,32)
                action_vector2[0][best_nodes[i_]] = True
                mcts_loss += lossfn2(action_vector,action_vector2)
                
                i_ -= 1
                b += 1

            i_ = 0

            #Teach the Reward Network
            for reward in rewards:
                actual_reward_vector = torch.Tensor([reward,losses[i_]]).to('cuda')
                rn_loss += lossfn(expected_rewards[i_],actual_reward_vector)
                i_ += 1
            i_ = 0
            #Teach the Dynamics Network
            for i in range(len(representations)):
                #Make sure to correctly model the representation:
                if i + 1 <= len(representations) - 1:
                    dynamics_loss += lossfn(dynamic_outputs[i].reshape(3072),representations[i+1].reshape(3072).detach())

            if high_score < net_reward:
                high_score = net_reward
                maxlength = int(high_score) + 15

            losses = F.normalize(torch.Tensor(losses),dim=0)
            for loss_ in losses:
                loss_.requires_grad = True
                loss2 += loss_

            old_probs = curr_probs
            old_probs2 = curr_probs2
            controller_losses.append(controller_loss.item())
            ppo_loss += mcts_loss
            ppo_losses.append(ppo_loss.item())
            rn_losses.append(rn_loss.item())
            extrinsic_returns.append(Return)
            intrinsic_returns.append(Return2)
            dynamics_losses.append(dynamics_loss.cpu().detach().numpy())
            
            plt.plot(ppo_losses)
            plt.savefig("PPO Loss.png")
            plt.clf()
            
            plt.plot(controller_losses)
            plt.savefig('Controller Loss.png')
            plt.clf()
            
            plt.plot(losses[:len(losses)-1].detach().numpy())
            plt.savefig("Curiosity Spikes.png")
            plt.clf()
            
            plt.plot(extrinsic_returns)
            plt.savefig("Extrinsic Returns.png")
            plt.clf()
            
            plt.plot(intrinsic_returns)
            plt.savefig("Intrinsic Returns.png")
            plt.clf()

            plt.plot(rn_losses)
            plt.savefig("Reward Network Loss.png")
            plt.clf()

            plt.plot(dynamics_losses)
            plt.savefig("Dynamics Network Loss.png")
            plt.clf()
            
            #Optimization Stage:
            OC.optimizer3.zero_grad()
            ppo_loss.requires_grad = True
            ppo_loss.backward()
            OC.optimizer3.step()

            OC.optimizer4.zero_grad()
            controller_loss.requires_grad = True
            controller_loss.backward()
            OC.optimizer4.step()
            
            curiosity.zero_grad()
            loss2.backward()
            curiosity.step()

            Reward_optimizer.zero_grad()
            rn_loss.backward()
            Reward_optimizer.step()

            dynamic_optimizer.zero_grad()
            dynamics_loss.backward()
            dynamic_optimizer.step()

            #Reset Stage:
            curr_probs.clear()
            curr_probs2.clear()
            critics.clear()
            critics2.clear()
            prob_actions.clear()
            rewards.clear()
            rewards2.clear()
            losses = []
            entropies.clear()
            entropies2.clear()
            actions_taken.clear()
            expected_rewards.clear()
            dynamic_outputs.clear()
            actions.clear()
            best_nodes.clear()
            net_reward = 0
            representations.clear()
            updated_parameters = True
            next_time = False
            
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
    OptionCritic1 = OptionCritic(aspace=12,k_options=1).to('cuda')
    pred = predictor_net()
    target = target_net()
    OptionCritic1.share_memory()
    pred.share_memory()
    target.share_memory()
    Counter = mp.Value('f',0)
    world = 3
    level = 3
    train(epochs,OptionCritic1,pred,target,Counter,world,level)
    for i in range(0):
        p = mp.Process(target=train,args=(epochs,OptionCritic1,pred,target,Counter,world,level)) 
        p.start()
        processes.append(p)
    for p in processes: 
        p.join()
    for p in processes: 
        p.terminate()
