import pdb
import random
import numpy as np
from dist import uniform_dist, delta_dist, mixture_dist,DDist
from util import argmax_with_val, argmax
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



class MDP:
    # Needs the following attributes:
    # states: list or set of states
    # actions: list or set of actions
    # discount_factor: real, greater than 0, less than or equal to 1
    # start: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    # These are functions:
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward

    def __init__(self, states, actions, transition_model, reward_fn,
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

    def state2vec(self, s):
        '''
        Return one-hot encoding of state s; used in neural network agent implementations
        '''
        v = np.zeros((1, len(self.states)))
        v[0,self.states.index(s)] = 1.
        return v

# Perform value iteration on an MDP, also given an instance of a q
# function.  Terminate when the max-norm distance between two
# successive value function estimates is less than eps.
# interactive_fn is an optional function that takes the q function as
# argument; if it is not None, it will be called once per iteration,
# for visuzalization

# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values This must be
# initialized before interactive_fn is called the first time.

#def value_iteration(mdp, q, eps = 0.01, max_iters = 1000):
def value_iteration(mdp, q, eps = 0.05, max_iters = 1000):

    # Your code here (COPY FROM HW9)
    gamma = mdp.discount_factor
    R = mdp.reward_fn
    T = mdp.transition_model

    #Assumes that Q has already been initialised
    while True:

        q_new = q.copy()
        i=0
        #print(len(q.states))
        for s in q.states:
            #i+=1
            #if i%100 == 0: print(i)
            for a in q.actions:
                expected_reward = 0
                for s_ in q.states:
                    #expected_reward += T(s,a).prob(s_) * R(s_,greedy(q,s_))
                    max_q_a = max([q.get(s_,a) for a in q.actions])
                    expected_reward += T(s,a).prob(s_) * max_q_a
                new_value = R(s,a) +gamma*expected_reward
                q_new.set(s,a,new_value)

        #find the maximum difference between old and new values
        diffs = [abs(q.get(s,a)-q_new.get(s,a)) for s in q.states for a in q.actions]
        if max(diffs) < eps:
            return q_new
        else:
            print(f"******** max: {max(diffs)}")
            q = q_new.copy()

# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    # Your code here (COPY FROM HW9)
    return max(q.get(s,a)for a in q.actions)

# Given a state, return the action that is greedy with reespect to the
# current definition of the q function
def greedy(q, s):
    rewards = [q.get(s,a)for a in q.actions]
    return q.actions[np.argmax(rewards)]

def epsilon_greedy(q, s, eps = 0.5):
    if random.random() < eps:  # True with prob eps, random action
        # Your code here
        return uniform_dist(q.actions).draw()
    else:
        return greedy(q, s)

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]
    def update(self, data, lr):
        for item in data:
            s,a,t = item
            self.set(s,a,self.get(s,a) + lr*(t-self.get(s,a)))
        
"""
q = TabularQ([0,1,2,3],['b','c'])
q.update([(0, 'b', 50), (2, 'c', 30)], 0.5)
print(f"{q.get(0, 'b')} - should be 25.0")
print(f"{q.get(2, 'c')} - should be 15.0")
"""

def Q_learn(mdp, q, lr=.1, iters=100, eps = 0.5, interactive_fn=None):

    s = mdp.init_state()
    for i in range(iters):
        # include this line in the iteration, where i is the iteration number
        a = epsilon_greedy(q,s,eps)
        reward,s_ = mdp.sim_transition(s,a)
        # print(reward)
        # print(s_)
        # print(value(q,s_))
        # print(mdp.discount_factor)
        future_val = 0 if mdp.terminal(s) else value(q, s_)
        q.update([(s, a, (reward + mdp.discount_factor * future_val))], lr)
        s=s_
        if interactive_fn: interactive_fn(q, i)

    return q


###########################################
def tinyTerminal(s):
    return s==4
def tinyR(s, a):
    if s == 1: return 1
    elif s == 3: return 2
    else: return 0
def tinyTrans(s, a):
    if s == 0:
        if a == 'a':
            return DDist({1 : 0.9, 2 : 0.1})
        else:
            return DDist({1 : 0.1, 2 : 0.9})
    elif s == 1:
        return DDist({1 : 0.1, 0 : 0.9})
    elif s == 2:
        return DDist({2 : 0.1, 3 : 0.9})
    elif s == 3:
        return DDist({3 : 0.1, 0 : 0.5, 4 : 0.4})
    elif s == 4:
        return DDist({4 : 1.0})


# def testQ():
#     tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
#     tiny.terminal = tinyTerminal
#     q = TabularQ(tiny.states, tiny.actions)
#     qf = Q_learn(tiny, q)
#     return list(qf.q.items())

# random.seed(0)
# testQ()
# print("should be")
# print([((0, 'a'), 0.6649739221724159), ((0, 'b'), 0.1712369526453748), ((1, 'a'), 0.7732751316011999), ((1, 'b'), 1.2034912054227331), ((2, 'a'), 0.37197205380133874), ((2, 'b'), 0.45929063274463033), ((3, 'a'), 1.5156163024818292), ((3, 'b'), 0.8776852768653631), ((4, 'a'), 0.0), ((4, 'b'), 0.0)])


###########################################


# Simulate an episode (sequence of transitions) of at most
# episode_length, using policy function to select actions.  If we find
# a terminal state, end the episode.  Return accumulated reward a list
# of (s, a, r, s') where s' is None for transition from terminal state.
# Also return an animation if draw=True.
def sim_episode(mdp, episode_length, policy, draw=False):
    episode = []
    reward = 0
    s = mdp.init_state()
    all_states = [s]
    for i in range(int(episode_length)):
        a = policy(s)
        (r, s_prime) = mdp.sim_transition(s, a)
        reward += r
        if mdp.terminal(s):
            episode.append((s, a, r, None))
            break
        episode.append((s, a, r, s_prime))
        if draw: 
            mdp.draw_state(s)
        s = s_prime
        all_states.append(s)
    animation = animate(all_states, mdp.n, episode_length) if draw else None
    return reward, episode, animation

# Create a matplotlib animation from all states of the MDP that
# can be played both in colab and in local versions.
def animate(states, n, ep_length):
    try:
        from matplotlib import animation, rc
        import matplotlib.pyplot as plt
        from google.colab import widgets

        plt.ion()
        plt.figure(facecolor="white")
        fig, ax = plt.subplots()
        plt.close()

        def animate(i):
            if states[i % len(states)] == None or states[i % len(states)] == 'over':
                return
            ((br, bc), (brv, bcv), pp, pv) = states[i % len(states)]
            im = np.zeros((n, n+1))
            im[br, bc] = -1
            im[pp, n] = 1
            ax.cla()
            ims = ax.imshow(im, interpolation = 'none',
                        cmap = 'viridis', 
                        extent = [-0.5, n+0.5,
                                    -0.5, n-0.5],
                        animated = True)
            ims.set_clim(-1, 1)
        rc('animation', html='jshtml')
        anim = animation.FuncAnimation(fig, animate, frames=ep_length, interval=100)
        return anim
    except:
        # we are not in colab, so the typical animation should work
        return None

# Return average reward for n_episodes of length episode_length
# while following policy (a function of state) to choose actions.
def evaluate(mdp, n_episodes, episode_length, policy):
    score = 0
    length = 0
    for i in range(n_episodes):
        # Accumulate the episode rewards
        r, e, _ = sim_episode(mdp, episode_length, policy)
        score += r
        length += len(e)
        # print('    ', r, len(e))
    return score/n_episodes, length/n_episodes

def Q_learn_batch(mdp, q, lr=.1, iters=100, eps=0.5,
                  episode_length=10, n_episodes=2,
                  interactive_fn=None):
    # Your code here
    all_episodes=[]
    for i in range(iters):
        for _ in range(n_episodes):
            reward,episode,animation = sim_episode(mdp,episode_length,lambda s:epsilon_greedy(q, s, eps))
            #episode.append((s, a, r, s_prime)) OR episode.append((s, a, r, None))
            all_episodes.append(episode)
        
        #convert all_experiences rewards into T using the existing Q values
        updates=[]
        for episode in all_episodes:
            for experience in episode:
                s,a,r,s_ = experience #NB s_ will be None if terminal
                if s_ is None:
                    future_value = 0
                else:
                    future_value = value(q,s_)
                updates.append((s, a, (r + mdp.discount_factor * future_value)))

        #run 1 large Q update
        q.update(updates,lr)

        # include this line in the iteration, where i is the iteration number
        if interactive_fn: interactive_fn(q, i)
    return q

def testBatchQ():
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = TabularQ(tiny.states, tiny.actions)
    qf = Q_learn_batch(tiny, q)
    return list(qf.q.items())

""" random.seed(0)
print(testBatchQ())
print("^^^^^^^^ should be:")
print([((0, 'a'), 4.7566600197286535), ((0, 'b'), 3.993296047838986), ((1, 'a'), 5.292467934685342), ((1, 'b'), 5.364014782870985), ((2, 'a'), 4.139537149779127), ((2, 'b'), 4.155347555640753), ((3, 'a'), 4.076532544818926), ((3, 'b'), 4.551442974149778), ((4, 'a'), 0.0), ((4, 'b'), 0.0)]) """

def make_nn(state_dim, num_hidden_layers, num_units):
    model = Sequential()
    model.add(Dense(num_units, input_dim = state_dim, activation='relu'))
    for i in range(num_hidden_layers-1):
        model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model

class NNQ:
    def __init__(self, states, actions, state2vec, num_layers, num_units, epochs=1):
        self.actions = actions
        self.states = states
        self.epochs = epochs
        self.state2vec = state2vec
        state_dim = state2vec(states[0]).shape[1] # a row vector
        self.models = {a:make_nn(state_dim, num_layers, num_units) for a in actions}
    def get(self, s, a):
        return self.models[a].predict(self.state2vec(s))
    def update(self, data, lr):
        for a in self.actions:
            if [s for (s, at, t) in data if a==at]:
                X = np.vstack([self.state2vec(s) for (s, at, t) in data if a==at])
                Y = np.vstack([t for (s, at, t) in data if a==at])
                self.models[a].fit(X, Y, epochs = self.epochs, verbose = False)
    def __init__(self, states, actions, state2vec, num_layers, num_units, epochs=1):
        self.actions = actions
        self.states = states
        self.state2vec = state2vec
        #NB dimension of the neural network will be the size of the one-hot encoding (altho this is the same as len(states))
        state_dim = state2vec(states[0]).shape[1]
        self.epochs = epochs
        self.models = {}
        #one NN per action
        for action in self.actions:
            self.models[action]=make_nn(state_dim,num_layers,num_units)


    def get(self, s, a):
        # Your code here
        return self.models[a].predict(self.state2vec(s))[0][0]

    def update(self, data, lr, epochs = 1):
        # extract relevant data for each action
        action_data_x={}
        action_data_y={}
        for action in self.actions:
            action_data_x[action]=[] #[X],[Y]
            action_data_y[action]=[] #[X],[Y]
        
        for point in data:
            s,a,t = point
            #action_data_x[a].append(self.state2vec(s))
            action_data_x[a].append(self.state2vec(s)[0])
            action_data_y[a].append(t)

        #for action in self.actions:
        #    if len(action_data_x[action]) > 0:
        #        self.models[action].fit(np.array(action_data_x[action]), np.array(action_data_y[action]), epochs=epochs)

        for a in self.actions:
            if [s for (s, at, t) in data if a==at]:
                X = np.vstack([self.state2vec(s) for (s, at, t) in data if a==at])
                Y = np.vstack([t for (s, at, t) in data if a==at])
                self.models[a].fit(X, Y, epochs = self.epochs, verbose = False)

def test_NNQ(data):
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = NNQ(tiny.states, tiny.actions, tiny.state2vec, 2, 10)
    q.update(data, 1)
    return [q.get(s,a) for s in q.states for a in q.actions]

#print("VVVV test_NNQ VVVV")
#print(test_NNQ([(0,'a',0.3),(1,'a',0.1),(0,'a',0.1),(1,'a',0.5)]))


class mit_NNQ:
    def __init__(self, states, actions, state2vec, num_layers, num_units, epochs=1):
        self.actions = actions
        self.states = states
        self.epochs = epochs
        self.state2vec = state2vec
        state_dim = state2vec(states[0]).shape[1] # a row vector
        self.models = {a:make_nn(state_dim, num_layers, num_units) for a in actions}
    def get(self, s, a):
        return self.models[a].predict(self.state2vec(s))
    def update(self, data, lr):
        for a in self.actions:
            if [s for (s, at, t) in data if a==at]:
                X = np.vstack([self.state2vec(s) for (s, at, t) in data if a==at])
                Y = np.vstack([t for (s, at, t) in data if a==at])
                self.models[a].fit(X, Y, epochs = self.epochs, verbose = False)


def test_mit_NNQ(data):
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = NNQ(tiny.states, tiny.actions, tiny.state2vec, 2, 10)
    q.update(data, 1)
    return [q.get(s,a) for s in q.states for a in q.actions]

#print("VVVV test_NNQ VVVV")
#print(test_mit_NNQ([(0,'a',0.3),(1,'a',0.1),(0,'a',0.1),(1,'a',0.5)]))