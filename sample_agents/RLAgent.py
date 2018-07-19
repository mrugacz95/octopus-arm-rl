import os
import pickle
import random
from collections import deque

from keras import Sequential
from keras.layers import Dense, np
from keras.optimizers import Adam


class Agent:
    __name = 'RLAgent'
    MODEL_PATH = "octopus_model.h5"
    HISTORY_PATH = "history"
    IS_LEARNING = 'is_learning'

    def __init__(self, _, action_dim, agent_params):
        """Initialize agent assuming floating point state and action"""
        self.is_learning = agent_params.get(self.IS_LEARNING, True)
        self.end_point = [9, -1]
        self.state_dim = len(self.get_all_features2(list(np.arange(0, 82))))
        self.action_dim = action_dim
        self.action = np.zeros(action_dim)
        self.gamma = 0.9
        self.memory = deque(maxlen=20000)
        self.epsilon_start = 0.8
        if not self.is_learning:
            self.epsilon_start = 0.0
        self.epsilon = self.epsilon_start  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.steps = 0
        self.last_state = None
        self.last_action = None
        self.possible_actions = [
            [0, 0, 0, 0, 0, 0],  # do nothing
            [1, 0, 0, 1, 0, 0],  # ∏
            [0, 0, 1, 0, 0, 1],  # ∐
            [1, 1, 1, 0.5, 0.5, 0.5],  # --
            [0.5, 0, 0, 0, 0, 1],  # Ƨ
            [0, 0, 0.5, 1, 0, 0],  # S
            [0, 0, 1, 0, 1, 0],  # ⊂_
            [1, 0, 0, 0, 1, 0],  # ∩_
        ]
        self.action_size = len(self.possible_actions)  # number of possible actions ZXC + IOP
        self.model = self._build_model()
        self.begin_point = [0, 0]
        self.dist_to_end = self.distance(self.begin_point[0], self.begin_point[1], self.end_point[0], self.end_point[1])
        self.max_repeat = 10
        self.repeat_count = self.max_repeat
        random.seed()

    def _build_model(self):
        # Neural Net for Deep Q Learning
        model = Sequential()
        # Input Layer of state size and Hidden Layer with 24 nodes
        model.add(Dense(28, input_dim=self.state_dim, activation='relu'))
        # Hidden layer with 24 nodes
        model.add(Dense(28, activation='relu'))
        # Output Layer with # of actions: 2 nodes (left, right)
        model.add(Dense(self.action_size, activation='linear'))
        # Create the model based on the information above
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.model = model
        self.load_weights()
        self.load_history()

        return model

    def remember(self, state, action, reward, next_state, done):
        if self.is_learning:
            self.memory.append((state, action, reward, next_state, done))

    def random_action(self):
        return random.randrange(self.action_size)

    def distance(self, x1, y1, x2, y2):
        """Distance between two points"""
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def distance_of_the_last_segment(self, state):
        """Distance between the last segment and the target point"""
        x = np.mean(np.array([state[34], state[38], state[74], state[78]]))
        y = np.mean(np.array([state[35], state[39], state[75], state[79]]))
        return self.distance(x, y, self.end_point[0], self.end_point[1])

    def distance_of_the_middle_segment(self, state):
        """Distance between the middle segment and the target point"""
        x = np.mean(np.array([state[18], state[22], state[62], state[66]]))
        y = np.mean(np.array([state[19], state[23], state[63], state[67]]))
        return self.distance(x, y, self.end_point[0], self.end_point[1])

    def first_segment_angle(self, state):
        """The attachment point angle"""
        return state[0]

    def fragment_gravity_center(self, state, fragment_number, dim):
        """The center of weight of the fragment
        fragment_number: should be in range [0,1]
        dim: should be in ['x','y']"""
        offset = 0 if dim == 'x' else 1
        start = 20 * fragment_number + 2
        up = state[start + offset:start + 20 + offset:4]
        bottom = state[start + offset + 40:start + 60 + offset:4]
        return np.mean(np.array([up, bottom]))

    def fragment_velocity(self, state, fragment_number, dim):
        """The velocity of the fragment
        fragment_number: should be in range [0,1,2]
        dim: should be in ['x','y']"""
        offset = 0 if dim == 'x' else 1
        start = 20 * fragment_number + 4
        up = state[start + offset:start + 20 + offset:4]
        bottom = state[start + offset + 40:start + 60 + offset:4]
        return np.mean(np.array([up, bottom]))

    def fragment_orientation(self, state, fragment_number):
        """The fragment orientation
        fragment_number: should be in range [0,1,2]"""
        ends = [[2, 42], [18, 59], [30, 71], [42, 83]]
        # [2,3], [42,43] and [14,15], [55,56] for first segment
        begin_x = (state[ends[fragment_number][0]] + state[ends[fragment_number][1]]) / 2
        begin_y = (state[ends[fragment_number][0] + 1] + state[ends[fragment_number][1] + 1]) / 2
        end_x = (state[ends[fragment_number + 1][0] - 4] + state[ends[fragment_number + 1][1] - 4]) / 2
        end_y = (state[ends[fragment_number + 1][0] - 3] + state[ends[fragment_number + 1][1] - 3]) / 2
        dx = end_x - begin_x
        dy = end_y - begin_y
        if dx == 0:
            return np.pi / 4
        else:
            return np.arctan(dy / dx)

    def angle_between_fragments2(self, state, fragment_number):
        """Angle between succeeding fragments
        fragment_number: should be in range [0,1,2]
        0 – the attachment point angle
        1 – angle between the first and the second half
        2 – angle between the second half and the target point """
        if fragment_number == 0:
            return self.first_segment_angle(state)
        else:
            a1 = self.fragment_orientation(state, fragment_number - 1)
            a2 = self.fragment_orientation(state, fragment_number)
            if a1 * a2 == -1:
                return np.pi / 4
            else:
                return np.arctan((a1 - a2) / (1 - a1 * a2))

    def get_all_features2(self, state):
        features = [self.fragment_gravity_center(state, 0, 'x'), self.fragment_gravity_center(state, 0, 'y'),
                    self.fragment_gravity_center(state, 1, 'x'), self.fragment_gravity_center(state, 1, 'y'),
                    self.fragment_velocity(state, 0, 'x'), self.fragment_velocity(state, 0, 'y'),
                    self.fragment_velocity(state, 1, 'x'), self.fragment_velocity(state, 1, 'y'),
                    self.distance_of_the_last_segment(state), self.distance_of_the_middle_segment(state),
                    self.first_segment_angle(state), self.angle_between_fragments2(state, 2)]
        return features

    def action_to_output(self, action):
        ret = np.zeros(self.action_dim)
        for idx, move in enumerate(self.possible_actions[action]):
            {
                0: lambda: ret[:self.action_dim // 2][0::3],  # first half upper mussels
                1: lambda: ret[:self.action_dim // 2][1::3],  # first half transverse mussels
                2: lambda: ret[:self.action_dim // 2][2::3],  # first half downer mussels
                3: lambda: ret[self.action_dim // 2:][0::3],  # second half upper mussels
                4: lambda: ret[self.action_dim // 2:][1::3],  # second half transverse mussels
                5: lambda: ret[self.action_dim // 2:][2::3],  # second half downer mussels
            }.get(idx)().fill(move)
        return ret

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            # print('rand: ', end='')
            return self.random_action()
            # Predict the reward value based on the given state

        # print('net: ', end='')
        act_values = self.model.predict(np.array([state]))
        # print(act_values, end=', ')
        # Pick the action based on the predicted reward
        return np.argmax(act_values[0])

    def start(self, state):
        state = self.get_all_features2(state)
        self.action = self.last_action = self.act(state)
        self.last_state = state
        return self.action_to_output(self.action)

    def reward_function(self, distance):
        return max(20, 10 / distance)

    def step(self, reward, state):
        done = reward == 10
        if done and not self.is_learning:
            print("Done")
        if self.repeat_count < self.max_repeat and not done and self.last_action is not None:
            self.repeat_count += 1
            return self.action_to_output(self.last_action)
        distance = self.distance_of_the_last_segment(state)
        state = self.get_all_features2(state)
        self.remember(self.last_state, self.last_action, self.reward_function(distance), state, done)
        self.steps += 1
        if self.steps == 300:
            self.learn(512)
            self.steps = 0
        action = self.act(state)
        self.last_action = action
        if done:
            self.last_action = None
        self.last_state = state
        if self.steps % 1000 == 0:
            self.save_model()
            self.save_history()
        self.repeat_count = 0
        return self.action_to_output(action)

    def getName(self):
        return self.__name

    def learn(self, batch_size):
        if not self.is_learning:
            return
        # Not enough samples
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array([state])
            target = reward + self.gamma * \
                              np.amax(self.model.predict(np.array([next_state]))[0])
            if done:
                target += 10
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(self.epsilon)

    def save_model(self):
        if self.is_learning:
            self.model.save_weights(self.MODEL_PATH)
            print("Model saved")

    def save_history(self):
        if self.is_learning:
            pickle.dump(self.memory, open(self.HISTORY_PATH, 'wb'))
            print("Hist saved")

    def load_weights(self):
        if os.path.isfile(self.MODEL_PATH):
            self.model.load_weights(self.MODEL_PATH)
            print("Model loaded")

    def load_history(self):
        if self.is_learning:
            if os.path.isfile(self.HISTORY_PATH):
                self.memory = pickle.load(open(self.HISTORY_PATH, 'rb'))
                print("History loaded")

    def reset_epsilon(self):
        self.epsilon = 0.5

    def safe_exit(self):
        if self.is_learning:
            print("safe exit")
            self.save_model()
            self.save_history()
