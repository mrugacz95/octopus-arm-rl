import random
from array import array


class Agent:
    "A template agent acting randomly"

    # name should contain only letters, digits, and underscores (not enforced by environment)
    __name = 'Python_My_Agent'

    def __init__(self, stateDim, actionDim, agentParams):
        "Initialize agent assuming floating point state and action"
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.action = array('d', [0 for x in range(actionDim)])
        self.reward = 0
        self.begin_point = [0, 0]
        self.end_point = [9, -1]
        self.dist_to_end = self.distance(self.begin_point[0], self.begin_point[1], self.end_point[0], self.end_point[1])
        # we ignore agentParams because our agent does not need it.
        # agentParams could be a parameter file needed by the agent.
        random.seed()

    def distance(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def distance_of_the_last_segment(self, state):
        x = state[78]
        y = state[79]
        return self.distance(x, y, self.end_point[0], self.end_point[1])

    def distance_of_the_middle_segment(self, state):
        x = state[18]
        y = state[19]
        return self.distance(x, y, self.end_point[0], self.end_point[1])

    def first_segment_alpha(self, state):
        return state[0]

    def my_action(self, state):
        if self.reward > -0.4:
            for i in range(self.actionDim):
                if i % 6 == 2:
                    self.action[i] = 1
                else:
                    self.action[i] = 0
        else:
            for i in range(self.actionDim):
                if i % 3 == 1:
                    self.action[i] = 1
                else:
                    self.action[i] = 0

    def start(self, state):
        "Given starting state, agent returns first action"
        self.my_action(state)
        return self.action

    def step(self, reward, state):
        "Given current reward and state, agent returns next action"
        if reward == 10:
            self.reward = 0
        self.my_action(state)
        self.end_action()
        return self.action

    def end_action(self):
        self.reward -= 0.01

    def cleanup(self):
        pass

    def getName(self):
        return self.__name
