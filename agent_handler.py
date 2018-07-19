# Echo client program
import argparse
import glob
# from sample_agents import MyAgent
# from sample_agents import RandomAgent
import os
import random
import socket
import subprocess
import sys
from time import sleep

import matplotlib.pyplot as plt
import numpy as np

from sample_agents import RLAgent


class AgentHandler(object):
    STAGE_STEPS_NUM = 1000

    def __init__(self, host, port, num_episodes, agent_params):
        self.agent_params = agent_params
        self.num_episodes = num_episodes
        self.ChosenAgent = RLAgent
        self.LINE_SEPARATOR = '\n'
        self.BUF_SIZE = 4096  # in bytes
        self.connected = False
        self.host = host
        self.sock = None
        self.port = port
        self.agent = None

    def connect(self, host, port):
        if not self.connected:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((host, port))
            except socket.error:
                print('Unable to contact environment at the given host/port.')
                self.connected = False
                sys.exit(1)
            self.connected = True

    def disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            self.connected = False

    def send_str(self, s):
        self.sock.send(bytes(s + self.LINE_SEPARATOR, encoding='utf8'))

    def receive(self, numTokens):
        data = ['']
        while len(data) <= numTokens:
            rawData = data[-1] + str(self.sock.recv(self.BUF_SIZE), encoding='utf8')
            del data[-1]
            data = data + rawData.split(self.LINE_SEPARATOR)

        del data[-1]
        return data

    def send_action(self, action):
        # sends all the components of the action one by one
        for a in action:
            self.send_str(str(a).replace('.', ','))

    def run(self):
        self.connect(self.host, self.port)

        self.send_str('GET_TASK')
        data = self.receive(2)
        stateDim = int(data[0])
        actionDim = int(data[1])

        # instantiate agent
        if self.agent is None:
            self.agent = self.ChosenAgent.Agent(stateDim, actionDim, agent_params)

        self.send_str('START_LOG')
        self.send_str(self.agent.getName())

        stage = 0
        while True:
            self.send_str('START')

            step = 0
            data = self.receive(2 + stateDim)
            terminalFlag = int(data[0])
            state = list(map(float, data[2:]))
            action = self.agent.start(state)
            while not terminalFlag:
                self.send_str('STEP')
                self.send_str(str(actionDim))
                self.send_action(action)

                data = self.receive(3 + stateDim)
                if not (len(data) == stateDim + 3):
                    print('Communication error: calling agent.cleanup()')
                    self.safe_exit()
                    sys.exit(1)

                reward = float(data[0])
                terminalFlag = int(data[1])
                state = list(map(float, data[3:]))

                action = self.agent.step(reward, state)

                if self.num_episodes is not None:
                    step += 1
                    if step >= self.STAGE_STEPS_NUM or reward == 10:
                        return
            stage += 1
            if stage > self.num_episodes:
                break
        if random.random() < 0.01:
            self.agent.reset_epsilon()

    def safe_exit(self):
        self.agent.safe_exit()


class EnvironmentHandler(object):
    def __init__(self, port, test_dir, gui, plot_score):
        self.plot_score = plot_score
        self.gui = gui
        self.port = port
        self.test_dir = test_dir
        self.tests = []
        self.load_tests()
        self.process = None
        self.score = []
        self.tests_angle = []
        self.current_test = 0

    def start_next(self):
        if self.plot_score:
            test = self.tests[self.current_test]
        else:
            test = random.choice(self.tests)
        print('New test', test)

        for log_file in glob.glob('*.log'):
            os.remove(log_file)
        self.process = subprocess.Popen(
            ['java', '-jar', 'octopus-environment.jar', self.gui, test, str(self.port)])
        sleep(1)

    def stop(self):
        if self.process is not None:
            self.process.terminate()
        sleep(0.5)
        log = glob.glob('*.log')
        if log:
            with open(log[0], 'r') as log_file:
                score = np.mean([sum(float(r) for r in line.split()) for line in log_file])
                print('Score:', score)
                if self.plot_score:
                    self.score.append(score)
                    self.tests_angle.append(float(self.tests[self.current_test].split('\\')[1][:-4][5:]))

        self.current_test += 1
        if self.current_test >= len(self.tests):
            self.current_test = 0
            if self.plot_score:
                plt.plot(self.tests_angle, self.score)
                plt.xlabel('Początkowy kąt')
                plt.ylabel('Wynik')
                plt.show()
                with open("score.csv", 'w', encoding='utf8') as file:
                    file.write("angle;score\n")
                    for ang, sc in zip(self.tests_angle, self.score):
                        file.write(f"{ang};{sc}\n")

    def load_tests(self):
        for test_path in glob.glob(os.path.join(self.test_dir, '*.xml')):
            self.tests.append(test_path)
        if self.plot_score:
            self.tests = sorted(self.tests, key=lambda x: float(x.split('\\')[1][:-4][5:]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gui', choices=['internal', 'external', 'external_gui'])
    parser.add_argument('--p', dest='port', default=1410, required=False)
    parser.add_argument('--h', dest='host', default='localhost', required=False)
    parser.add_argument('--t', dest='test_dir', default='tests', required=False)
    parser.add_argument('--e', dest='episodes', default=5, required=False, type=int)
    parser.add_argument('--no_learning', action='store_false', default=True)
    parser.add_argument('--plot_score', action='store_true', default=False)
    args = parser.parse_args()
    agent_params = {
        RLAgent.Agent.IS_LEARNING: args.no_learning
    }
    agent_handler = AgentHandler(args.host, args.port, args.episodes, agent_params)
    env_handler = EnvironmentHandler(args.port, args.test_dir, args.gui, args.plot_score)

    try:
        while True:
            env_handler.start_next()
            agent_handler.run()
            agent_handler.disconnect()
            env_handler.stop()
    except (ConnectionResetError, KeyboardInterrupt, Exception) as e:
        print(e)
        agent_handler.safe_exit()
        agent_handler.disconnect()
        env_handler.stop()
        print('Agent has stopped safely')
