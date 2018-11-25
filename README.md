### Octopus Arm RL

Project demonstrates powers of Reinforcement Learning on Octopus Arm model.

Agent's target is to touch red dot using 3 types of muscles of arm.

![octopus_arm](https://user-images.githubusercontent.com/12548284/48979783-e10dd500-f0bf-11e8-883f-f12b7a72ef38.gif)

### Usage

Environment is implemented in java and communcation with python is provided with sockets.

```
$ python agent_handler.py --help
usage: agent_handler.py [-h] [--p PORT] [--h HOST] [--t TEST_DIR]
                        [--e EPISODES] [--no_learning] [--plot_score]
                        {internal,external,external_gui}
```

Example:
```
python agent_handler.py externa_gui --no_learning --t test_hard
```

### Tests

For leaning procces test has been divided on dirs by the level of difficulty.

* easy
* medium
* hard
* nightmare

### Results

With RL our agent was able to solve all tests and achieve average score 8.35935.

Plot below shows score depending on starting angle of arm:

![35398984_1674484212648368_6107770346218192896_n](https://user-images.githubusercontent.com/12548284/48979948-87f37080-f0c2-11e8-9ade-7de4f501ab82.png)

### More information about implementation (PL)

[Report](https://github.com/mrugacz95/octopus-arm-rl/blob/master/report.pdf)

### Bibliography

* Octopus Arm Control with RVM-RL, https://www.cs.colostate.edu/~lemin/octopus.php
* Keon, Deep Q-Learning with Keras and Gym, https://keon.io/deep-q-learning/
* Arthur Juliani,Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning
with Tables and Neural Networks,
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-p
art-0-q-learning-with-tables-and-neural-networks-d195264329d0
* Tambet Matiisen, Guest Post (Part I): Demystifying Deep Reinforcement Learning,
https://ai.intel.com/demystifying-deep-reinforcement-learning/
* Alex Lenail, http://alexlenail.me/NN-SVG/index.html