import retro
import gym
import numpy as np
from DQ import DuelingDQNPrioritizedReplay
from matplotlib import pyplot as plt
import cv2




class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    # B is do nothing
    # down

    # def __init__(self, env):
    #     super(SonicDiscretizer, self).__init__(env)
    #     buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
    #     actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
    #                ['DOWN', 'B'], ['B']]
    #     self._actions = []
    #     for action in actions:
    #         arr = np.array([False] * 12)
    #         for button in action:
    #             arr[buttons.index(button)] = True
    #         self._actions.append(arr)
    #     self.action_space = gym.spaces.Discrete(len(self._actions))


    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['RIGHT'], ['DOWN', 'B'], ['B'], ['RIGHT', 'DOWN']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()

# function starter.
if __name__ == "__main__":

    # load SonicGame
    env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

    # ActionWrapper for sonicGame
    env = SonicDiscretizer(env)
    print(env.action_space)

    # env = env.unwrapped


    # initialize the argument for DeepQ network.
    # N_A = 7
    N_A = env.action_space.n
    N_S = 1120
    MEMORY_CAPACITY = 50000
    TARGET_REP_ITER = 2000
    MAX_EPISODES = 3000
    E_GREEDY = 0.95
    E_INCREMENT = 0.00001
    GAMMA = 0.99
    LR = 0.0001
    BATCH_SIZE = 32
    HIDDEN = [400, 400]
    RENDER = True

    RL = DuelingDQNPrioritizedReplay(n_actions=N_A, n_features=N_S,
                                     learning_rate=LR, e_greedy=E_GREEDY,
                                     reward_decay=GAMMA, hidden=HIDDEN,
                                     batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER,
                                     memory_size=MEMORY_CAPACITY, e_greedy_increment=E_INCREMENT, output_graph=True)


    total_steps = 0

    rewardY = []
    iterX = []
    for iteration in range(MAX_EPISODES):

        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx / 8)
        iny = int(iny / 8)


        n = 0
        fitness_max_current = 0
        fitness_current = 0
        x_pos_max = 0
        x_pos_end = 0
        counter = 0
        input_array = []
        output_array = []


        while True:
            # if total_steps > MEMORY_CAPACITY:
            #     env.render()


            # ---------- scale the input state ----------
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            for x in ob:
                for y in x:
                    input_array.append(y)
            current_state = np.asarray(input_array)
            input_array.clear()


            # get next action from the current state
            action = RL.choose_action(current_state)
            next_ob, reward, done, info = env.step(action)



            # ---------- scale the output state ----------
            next_state = cv2.resize(next_ob, (inx, iny))
            next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            next_state = np.reshape(next_state, (inx, iny))

            for x in next_state:
                for y in x:
                    output_array.append(y)
            next_state = np.asarray(output_array)
            output_array.clear()

            x_pos_end = info['screen_x_end']
            xpos = info['x']
            if xpos > x_pos_max:
                fitness_current += 1
                x_pos_max = xpos

            if xpos == x_pos_end and xpos > 500:
                fitness_current += 100000
                break

            if fitness_current > fitness_max_current:
                fitness_max_current = fitness_current
                counter = 0
            else:
                counter += 1

            RL.store_transition(current_state, action, fitness_current, next_state)

            if total_steps > MEMORY_CAPACITY:
                RL.learn()


            if done or counter == 500:
                iterX.append(iteration)
                rewardY.append(fitness_current)
                print(iteration)
                print(fitness_current)
                print("-----------")
                break
            if iteration == 200:
                plt.plot(iterX, rewardY)
                plt.suptitle('Sonic Reward Chart')
                plt.ylabel('Sonic Reward')
                plt.xlabel('Iteration')
                plt.show()

            ob = next_ob
            total_steps += 1