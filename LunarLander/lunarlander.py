import gym
from gym import wrappers
from DQ import DuelingDQNPrioritizedReplay


# function to run lunar_lander game
def run_lunar_lander():
    print("------------------------")
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    print("------------------------")


    # to count which step agent currently on
    total_steps = 0
    running_r = 0
    r_scale = 100

    for iteration in range(MAX_EPISODES):

        # initialize start state for each iteration
        state = env.reset()  # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground, r_leg_on_ground)
        ep_r = 0

        while True:
            # refresh env
            if total_steps > MEMORY_CAPACITY:
                env.render()

            # using neural network to choose an action
            action = RL.choose_action(state)

            # current env take the action and reach next state and get reward.
            next_state, reward, done, _ = env.step(action)

            # ------------------- update reward -------------------
            if reward == -100:
                reward = -30
            reward /= r_scale
            ep_r += reward

            # Remember or store the transition function to neural network.
            RL.store_transition(state, action, reward, next_state)

            if total_steps > MEMORY_CAPACITY:
                RL.learn()

            # if done is true (lunar_lander landed or crashed) print the information.
            # and go to next iteration.
            if done:
                land = '| Landed' if reward == 100 / r_scale else '| ------'
                running_r = 0.99 * running_r + 0.01 * ep_r
                print('Iter: ', iteration, land,
                      '| Epi_R: ', round(ep_r, 2),
                      '| Running_R: ', round(running_r, 2),
                      '| Epsilon: ', round(RL.epsilon, 3))
                break

            # update current state to next state.
            state = next_state
            total_steps += 1

if __name__ == "__main__":

    # load the lunarLander env from gym.
    env = gym.make('LunarLander-v2')
    # env = env.unwrapped
    env.seed(1)

    # ----- initialize the argument for DeepQ network.
    N_A = env.action_space.n
    N_S = env.observation_space.shape[0]
    MEMORY_CAPACITY = 50000
    TARGET_REP_ITER = 2000
    MAX_EPISODES = 900
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
                                     memory_size=MEMORY_CAPACITY, e_greedy_increment=E_INCREMENT, )
    run_lunar_lander()
