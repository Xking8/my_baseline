import gym

from baselines import deepq


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("LunarLander-v2")
    #model = deepq.models.mlp([64])
    model = deepq.models.mlp([40,50])
    act = deepq.learn(
        env,
        q_func=model,
        #lr=1e-3,
	lr=0.00025,
        #max_timesteps=10000,
        #max_timesteps=100000,
        max_timesteps=6000000,
        #buffer_size=50000,
        buffer_size=100000,
	target_network_update_freq=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        prioritized_replay=True,
        gamma = 0.99,
        print_freq=10,
        callback=callback
    )
    print("NOT Saving model to cartpole_model.pkl")
    #act.save("LunarLander_model.pkl")


if __name__ == '__main__':
    main()
