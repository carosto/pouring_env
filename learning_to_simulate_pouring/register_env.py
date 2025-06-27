import gymnasium as gym
from gymnasium.envs.registration import register


# Register the Pouring environment
register(
    id="PouringEnv-v0",
    entry_point="learning_to_simulate_pouring.rl_environment:PouringEnv",
)


"""if __name__ == "__main__":
    gym.pprint_registry()"""