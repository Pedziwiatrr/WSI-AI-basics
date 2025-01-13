import argparse
import gymnasium as gym
from cliff_walking import q_learn


def main():
    environment = gym.make("CliffWalking-v0")
    q_learn(environment)


if __name__ == '__main__':
    main()