import os
import json
import pickle
import argparse

import torch
import gym

import wandb
# wandb.login(key="0569aea85a4d43c84d0994c5ae88cc6d236faa5f")

from models.nets import Expert
from models.optimistic_tbs import GAIL


def main(env_name):
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    if env_name not in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3"]:
        print("The environment name is wrong!")
        return

    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, env_name)

    with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
        expert_config = json.load(f)

    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = gym.make(env_name)
    env.reset()

    state_dim = len(env.observation_space.high)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    expert = Expert(
        state_dim, action_dim, discrete, **expert_config
    ).to(device)
    expert.pi.load_state_dict(
        torch.load(
            os.path.join(expert_ckpt_path, "policy.ckpt"), map_location=device
        )
    )

    # # Initialise WandB
    # wandb.init(
    #     project="GAIL_Project_{}".format(env_name),
    #     tags=["GAIL"],
    #     mode="online",
    # )# TODO: switch to hydra to specify the run name.

    # model = GAIL(state_dim, action_dim, discrete, config).to(device)

    # results = model.train(env, expert)

    # Before starting the main loop
    all_rewards = []
    # Loop to repeat the experiment 8 times
    for experiment in range(5):
        # Initialise WandB
        wandb.init(
            project="GAIL_Project_{}".format(env_name),
            tags=["GAIL_AC_opt_3"],
            mode="offline",
            group="experiment_AC_opt_3",
            job_type="AC_opt_3"
        )  # TODO: switch to hydra to specify the run name.

        # Reset the environment and model for each experiment
        env.reset()

        model = GAIL(state_dim, action_dim, discrete, config).to(device)

        # The train function returns the experiment's results, including rewards
        results = model.train(env, expert)

        # with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        #     pickle.dump(results, f)

        # Collect rewards from each iteration
        # all_rewards.append(results[1])
        wandb.finish()

    env.close()

    # with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
    #     pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(
            model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt")
        )
    if hasattr(model, "v"):
        torch.save(
            model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt")
        )
    if hasattr(model, "d"):
        torch.save(
            model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    args = parser.parse_args()

    main(**vars(args))
