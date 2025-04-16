# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
from classes.DataCollector import DecisionTransformerGymDataCollator
from datasets import DatasetDict
from dt_gridnet_eval import decode_action, decode_obs, get_action
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder

from experiments.classes.TrainableDT import TrainableDT
from gym_microrts import microrts_ai  # noqa

import importlib

import sys
import os
pwd = os.path.dirname(__file__)

packages_path = os.path.join(pwd, 'packages')
models_path = os.path.join(pwd, 'models')

packages = [f.name for f in os.scandir(packages_path) if f.is_dir()]
for package in packages:
    package_path = os.path.join(packages_path, package)
    sys.path.append(package_path)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=False,
        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--dt-dataset', type=str, default="episode_data/cm-mcrp-dataset-v2/save_0",
        help='the path to the decision transformer dataset')

    # Algorithm specific arguments
    parser.add_argument(
        '--games-per-match', 
        type=int,
        default=10,
        help='the number of games to be played in a match'
    )
    parser.add_argument(
        "--agents",
        type=str,
        help="the path to the agent models to be ran in the tournament"
    )
    parser.add_argument(
        '--ais', 
        nargs='+',
        help="the ais to be ran in the tournament"
    )
    parser.add_argument(
        '--dts',
        nargs='+',
        help="the path to the decision transformers to be ran in the tournament"
    )
    parser.add_argument(
        '--eval-map',
        type=str,
        default="maps/16x16/basesWorkers16x16A.xml",
        help="the map to be used in the tournament"
    )

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    return args


def get_bot_name(bot):
    if bot["type"] == "agent" or bot["type"] == "ai":
        return bot["path"]
    elif bot["type"] == "dt":
        delimiter = "/" if "/" in bot["path"] else "\\"
        split_path = bot["path"].split(delimiter)
        if split_path[-1] == "":
            return split_path[-2]
        else:
            return split_path[-1]
    else:
        raise ValueError(f"Unknown bot type: {bot_type}")


def print_final_results(bots_wins, all_bots):
    print("\nFinal Results:")
    bot_names = [get_bot_name(bot) for bot in all_bots]
    header = " " * 15 + " | " + " | ".join(f"{name:>15}" for name in bot_names) + " | Total Wins"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(bots_wins):
        total_wins = sum(row)
        row_str = f"{bot_names[i]:>15} | " + " | ".join(f"{win:>15}" for win in row) + f" | {total_wins:>10}"
        print(row_str)


if __name__ == "__main__":
    args = parse_args()

    from ppo_gridnet import MicroRTSStatsRecorder

    from gym_microrts.envs.vec_env import (MicroRTSBotVecEnv,
                                           MicroRTSGridModeVecEnv)

    agents_file = open(args.agents, 'r')
    
    # TRY NOT TO MODIFY: seeding
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    max_ep_length = 2000
    exp_name = f"tournament-{time.time()}".replace(".", "")
    mapsize = 16 * 16

    # variables for decision transformer
    TARGET_RETURN = 10
    if args.dts:
        dataset = DatasetDict.load_from_disk(args.dt_dataset)
        collector = DecisionTransformerGymDataCollator(dataset["train"])

    all_bots = []
    if args.agents:
        for line in agents_file:
            if line.strip()[0] == '#':
                continue
            split = line.split()
            all_bots.append(
                    {"type": "agent", "path": split[0], "module_name": split[1], "class_name": split[2]}
            )
    if args.ais:
        all_bots.extend(
            [{"type": "ai", "class_name": ai} for ai in args.ais]
        )
    if args.dts:
        all_bots.extend(
            [{"type": "dt", "path": dt} for dt in args.dts]
        )
    assert len(all_bots) > 1, "at least 2 agents/ais are required to play a tournament"

    bots_wins = np.zeros((len(all_bots), len(all_bots)), dtype=np.int32)

    for i in range(len(all_bots)):
        for j in range(i + 1, len(all_bots)):
            player1 = all_bots[i]
            player2 = all_bots[j]
            player1_type = player1["type"]
            player2_type = player2["type"]
            ai1s = []
            ai2s = []

            player1_name = get_bot_name(player1)
            player2_name = get_bot_name(player2)

            if player1_type == "ai" and player2_type == "ai":
                bot_envs, selfplay_envs = 1, 0
                ai1s = [eval(f"microrts_ai.{player1_name}")]
                ai2s = [eval(f"microrts_ai.{player2_name}")]
                envs = MicroRTSBotVecEnv(
                    ai1s=ai1s,
                    ai2s=ai2s,
                    partial_obs=False,
                    max_steps=max_ep_length,
                    render_theme=2,
                    map_paths=[args.eval_map],
                    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                    autobuild=False
                )
            else:
                if player1_type == "ai":
                    bot_envs, selfplay_envs = 1, 0
                    ai2s = [eval(f"microrts_ai.{player1_name}")]
                elif player2_type == "ai":
                    bot_envs, selfplay_envs = 1, 0
                    ai2s = [eval(f"microrts_ai.{player2_name}")]
                else:
                    bot_envs, selfplay_envs = 0, 2 

                envs = MicroRTSGridModeVecEnv(
                    num_bot_envs=bot_envs,
                    num_selfplay_envs=selfplay_envs,
                    ai2s=ai2s,
                    partial_obs=False,
                    max_steps=max_ep_length,
                    render_theme=2,
                    map_paths=[args.eval_map],
                    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                    autobuild=False
                )
    
            envs = MicroRTSStatsRecorder(envs)
            envs = VecMonitor(envs)
            next_obs = torch.Tensor(envs.reset()).to(device)

            if args.capture_video:
                envs = VecVideoRecorder(
                    envs, 
                    f"videos/{exp_name}/{player1_name}-{player2_name}",
                    record_video_trigger=lambda x: x == 0,
                    video_length=max_ep_length,
                )

            agent1 = None
            agent2 = None
            if player1_type == "agent":
                agent1_module = importlib.import_module(player1["module_name"])
                agent1_class = getattr(agent1_module, player1["class_name"])
                agent1 = agent1_class(envs).to(device)
                agent1.load_state_dict(torch.load(os.path.join(models_path, player1["path"]), map_location=device, weights_only=True), strict=False)
                agent1.eval()

            if player2_type == "agent":
                agent2_module = importlib.import_module(player2["module_name"])
                agent2_class = getattr(agent2_module, player2["class_name"])
                agent2 = agent2_class(envs).to(device)
                agent2.load_state_dict(torch.load(os.path.join(models_path, player2["path"]), map_location=device, weights_only=True), strict=False)
                agent2.eval()
            
            if player1_type == "dt":
                agent1 = TrainableDT.from_pretrained(player1["path"]).to(device)
                agent1.eval()
            
            if player2_type == "dt":
                agent2 = TrainableDT.from_pretrained(player2["path"]).to(device)
                agent2.eval()

            print("\n\n====== Next Match ======")
            print(f"{player1_name} vs. {player2_name}")

            for game in range(args.games_per_match):
                # reset the DT trajectory
                if player1_type == "dt" or player2_type == "dt":
                    states = decode_obs(next_obs[0].view(mapsize, -1)).reshape(
                        1, collector.state_dim).to(device=device, dtype=torch.float32)
                    target_return = torch.tensor(
                        TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
                    actions = torch.zeros(
                        (0, collector.act_dim), device=device, dtype=torch.float32)
                    rewards = torch.zeros(0, device=device, dtype=torch.float32)
                    timesteps = torch.tensor(
                        0, device=device, dtype=torch.long).reshape(1, 1)

                for update in range(max_ep_length):
                    envs.render(mode="human")
                    with torch.no_grad():
                        p1_action = None
                        p2_action = None
                        invalid_action_masks = torch.tensor(np.array(envs.get_action_mask())).to(device)

                        if player1_type == "agent":
                            p1_obs = next_obs[::2]
                            p1_action = agent1.get_action(p1_obs, invalid_action_masks[::2])
                        if player2_type == "agent":
                            p2_obs = next_obs[1::2]
                            p2_action = agent2.get_action(p2_obs, invalid_action_masks[1::2])
                        if player1_type == "dt":
                            invalid_action_masks = torch.tensor(
                                np.array(envs.get_action_mask())).to(device)

                            actions = torch.cat([actions, torch.zeros(
                                (1, collector.act_dim), device=device)], dim=0)
                            rewards = torch.cat(
                                [rewards, torch.zeros(1, device=device)])

                            p1_action = get_action(
                                agent1,
                                (states - collector.state_mean) / collector.state_std,
                                actions,
                                target_return,
                                timesteps,
                                invalid_action_masks[0]
                            )

                            action[::2] = decode_action(
                                p1_action.view(mapsize, -1)).view(1, mapsize, -1)

                            next_state = torch.Tensor(
                                decode_obs(next_obs[0].view(mapsize, -1))
                            ).to(device).reshape(1, collector.state_dim)

                            states = torch.cat([states, next_state], dim=0)
                            rewards[-1] = rs[0]

                            pred_return = target_return[0, -1] - rs[0]
                            target_return = torch.cat(
                                [target_return, pred_return.reshape(1, 1)], dim=1)
                            timesteps = torch.cat([timesteps, torch.ones(
                                (1, 1), device=device, dtype=torch.long) * (update + 1)], dim=1)
                        if player2_type == "dt":
                            invalid_action_masks = torch.tensor(
                                np.array(envs.get_action_mask())).to(device)

                            actions = torch.cat([actions, torch.zeros(
                                (1, collector.act_dim), device=device)], dim=0)
                            rewards = torch.cat(
                                [rewards, torch.zeros(1, device=device)])

                            p2_action = get_action(
                                agent2,
                                (states - collector.state_mean) / collector.state_std,
                                actions,
                                target_return,
                                timesteps,
                                invalid_action_masks[0]
                            )

                            p2_action = decode_action(
                                p2_action.view(mapsize, -1)).view(1, mapsize, -1)

                            next_state = torch.Tensor(
                                decode_obs(next_obs[1].view(mapsize, -1))
                            ).to(device).reshape(1, collector.state_dim)

                            states = torch.cat([states, next_state], dim=0)
                            rewards[-1] = rs[1]

                            pred_return = target_return[0, -1] - rs[1]
                            target_return = torch.cat(
                                [target_return, pred_return.reshape(1, 1)], dim=1)
                            timesteps = torch.cat([timesteps, torch.ones(
                                (1, 1), device=device, dtype=torch.long) * (update + 1)], dim=1)

                        action = torch.zeros(
                            (envs.num_envs, mapsize, 7)) # 7 action planes

                        if p1_action is not None:
                            action[::2] = p1_action
                        if p2_action is not None:
                            action[1::2] = p2_action

                        try:
                            next_obs, _, ds, infos = envs.step(
                                action.cpu().numpy().reshape(envs.num_envs, -1))
                            next_obs = torch.Tensor(next_obs).to(device)
                        except Exception as e:
                            e.printStackTrace()
                            raise

                    
                    if "episode" in infos[0].keys():
                        # game finished
                        score = int(infos[0]["microrts_stats"]["WinLossRewardFunction"])
                        if score > 0: # player1 wins
                            bots_wins[i][j] += 1
                            print(f"Game {game + 1}: {player1_name} wins!")
                        elif score < 0: # player2 wins
                            bots_wins[j][i] += 1
                            print(f"Game {game + 1}: {player2_name} wins!")
                        else: # draw
                            print(f"Game {game + 1}: it's a draw!")

                    # game exit condition
                    if ds[0]:
                        break;

            print("====== Match Done ======\n\n")

    print_final_results(bots_wins, all_bots)
