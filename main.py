import gym
import torch
import sys
import random
import numpy as np
from options import args_parser
from ddpg import *
from utils import *
from noise import *
from memory import *
from tensorboardX import SummaryWriter
import gym_cartpole_swingup

args = args_parser()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print("using device", device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
# device = "cuda:1"
print("DEVICE:", device, flush=True)

if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=sys.maxsize)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    """environment"""
    env = gym.make(args.env_name)
    env.seed(args.seed)

    """parameters"""
    gamma = args.gamma
    tau = args.tau
    hfeat1 = args.hfeat1
    hfeat2 = args.hfeat2
    infeat = env.observation_space.shape[0]
    if env.action_space.shape:
        outfeat = env.action_space.shape[0]
    else:
        outfeat = env.action_space.n
    noise_scale = args.noise_scale
    episodes = args.episodes
    doubleDQN = args.doubleDQN
    policy_update_interval = args.policy_update_interval
    policy_smooth = args.policy_smooth
    replay_size = args.replay_size
    priority = args.priority
    nstep = args.nstep
    batchnorm = args.batchnorm
    show = args.show
    max_action = float(env.action_space.high[0])

    """print the setting"""
    print("Double DQN:", doubleDQN, flush=True)
    print("Delayed Policy Update:", policy_update_interval > 1, flush=True)
    print("Policy Smooth:", policy_smooth, flush=True)
    print("Priority Memory:", priority, flush=True)
    print("Nstep Learning:", nstep, flush=True)
    print("Batch Normalization:", batchnorm, flush=True)

    """location to store data"""
    image_url = "image/"+args.env_name + ".png"
    rewards_url = "reward/"+args.env_name+"/reward"
    q_step_url = "q_step/"+args.env_name+"/q_step"
    reward_step_url = "reward_step/"+args.env_name+"/reward_step"
    if batchnorm:
        rewards_url += "_batchnorm"
        q_step_url += "_batchnorm"
        reward_step_url += "_batchnorm"
    if doubleDQN:
        rewards_url += "_doubleDQN"
        q_step_url += "_doubleDQN"
        reward_step_url += "_doubleDQN"
    if policy_update_interval > 1:
        rewards_url += "_delay"
        q_step_url += "_delay"
        reward_step_url += "_delay"
    if policy_smooth:
        rewards_url += "_smooth"
        q_step_url += "_smooth"
        reward_step_url += "_smooth"
    if nstep:
        rewards_url += "_nstep"
        q_step_url += "_nstep"
        reward_step_url += "_nstep"
    if priority:
        rewards_url += "_priority"
        q_step_url += "_priority"
        reward_step_url += "_priority"

    """define agent"""
    agent = DDPG(gamma, tau, infeat, outfeat,
                 hfeat1, hfeat2, doubleDQN, policy_update_interval, policy_smooth, max_action, batchnorm, device)

    """memory"""
    if priority:
        memory = PrioritizedReplayMemory(replay_size, gamma, device)
    else:
        memory = ReplayMemory(replay_size, gamma, device)

    """training parameters"""
    rewards = []
    total_numsteps = 0
    updates = 0
    max_reward = None
    q_step = []
    reward_step = []

    """tensorboard visualization"""
    writer = SummaryWriter()

    """noise"""
    ounoise = OUNoise(outfeat)
    ounoise.scale = noise_scale

    """train"""
    for episode in range(1, episodes+1):
        state = torch.Tensor([env.reset()])
        ounoise.reset()

        episode_reward = 0
        
        """interact with the environment"""
        while True:
            """obtain the information from the environment"""
            action = agent.act(state, ounoise)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward])

            """push the record into the memory pool"""
            memory.push(state, action, mask, next_state, reward)

            state = next_state

            """update parameters"""
            if len(memory) > args.batch_size:
                for _ in range(args.updates_per_step):
                    """n-step reward"""
                    if nstep:
                        transitions = memory.nstep_sample(args.batch_size)
                    else:
                        transitions = memory.sample(args.batch_size)
                    batch = Transition(*zip(*transitions))

                    """update parameters"""
                    if doubleDQN:
                        value_loss, _value_loss = agent.update(batch)
                    else:
                        value_loss = agent.update(batch)

                    writer.add_scalar('loss/value', value_loss, updates)

                    updates += 1
            if done:
                break

        writer.add_scalar('reward/train', episode_reward, episode)

        """test"""
        if episode % 10 == 0:
            with torch.no_grad():
                state = torch.Tensor([env.reset()])
                episode_reward = 0
                while True:
                    """take action"""
                    action = agent.act(state, pattern="test")
                    """get the estimated Q"""
                    q_estimate = agent.critic_target(
                        state.to(device), action.to(device)).cpu().numpy()[0]
                    
                    q_step.append(q_estimate)

                    """interact with the environment"""
                    next_state, reward, done, _ = env.step(
                        torch.argmax(action).cpu().numpy())
                    episode_reward += reward
                    reward_step.append(reward)

                    next_state = torch.Tensor([next_state])

                    state = next_state
                    if done:
                        break

                writer.add_scalar('reward/test', episode_reward, episode)

                rewards.append(episode_reward)

                """update the max_reward"""
                if max_reward == None:
                    max_reward = episode_reward
                else:
                    max_reward = max(max_reward, episode_reward)

                """print test reward"""
                print("Episode: {}, reward: {:.4f}, average reward: {:.4f}, max reward: {:.4f}".format(
                    episode, rewards[-1], np.mean(rewards[-10:]), max_reward), flush=True)

            """store the data for visualization"""
            np.save(q_step_url, q_step)
            np.save(reward_step_url, reward_step)
            np.save(rewards_url, rewards)
    env.close()
