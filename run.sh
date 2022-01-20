nohup python3 main.py --cuda --env_name=$1 --noise_scale=0 > result/$1/base.txt 2>&1 &

nohup python3 main.py --cuda --env_name=$1 --batchnorm --noise_scale=0> result/$1/batchnorm.txt 2>&1 &

nohup python3 main.py --cuda --env_name=$1 --batchnorm --show> result/$1/ddpg.txt 2>&1 &

nohup python3 main.py --cuda --env_name=$1 --batchnorm --doubleDQN > result/$1/ddpg_doubleDQN.txt 2>&1 &

nohup python3 main.py --cuda --env_name=$1 --batchnorm --policy_update_interval=2 > result/$1/ddpg_delay.txt 2>&1 &

nohup python3 main.py --cuda --env_name=$1 --batchnorm --policy_smooth > result/$1/ddpg_smooth.txt 2>&1 &

nohup python3 main.py --cuda --env_name=$1 --batchnorm --priority > result/$1/ddpg_priority.txt 2>&1 &

nohup python3 main.py --cuda --env_name=$1 --batchnorm --nstep > result/$1/ddpg_nstep.txt 2>&1 &

nohup python3 main.py --cuda --env_name=$1 --batchnorm --doubleDQN --policy_update_interval=2 --policy_smooth --priority --nstep > result/$1/full.txt 2>&1 &