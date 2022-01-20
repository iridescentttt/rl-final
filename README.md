# Reproduction of Continuous Control with Deep Reinforcement Learning

项目架构

- ddpg.py: DDPG架构定义
- main.py: 程序主入口
- memory.py: 经验池文件
- model.py: 定义神经网络结构
- noise.py: 定义噪声
- options.py: 可选选项
- utils.py: 辅助函数
- run.sh: 运行脚本

运行命令示例

```bash
python3 main.py --cuda --env_name=Hopper-v2
```

具体参数见 options.py

也可通过 run.sh 脚本直接运行