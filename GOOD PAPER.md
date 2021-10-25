#### A Lifelong Learning Approach to Mobile Robot Navigation

**原有问题**：经典的静态导航方法需要环境特定的原位系统调整，例如来自人类专家，或者无论他们在同一环境中导航多少次都可能重复他们的错误。 基于学习的导航具有随着经验改进的潜力，高度依赖于对训练资源的访问，例如足够的内存和快速计算，并且容易忘记以前学习的能力，尤其是在面对不同环境时。 

**贡献**：导航终身学习（LLfN），它（1）纯粹基于自身经验改善移动机器人的导航行为，（2）在新环境中学习后保留机器人在先前环境中导航的能力

**大致做法：**在每个环境中它导航到一个固定目标，不断识别次优行为，并在保留其过去知识的同时改进它们。

和强化学习设置优先级经验池思路相似，还需要细看为什么它如此自信可以实现终身学习



### 强化学习

#### Bi-Directional Domain Adaptation for Sim2Real Transfer of Embodied Navigation Agents

**原有问题**：仅在模拟中训练的机器人无法推广到现实世界，导致“模拟与真实的差距”，我们如何克服来自模拟器的大量不太准确的人工数据与可靠的真实数据稀缺之间的权衡？

**贡献**：在这封信中，我们提出了双向域适应 (BDA)，这是一种在两个方向上弥合 sim-vs-real 差距的新方法

**大致做法：**学习了一个 sim2real 动态适应模块来预测模拟和现实中状态转换之间的残差。学习了一个 real2sim 观察适应模块，将机器人在测试时在现实世界中看到的图像转换为与机器人在训练期间在模拟中看到的更接近的图像。

![image-20211015231324940](https://gitee.com/xiaohui288/typora_img/raw/master/img/image-20211015231324940.png)



#### Visual Navigation in Real World Indoor Environments Using End to End Deep Reinforcement Learning



#### Reinforcement Learning for Autonomous Driving with Latent State Inference and Spatial-Temporal Relationships

编码现实中的细微线索提升学习效果



#### Coding for Distributed Multi-Agent Reinforcement Learning

减轻多智能体强化学习 (MARL) 问题的同步分布式学习中的落后者效应。



### 其他

#### Autonomous Aerial Swarming in GNSS-denied Environments with High Obstacle Density

**贡献**，集群，无通信，高障碍物密度区域中相对局部化的无人机 (UAV) 的紧凑群集，基于平面 LIDAR 的同时定位和映射 (SLAM) 和相对定位系统紫外线方向和测距 (UVDAR) 的框架

我们提出了一种完全分散的、仿生的控制律，该控制律仅使用机载传感器数据在环境中安全地集群，而无需与其他代理进行任何通信。 在所提出的方法中，每个无人机代理使用机载传感器进行自我定位并估计其他代理在其本地参考系中的相对位置。

**做法**：通过使用控制律控制每个无人机代理的运动，实现自聚集和群集。 将每个无人机视为一个**双积分器系统**，控制律用于生成代理所需的加速度。 每个智能体使用相同的控制律，集群从他们的集体运动中出现。 提议的控制律是几个组件的组合，负责控制与其他智能体的**距离、避免与障碍物的碰撞以及引导智能体朝向目标**。

这不就是强化学习的原理？但是是用数学积分项控制



#### Learning from Demonstration without Demonstrations

我们提出了演示发现的概率规划（P2D2），这是一种无需专家就可以自动发现演示的技术。我们将发现演示制定为搜索问题，并利用广泛使用的规划算法（例如快速探索随机树）来查找演示轨迹。这些演示用于初始化策略，然后通过通用 RL 算法进行细化。



#### Towards Safe Motion Planning in Human Workspaces: A Robust Multi-Agent Approach

我们将人类工作空间中的机器人规划建模为随机游戏，并提供了一种稳健的规划算法，该算法使机器人能够在人类反应中考虑其预测误差以防止碰撞



#### （有意思的点）Do You See What I See? Coordinating Multiple Aerial Cameras for Robot Cinematography

实时多无人机协调系统，该系统能够记录动态目标，同时最大限度地提高镜头多样性并避免相机之间的碰撞和相互可见



### 高飞

#### Fast-Tracker: A Robust Aerial System for Tracking Agile Target in Cluttered Environments  

目标检测+追踪

#### Teach-Repeat-Replan: A Complete and Robust System for Aggressive Flight in Complex Environments

我们提出了一种自动将可以任意抖动的无人驾驶轨迹转换为拓扑等距的方法借给一个。生成的轨迹平滑、安全、动态可行，具有更好的攻击性