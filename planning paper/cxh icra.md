### **Session** TuDT22 : Aerial Systems: Multi-Robots    

#### (no paper)A Multi-UAV System for Detection and Elimination of Multiple Targets  

快速可靠地拦截 3D 目标

#### Optic Flow-Based Reactive Collision Prevention for MAV Using Fictitious Obstacle Hypothesis 

一种基于扩展卡尔曼滤波器（EKF）的新障碍物检测策略，结合了 光流发散（OFD） 和惯性传感。区分周围产生的 OFD 和由实际障碍引起的 OFD

#### Autonomous Aerial Swarming in GNSS-denied Environments with High Obstacle Density 

分布式，集群，无通信，高障碍物密度区域中相对局部化的无人机 (UAV) 的紧凑群集，基于平面 LIDAR 的同时定位和映射 (SLAM) 和相对定位系统紫外线方向和测距 (UVDAR) 的框架

#### Forceful Aerial Manipulation Based on an Aerial Robotic Chain: Hybrid Modeling and Control

空中机器人链条机械手的系统设计、建模和控制



### Session TuAT15 : Learning for Motion Planning

#### (no paper)Deep Imitation Learning for Autonomous Navigation in Dynamic Pedestrian Environments



#### ※Learning from Demonstration without Demonstrations

我们提出了演示发现的概率规划（P2D2），这是一种无需专家就可以自动发现演示的技术。我们将发现演示制定为搜索问题，并利用广泛使用的规划算法（例如快速探索随机树）来查找演示轨迹。这些演示用于初始化策略，然后通过通用 RL 算法进行细化。

#### Optimal Cooperative Maneuver Planning for Multiple Nonholonomic Robots in a Tiny Environment  via Adaptive-scaling Constrained Optimization

这项工作提出了一种自适应缩放约束优化 (ASCO) 方法，旨在以解耦的方式找到名义上难以处理的 MVTP 问题的最优解

#### Optimization-Based Framework for Excavation Trajectory Generation

我们通过约束铲斗的瞬时运动并添加面向目标的约束来控制挖掘的土壤量来制定挖掘的通用任务规范。



### Session ThIT23 : Motion Planning for Aerial Robotics    

#### The Reachable Set of a Drone: Exploring the Position Isochrones for a Quadcopter

能够在时间预算内有效地计算四轴飞行器可到达的位置集，可以实现碰撞避免和追避策略。

#### Two-Stage Trajectory Optimization for Flapping Flight with Data-Driven Models

引入两阶段优化程序来规划扑翼飞行轨迹。第一阶段使用用实验飞行数据训练的数据驱动的固定翼近似模型解决轨迹优化问题，被用作第二阶段优化的初始猜测。

#### Online Trajectory Optimization for Dynamic Aerial Motions of a Quadruped Robot

用于在四足机器人上在线规划和执行动态空中运动

#### SwarmCCO: Probabilistic Reactive Collision Avoidance for Quadrotor Swarms under Uncertainty  

在不确定状态估计下运行的四旋翼群的分散避碰算法。我们的方法利用微分平坦度特性和前馈线性化来近似四旋翼动力学并执行相互碰撞避免。



### Session ThKT21 : Motion Planning in Multi-Agents System I    

#### ※Towards Safe Motion Planning in Human Workspaces: A Robust Multi-Agent Approach

我们将人类工作空间中的机器人规划建模为随机游戏，并提供了一种稳健的规划算法，该算法使机器人能够在人类反应中考虑其预测误差以防止碰撞

#### (no paper)Anytime Fault-Tolerant Adaptive Routing for Multi-Robot Teams

车辆路线选择问题

随时启发式方法，它迭代地调整初始路线集，从而提高任务的整体稳健性，并仍然收集最有利可图的奖励

代码：https://git.jl-k.com/verlab/Ftolerant_CTOP_ICRA_2021

#### Exploiting Collisions for Sampling-Based Multicopter Motion Planning

基于采样的方法，可以利用碰撞进行更好的运动规划。该方法建立在RRT*算法的基础上，利用多旋翼飞行器的快速运动原语生成和碰撞检查的优点，通过检测运动原语和障碍物之间的潜在交叉点来生成碰撞状态，并将这些状态与其他采样状态连接起来形成包含碰撞的轨迹。

#### (no paper)Multi-Robot Motion Planning with Unlabeled Goals for Mobile Robots with Differential Constraints

具有未标记目标的多机器人运动规划问题，基于采样的运动规划与目标分配和多代理搜索相结合的想法，多智能体搜索提供了路线图上的非冲突路径，然后指导运动树的基于采样的扩展

作者网址：https://cs.gmu.edu/~plaku/Publications.html



### **Session** ThHT20 : Motion Planning in Multi-Agents System II    

#### A Visibility Roadmap Sampling Approach for a Multi-Robot Visibility-Based Pursuit-Evasion  Problem

基于多机器人可见性的追逃问题任务多个追击机器人

将环境作为其输入并返回一个联合运动策略，以确保逃避者被其中一个追捕者捕获。

#### (no paper)Time-Optimal Multi-Quadrotor Trajectory Planning for Pesticide Spraying

具有有限农药承载能力但能够从农田对面的农药罐

将多个四旋翼的时间最优轨迹生成问题简化为加权图上的多个旅行商问题

首先将感染区域分解为对应于 k 个机器人的 k 个集群，将每个四轴飞行器放置在相应集群的中心，最后，为每个四轴独立求解时间最优轨迹生成问题-转子使用第一种方法

#### （有意思的点）Do You See What I See? Coordinating Multiple Aerial Cameras for Robot Cinematography

实时多无人机协调系统，该系统能够记录动态目标，同时最大限度地提高镜头多样性并避免相机之间的碰撞和相互可见

#### MIDAS: Multi-Agent Interaction-Aware Decision-Making with Adaptive Strategies for Urban  Autonomous Navigation

预测智能体未来动作，模拟交互代理行为的相互影响

本文构建了一种名为 MIDAS 的基于强化学习的方法，其中自我代理学习影响城市驾驶场景中其他汽车的控制动作。



### **Session** ThIT20 : Motion Planning in Multi-Agents System III    

#### Scalable Active Information Acquisition for Multi-Robot Systems 

分布式AIA，本地任务化

多机器人主动信息采集 (AIA) 任务的新型高度可扩展的非近视规划算法。

目标是计算多个机器人的控制策略，以最小化先验未知范围内静态隐藏状态的累积不确定性

#### MAPS-X: Explainable Multi-Robot Motion Planning Via Segmentation

我们提出了对 MMP 计划的解释概念，基于将计划可视化为表示时间段的短图像序列，其中在每个时间段中，代理的轨迹是不相交的，清楚地说明了计划的安全性。

#### Representation-Optimal Multi-Robot Motion Planning Using Conflict-Based Search

将 CBS（基于冲突的搜索算法） 在 MAPF（多智能体寻路） 场景中发现的技术应用于**连续空间**中异构代理的更一般问题来解决 MAMP（多智能体行为规划） 问题。

#### Spatial and Temporal Splitting Heuristics for Multi-Robot Motion Planning

时空分割启发式算法在图论环境中的多机器人运动规划 (MRMP) 问题中的应用     可以以正交方式应用于任何现有的 MRMP 算法



### Session ThJT19 : Multiple and Distributed Intelligence    

#### Multi-Robot Distributed Semantic Mapping in Unfamiliar Environments through Online Matching of  Learned Representations

解决新奇和陌生环境的多机器人分布式语义映射的方法

我们提出的解决方案通过让每个机器人学习无监督语义场景来克服这些障碍在线模型并使用多路匹配算法来识别属于不同机器人的学习语义标签之间的一致匹配集。

#### Learning to Herd Agents Amongst Obstacles: Training Robust Shepherding Behaviors Using Deep  Reinforcement Learning

机器人牧羊问题考虑通过外部机器人（称为牧羊人）的运动对一组连贯的代理（例如，一群鸟或一群无人机）进行控制和导航。

通过使用深度强化学习技术结合概率路线图，我们使用嘈杂但受控的环境和行为参数训练牧羊模型。

#### Sensor Placement for Globally Optimal Coverage of 3D-Embedded Surfaces

针对嵌入在 3D 工作空间中的 2D 表面，对移动传感器覆盖优化问题进行了结构和算法研究

#### (no paper)Reachability Analysis for FollowerStopper: Safety Analysis and Experimental Results

使用可达性分析来验证 FollowerStopper 算法的安全性，该算法是一种设计用于抑制走走停停交通波的控制器。



### Session ThFT7 : Path Planning for Multiple Mobile Robots    

#### (no paper)Hierarchical and Flexible Traffic Management of Multi-AGV Systems Applied to Industrial  Environments

自动化工厂或仓库中多台自动导引车 (AGV) 的交通管理。

#### (no paper)Combining Multi-Robot Motion Planning and Goal Allocation Using Roadmaps

凸多边形定义的环境中导航的具有非完整动力学的机器人车队的自动化问题，同时不会相互碰撞并实现一组目标

我们提出了一种在减少的配置空间中构建抽象多机器人路线图的方法，其中我们考虑了占据相同多边形的机器人之间的环境连通性和干扰成本

#### (no paper)Asynchronous Reliability-Aware Multi-UAV Coverage Path Planning

 Reliability-Aware Multi-Agent Coverage Path Planning (RA-MCPP) 问题为每个机器人找到路径计划，以最大限度地提高在给定期限前完成任务的概率。本文提出了一种在连续时间内制定的 RA-MCPP 路径规划器，可以考虑更复杂的现实环境。

#### Distributed Coordinated Path Following Using Guiding Vector Fields

我们设计了一个引导向量场来引导多个机器人在协调其运动的同时遵循可能不同的所需路径。矢量场使用路径参数作为在相邻机器人之间通信的虚拟坐标。然后，利用虚拟坐标来控制机器人之间沿路径的相对参数位移。



### Session ThAT20 : Semantic Planning    

#### Towards Real-time Semantic RGB-D SLAM in Dynamic Environments

基于深度学习的语义信息引入 SLAM 系统来消除动态对象的影响

用于动态环境的实时语义 RGB-D SLAM 系统，该系统能够检测已知和未知的运动物体。为了降低计算成本，我们仅对关键帧执行语义分割以移除已知动态对象，并维护静态地图以实现稳健的相机跟踪。

我们提出了一个有效的几何模块，通过将深度图像聚类到几个区域并通过它们的重投影误差识别动态区域来检测未知的运动物体

#### (no paper)Real-Time Robot Path Planning Using Rapid Visible Tree

新的路径规划策略——快速可见树（RVT）算法

通过将可见性信息与经典的基于树的搜索方法融合，RVT 仅以从环境中局部获取的噪声点作为输入，计算每个位置的可见区域来决定路径树的生长方向

#### (no paper)Semantically Guided Multi-View Stereo for Dense 3D Road Mapping

低纹理区域（例如路面）留下孔洞和异常值。为此，本文提出了一种用于密集 3D 道路映射的新型语义引导多视图立体方法

该方法将语义信息集成到基于 PatchMatch 的 MVS 管道中，并使用图像语义分割作为邻居视图选择、深度图初始化、深度传播和深度图完成。

#### Spatial Reasoning from Natural Language Instructions for Robot Manipulation

机器人语义分析

我们提出了一个两阶段的流水线架构来对文本输入进行空间推理。



### Session ThHT23 : Motion Planning for Autonomous Vehicle

#### Path Optimization for Ground Vehicles in Off-Road Terrain

高速越野环境中对地面车辆进行路径优化的方法

数学方法

#### Robust & Asymptotically Locally Optimal UAV-Trajectory Generation Based on Spline Subdivision  

基于优化的本地无人机轨迹生成器

我们使用轨迹的渐近精确分段近似，其离散化分辨率可自动调节。轨迹在细化下收敛到精确非凸规划问题的一阶平稳点。

#### Vehicle Trajectory Prediction Using Generative Adversarial Network with Temporal Logic Syntax  Tree Features 

我们提出了一种将规则集成到交通代理轨迹预测中的新方法

基于**生成对抗网络**的框架，该框架使用来自形式方法的工具，即信号时间逻辑和语法树。 这使我们能够利用有关规则服从的信息作为神经网络中的特征，并在不偏向于合法行为的情况下提高预测准确性。

#### (no paper)Autonomous Vehicle Motion Planning Via Recurrent Spline Optimization



### Fast-Tracker: A Robust Aerial System for Tracking Agile Target in Cluttered Environments  

目标检测+追踪

### Teach-Repeat-Replan: A Complete and Robust System for Aggressive Flight in Complex Environments

我们提出了一种自动将可以任意抖动的无人驾驶轨迹转换为拓扑等距的方法借给一个。生成的轨迹平滑、安全、动态可行，具有更好的攻击性



### help find

#### Molecular Transport of a Magnetic Nanoparticle Swarm towards Thrombolytic Therapy

医学纳米机器人

#### Long-Range Hand Gesture Recognition Via Attention-Based SSD Network

手势识别在人机交互（HCI）领域发挥着重要作用。

#### Multi-Robot Informative Path Planning Using a Leader-Follower Architecture

我们考虑使用自主多机器人团队进行大区域调查的典型场景。所需的输出是感兴趣的物理值的空间图。考虑到空间相关性和不确定性，地图使用高斯过程建模。考虑到现实世界的约束，例如有限的时间预算和避免碰撞，我们将团队的任务建模为一个联合信息路径规划问题，该问题使用领导者-跟随者架构平衡集中和完全分布式计划计算来解决。领导者首先确定一个由团队采样的凸形收容区域。接下来，通过贝叶斯优化和蒙特卡罗模拟的组合，确定不同的采样位置并将其分配给跟随者。每个跟随者独立解决定向运动问题，以找到最大化信息增益的无碰撞路径。团队级别的自适应重新规划标准旨在将采样重新定向到信息量最大的区域。该算法已在地图估计的计算实验中得到验证。与基线参考算法相比，它显示出明显更高的准确性。此外，该方法显示出良好的支持网络连接的能力，以及良好的计算可扩展性。