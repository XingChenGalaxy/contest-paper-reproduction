# ConTest 项目：Lyapunov 函数简化复现

## 运行方式

`contest_lyapunov_demo.py` 使用一套不区分 PX4 和 ArduPilot 的抽象模拟参数，复现论文问题三中的核心数据路径：参数经过映射后影响闭环状态轨迹，实际状态与参考状态形成误差，误差再代入 Lyapunov 二次型。

```bash
python contest_lyapunov_demo.py
```

依赖 Python 3.9+ 和 NumPy。程序不读取PX4/ArduPilot 文件、飞控日志或外部参数配置。

数据路径可以表示为：

```text
抽象参数 -> 参数效果映射 -> 闭环状态轨迹
                              -> 实际状态与参考状态
                              -> 误差 -> Lyapunov 值
```

## 论文中的数学对象

论文使用如下 12 维状态向量：

$$
x_t=[p_n,p_e,p_d,\phi,\theta,\psi,v_u,v_v,v_w,r_p,r_q,r_r]^T.
$$

其中前三项是位置，接着三项是横滚、俯仰、偏航角，再接三项线速度和三项角速度。程序中的 `STATE_NAMES` 使用相同顺序。

跟踪误差定义为：

$$
e(t)=x(t)-x^r(t).
$$

Lyapunov 函数定义为：

$$
V(e(t))=\frac{1}{2}e(t)^T P_t e(t).
$$

附录中的局部线性系统和 Lyapunov 方程写作：

$$
\dot e(t)=A_t e(t),
\qquad
A_t^T P_t+P_tA_t=-Q,
\qquad Q\succ0.
$$

论文没有公开完整的逐时刻状态轨迹、局部矩阵 $A_t$、矩阵 $Q$ 和矩阵 $P_t$ 数值，因此当前程序只能复现计算链路，不能声称复现论文作者的原始数值结果。

## 当前模拟参数

程序只构造一套 `generic_simulator` 参数，不区分两个飞控平台。参数值是为了让示例稳定、可重复，并保留参数对状态轨迹的影响路径而设计的模拟值。

| 参数 | 数值 | 设计依据 |
|---|---:|---|
| `POS_XY_P` | 1.0 | 水平位置比例反馈基准 |
| `VEL_XY_P` | 2.0 | 加强水平速度反馈，为位置误差提供阻尼 |
| `VEL_XY_I` | 0.5 | 消除慢性位置偏差，同时避免积分作用过强 |
| `VEL_XY_D` | 0.1 | 提供小幅速度阻尼，抑制过冲 |
| `POS_Z_P` | 0.5 | 垂直位置反馈较温和，避免与水平控制差异过大 |
| `VEL_Z_P` | 1.0 | 垂直速度比例反馈 |
| `VEL_Z_I` | 1.0 | 补偿垂直方向的持续误差 |
| `VEL_Z_D` | 0.02 | 垂直方向的小阻尼项 |
| `ROLL_P` / `PITCH_P` | 0.8 / 0.8 | 姿态反馈保持在与位置控制相近的数量级 |
| `YAW_P` | 0.6 | 偏航反馈略弱于横滚和俯仰 |
| `GPS_X_BIAS` / `GPS_Y_BIAS` | 0 / 0 | 基准运行不额外注入水平传感器偏置 |
| `ACTUATOR_FLOOR` | 0.01 | 保留一个很小的执行器偏置影响 |

## 模拟矩阵、轨迹和控制

`local_weight_matrix()` 使用如下演示矩阵：

$$
P_t=\operatorname{diag}(1,1,1.5,2,2,1.5,0.8,0.8,1,0.4,0.4,0.4)\alpha(t),
$$

其中：

$$
\alpha(t)=1+0.08\sin(0.1t).
$$

每个对角元素为正，且 $\alpha(t)\in[0.92,1.08]$，所以 $P_t$ 始终正定。位置和姿态权重相对较大，是为了使主要任务误差在 $V(e(t))$ 中更明显；速度和角速度权重较小，是为了避免单位和量级差异完全支配结果。这个矩阵是演示构造，不是论文公开的真实矩阵。

参考轨迹使用连续低频正弦函数。位置幅值约为 2 m，垂直位置幅值为 0.25 m，姿态角幅值为 0.04–0.08 rad，频率为 0.10–0.30 rad/s。轨迹保持平滑并处于小扰动范围内。部分速度项与位置项的导数一致，例如：

$$
p_n(t)=2\sin(0.15t),
\qquad
v_u(t)=0.30\cos(0.15t).
$$

控制器由位置比例、位置误差积分、速度比例和速度阻尼组成。线速度阻尼系数为 0.35，角速度阻尼系数为 0.25，姿态目标耦合系数为 0.08，偏航目标系数为 0.02。加速度控制量限制在 $[-8,8]$，积分状态限制在 $[-2,2]$。显式 Euler 仿真步长为 $0.02\,\mathrm{s}$，即 50 Hz，总时长为 20 s。

离散化公式为：

$$
x_{k+1}=x_k+\Delta t\,\dot{x}_k.
$$

实际状态初始值相对于参考状态加入位置扰动 $(0.2,-0.1,0.1)\,\mathrm{m}$ 和姿态扰动 $(0.03,-0.02,0.02)\,\mathrm{rad}$，以便从非零误差开始计算 $V(e(t))$。

## 程序与论文概念的对应

| 程序函数 | 对应概念 | 当前实现 |
|---|---|---|
| `make_demo_parameter_set()` | 参数输入 | 一套抽象模拟参数 |
| `PARAMETER_TO_EFFECT` | 参数依赖映射 | 手工定义的简化映射表 |
| `reduce_and_map_parameters()` | 参数筛选和映射 | 将参数转成控制效果 |
| `normalized_control_gains()` | 控制器增益整理 | 生成简化控制器使用的增益 |
| `reference_state()` | 参考状态 $x^r(t)$ | 可重复的 12 维正弦轨迹 |
| `controller_acceleration()` | 控制输入影响 | 简化位置、速度 PI/D 反馈 |
| `dynamics_step()` | 动力学推进 | 12 维显式 Euler 模型 |
| `local_weight_matrix()` | 时变 $P_t$ | 正定演示矩阵 |
| `lyapunov_values()` | Lyapunov 评价 | 计算 $V(e(t))$ |
| `finite_difference_sensitivity()` | 参数影响估计 | 对最小 $V$ 做中心差分 |

## 与论文完整实现的边界

论文中的 ConTest 还包含 LLVM 数据流/控制流分析、真实 PX4/ArduPilot 参数筛选、系统辨识、SITL/HITL 日志、任务完成性检查以及开发者确认流程。当前程序没有实现这些模块，只用于说明：

$$
\lambda
\longrightarrow \text{参数效果}
\longrightarrow x(t)
\longrightarrow e(t)
\longrightarrow V(e(t)).
$$

参数不是直接作为 Lyapunov 函数的坐标输入，而是先改变控制器、传感器偏置或执行器项，进而改变状态轨迹和误差，最后影响 $V(e(t))$。

## 相关文件

- [contest_lyapunov_demo.py](contest_lyapunov_demo.py)：可运行的简化模拟程序。
- [ConTest_论文五个问题详解.md](ConTest_论文五个问题详解.md)：论文问题、实验、消融实验和 Lyapunov 作用分析。
