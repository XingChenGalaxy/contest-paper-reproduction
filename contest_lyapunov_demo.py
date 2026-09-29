"""ConTest 参数到 Lyapunov 值的单参数集简化复现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple
import math
import numpy as np

Array = np.ndarray

STATE_NAMES = (
    # 该顺序对应论文中的 12 维状态向量。
    "p_n", "p_e", "p_d",       # 北、东、下位置
    "phi", "theta", "psi",     # 横滚、俯仰、偏航角
    "v_u", "v_v", "v_w",        # 三轴线速度
    "r_p", "r_q", "r_r",        # 三轴角速度
)
STATE_DIM = len(STATE_NAMES)



@dataclass(frozen=True)
class ParameterSet:
    platform: str
    values: Mapping[str, float]


def make_demo_parameter_set() -> ParameterSet:
    """构造一套不区分飞控平台的抽象模拟参数。"""
    return ParameterSet(
        platform="generic_simulator",
        values={
            "POS_XY_P": 1.0,
            "VEL_XY_P": 2.0,
            "VEL_XY_I": 0.5,
            "VEL_XY_D": 0.1,
            "POS_Z_P": 0.5,
            "VEL_Z_P": 1.0,
            "VEL_Z_I": 1.0,
            "VEL_Z_D": 0.02,
            "ROLL_P": 0.8,
            "PITCH_P": 0.8,
            "YAW_P": 0.6,
            "GPS_X_BIAS": 0.0,
            "GPS_Y_BIAS": 0.0,
            "ACTUATOR_FLOOR": 0.01,
        },
    )



PARAMETER_TO_EFFECT = {
    "POS_XY_P": ("xy_position_gain", 1.0),
    "VEL_XY_P": ("xy_velocity_gain", 1.0),
    "VEL_XY_I": ("xy_integral_gain", 1.0),
    "VEL_XY_D": ("xy_damping_gain", 1.0),
    "POS_Z_P": ("z_position_gain", 1.0),
    "VEL_Z_P": ("z_velocity_gain", 1.0),
    "VEL_Z_I": ("z_integral_gain", 1.0),
    "VEL_Z_D": ("z_damping_gain", 1.0),
    "ROLL_P": ("roll_gain", 1.0),
    "PITCH_P": ("pitch_gain", 1.0),
    "YAW_P": ("yaw_gain", 1.0),
    "GPS_X_BIAS": ("gps_x_bias", 1.0),
    "GPS_Y_BIAS": ("gps_y_bias", 1.0),
    "ACTUATOR_FLOOR": ("actuator_floor", 1.0),
}

def reduce_and_map_parameters(parameters: Mapping[str, float]) -> Dict[str, float]:
    """把平台参数映射成简化模型能够使用的控制效果。"""
    effects: Dict[str, float] = {}
    for name, value in parameters.items():
        # 未出现在映射表中的参数不会进入本示例动力学。
        mapping = PARAMETER_TO_EFFECT.get(name)
        if mapping is None:
            continue
        effect_name, scale = mapping
        effects[effect_name] = effects.get(effect_name, 0.0) + float(value) * scale
    return effects


def normalized_control_gains(effects: Mapping[str, float]) -> Dict[str, float]:
    """把映射结果整理成控制器使用的增益字典。"""
    def positive(name: str, default: float) -> float:
        # 避免负增益或零增益使示例模型失去稳定反馈。
        return max(float(effects.get(name, default)), 1e-4)

    return {
        "xy_p": positive("xy_position_gain", 1.0),
        "xy_v": positive("xy_velocity_gain", 1.0),
        "xy_i": positive("xy_integral_gain", 0.1),
        "xy_d": positive("xy_damping_gain", 0.1),
        "z_p": positive("z_position_gain", 1.0),
        "z_v": positive("z_velocity_gain", 1.0),
        "z_i": positive("z_integral_gain", 0.1),
        "z_d": positive("z_damping_gain", 0.1),
        "roll": positive("roll_gain", 1.0),
        "pitch": positive("pitch_gain", 1.0),
        "yaw": positive("yaw_gain", 1.0),
        "gps_x": float(effects.get("gps_x_bias", 0.0)),
        "gps_y": float(effects.get("gps_y_bias", 0.0)),
        "actuator_floor": float(effects.get("actuator_floor", 0.0)),
    }



def reference_state(t: float) -> Array:
    """根据时间 t 生成 12 维参考状态 x^r(t)。"""
    x = np.zeros(STATE_DIM, dtype=float)
    x[0] = 2.0 * math.sin(0.15 * t)
    x[1] = 2.0 * math.cos(0.15 * t) - 2.0
    x[2] = -1.0 - 0.25 * math.sin(0.10 * t)
    x[3] = 0.04 * math.sin(0.30 * t)
    x[4] = 0.04 * math.cos(0.25 * t)
    x[5] = 0.08 * math.sin(0.20 * t)
    x[6] = 0.30 * math.cos(0.15 * t)
    x[7] = -0.30 * math.sin(0.15 * t)
    x[8] = -0.025 * math.cos(0.10 * t)
    x[9] = 0.012 * math.cos(0.30 * t)
    x[10] = -0.010 * math.sin(0.25 * t)
    x[11] = 0.016 * math.cos(0.20 * t)
    return x


def local_weight_matrix(t: float) -> Array:
    """返回演示用的正定 P_t；论文没有公开逐时刻的真实 P_t 数值。"""
    # 这些权重只用于构造可运行示例，不能当作论文实验数据。
    weights = np.array([
        1.0, 1.0, 1.5, 2.0, 2.0, 1.5,
        0.8, 0.8, 1.0, 0.4, 0.4, 0.4,
    ])
    # 用小幅时间调制模拟论文中的时变 P_t。
    modulation = 1.0 + 0.08 * math.sin(0.1 * t)
    return np.diag(weights * modulation)


def controller_acceleration(error: Array, integral: Array, gains: Mapping[str, float]) -> Array:
    """根据位置、积分和速度误差计算简化的三轴加速度控制量。"""
    pos = error[[0, 1, 2]]
    vel = error[[6, 7, 8]]
    acc = np.zeros(3)
    # 负反馈使位置和速度误差趋向参考轨迹。
    acc[0] = -(gains["xy_p"] * pos[0] + gains["xy_i"] * integral[0]
               + (gains["xy_v"] + gains["xy_d"]) * vel[0])
    acc[1] = -(gains["xy_p"] * pos[1] + gains["xy_i"] * integral[1]
               + (gains["xy_v"] + gains["xy_d"]) * vel[1])
    acc[2] = -(gains["z_p"] * pos[2] + gains["z_i"] * integral[2]
               + (gains["z_v"] + gains["z_d"]) * vel[2])
    # 模拟执行器饱和，避免单步控制量无限增大。
    return np.clip(acc, -8.0, 8.0)


def dynamics_step(state: Array, reference: Array, integral: Array,
                  gains: Mapping[str, float], dt: float) -> Tuple[Array, Array]:
    """用显式 Euler 法推进一个时间步，并返回新状态和积分误差。"""
    error = state - reference
    # 只对三个位置信息积分，并限制积分器范围。
    new_integral = np.clip(integral + error[[0, 1, 2]] * dt, -2.0, 2.0)
    accel = controller_acceleration(error, new_integral, gains)
    dx = np.zeros(STATE_DIM)
    # 位置导数是线速度，姿态角导数用角速度近似。
    dx[0:3] = state[6:9]
    dx[3:6] = state[9:12]
    dx[6:9] = accel - 0.35 * state[6:9]
    desired_roll = 0.08 * accel[1]
    desired_pitch = -0.08 * accel[0]
    desired_yaw = 0.02 * (gains["xy_p"] - gains["z_p"])
    dx[9] = gains["roll"] * (desired_roll - state[3]) - 0.25 * state[9]
    dx[10] = gains["pitch"] * (desired_pitch - state[4]) - 0.25 * state[10]
    dx[11] = gains["yaw"] * (desired_yaw - state[5]) - 0.25 * state[11]
    dx[0] += gains["gps_x"]
    dx[1] += gains["gps_y"]
    dx[6:9] -= gains["actuator_floor"]
    # Euler 离散化：x_{k+1}=x_k+dt*dx_k。
    return state + dt * dx, new_integral


def simulate(parameters: ParameterSet, horizon: float = 20.0,
             dt: float = 0.02) -> Tuple[Array, Array, Array]:
    """模拟实际轨迹、参考轨迹和采样时间。"""
    effects = reduce_and_map_parameters(parameters.values)
    gains = normalized_control_gains(effects)
    times = np.arange(0.0, horizon + 0.5 * dt, dt)
    actual = np.zeros((len(times), STATE_DIM))
    refs = np.vstack([reference_state(t) for t in times])
    # 初始状态相对参考状态加入一个小扰动。
    actual[0] = refs[0] + np.array([0.2, -0.1, 0.1, 0.03, -0.02, 0.02,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    integral = np.zeros(3)
    # 逐步调用动力学，得到整条实际状态轨迹。
    for k in range(len(times) - 1):
        actual[k + 1], integral = dynamics_step(
            actual[k], refs[k], integral, gains, dt
        )
    return times, actual, refs



def lyapunov_values(times: Array, actual: Array, refs: Array) -> Array:
    """按 V_t = 1/2 e_t^T P_t e_t 计算整条轨迹的 Lyapunov 值。"""
    values = np.empty(len(times))
    for k, t in enumerate(times):
        # 论文定义误差 e(t) 为实际状态减去参考状态。
        error = actual[k] - refs[k]
        P_t = local_weight_matrix(float(t))
        # 将误差代入二次型，得到一个非负的标量 V_t。
        values[k] = 0.5 * float(error.T @ P_t @ error)
    return values


def finite_difference_sensitivity(base: ParameterSet, parameter: str,
                                  delta: float = 1e-3) -> float:
    """用中心差分估计参数对最小 Lyapunov 值的影响。"""
    original = float(base.values[parameter])
    low = dict(base.values)
    high = dict(base.values)
    # 分别运行参数减小和增大的两次仿真。
    low[parameter] = original - delta
    high[parameter] = original + delta
    low_t, low_x, low_r = simulate(ParameterSet(base.platform, low))
    high_t, high_x, high_r = simulate(ParameterSet(base.platform, high))
    low_min = float(np.min(lyapunov_values(low_t, low_x, low_r)))
    high_min = float(np.min(lyapunov_values(high_t, high_x, high_r)))
    # 中心差分近似 d(min V)/d(parameter)。
    return (high_min - low_min) / (2.0 * delta)


def print_report(parameters: ParameterSet) -> None:
    """打印一套参数对应的轨迹、V 值和敏感度摘要。"""
    times, actual, refs = simulate(parameters)
    effects = reduce_and_map_parameters(parameters.values)
    gains = normalized_control_gains(effects)
    values = lyapunov_values(times, actual, refs)
    print(f"\n=== {parameters.platform} ===")
    print("retained state-related effects:")
    for key, value in sorted(effects.items()):
        print(f"  {key:20s} {value: .6f}")
    print("state dimension:", STATE_DIM)
    print("P_t dimension:", local_weight_matrix(0.0).shape)
    print("trajectory samples:", len(times))
    print("V(0)       =", f"{values[0]:.8f}")
    print("min V(t)   =", f"{values.min():.8f}")
    print("max V(t)   =", f"{values.max():.8f}")
    print("V(end)     =", f"{values[-1]:.8f}")

    candidates = [
        name for name in parameters.values
        if name in PARAMETER_TO_EFFECT
    ]
    print("finite-difference sensitivities of min V:")
    for name in candidates[:4]:
        sensitivity = finite_difference_sensitivity(parameters, name)
        print(f"  d(min V)/d({name}) = {sensitivity: .6f}")


def main() -> None:
    # 使用同一套抽象参数完整走一遍流程。
    parameters = make_demo_parameter_set()
    print_report(parameters)


if __name__ == "__main__":
    main()
