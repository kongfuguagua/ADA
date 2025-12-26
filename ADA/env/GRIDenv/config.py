# -*- coding: utf-8 -*-
"""
L2RPN 环境配置模块

本模块定义了不同 L2RPN 比赛环境的配置。
支持 NeurIPS 2020（赛道1和赛道2）、ICAPS 2021 和 WCCI 2022 比赛。

使用方法:
    from config import EnvConfig, NEURIPS_2020_TRACK1, NEURIPS_2020_TRACK2, ICAPS_2021, WCCI_2022
    
    # 获取特定比赛的配置
    config = WCCI_2022
    
    # 或者创建自定义配置
    custom_config = EnvConfig(
        name="custom_env",
        env_name="l2rpn_case14_sandbox",
        ...
    )
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class Competition(Enum):
    """
    L2RPN 比赛枚举类
    
    定义了所有支持的 L2RPN 比赛类型
    """
    NEURIPS_2020_TRACK1 = "neurips_2020_track1"  # NeurIPS 2020 鲁棒性赛道
    NEURIPS_2020_TRACK2 = "neurips_2020_track2"  # NeurIPS 2020 适应性赛道
    ICAPS_2021 = "icaps_2021"                    # ICAPS 2021 信任赛道
    WCCI_2022 = "wcci_2022"                      # WCCI 2022 碳中和赛道


@dataclass
class EnvConfig:
    """
    L2RPN 环境初始化配置类
    
    属性说明:
        name: 配置的可读名称
        env_name: Grid2Op 环境名称（用于 grid2op.make()）
        competition: 比赛枚举值
        description: 比赛/环境的简要描述
        use_lightsim: 是否使用 LightSim2Grid 后端（更快的仿真）
        backend_class: 自定义后端类名（可选）
        action_class: 使用的动作类（如 TopologyAndDispatchAction）
        reward_class: 使用的奖励类（可选）
        observation_class: 使用的观测类（可选）
        difficulty: 难度级别（如适用）
        chronics_class: 时序数据处理类（可选）
        has_storage: 环境是否有储能单元
        has_renewable: 环境是否有可再生能源发电机
        has_curtailment: 是否可以进行弃风操作
        has_redispatch: 是否可以进行再调度操作
        has_alarm: 是否有告警机制
        max_episode_steps: 每个回合的最大步数（可选）
        thermal_limit_factor: 热限制调整因子（可选）
        extra_params: grid2op.make() 的额外参数
    """
    name: str                                    # 配置名称
    env_name: str                                # Grid2Op 环境标识符
    competition: Competition                      # 所属比赛
    description: str = ""                        # 环境描述
    
    # 后端配置
    use_lightsim: bool = True                    # 是否使用 LightSim2Grid（更快）
    backend_class: Optional[str] = None          # 自定义后端类
    
    # 动作/观测配置
    action_class: Optional[str] = None           # 动作空间类
    reward_class: Optional[str] = None           # 奖励函数类
    observation_class: Optional[str] = None      # 观测空间类
    
    # 环境特性
    difficulty: Optional[str] = None             # 难度设置
    chronics_class: Optional[str] = None         # 时序数据类
    
    # 功能标志
    has_storage: bool = False                    # 是否有储能
    has_renewable: bool = False                  # 是否有可再生能源
    has_curtailment: bool = False                # 是否支持弃风
    has_redispatch: bool = True                  # 是否支持再调度
    has_alarm: bool = False                      # 是否有告警系统
    
    # 仿真参数
    max_episode_steps: Optional[int] = None      # 最大回合步数
    thermal_limit_factor: Optional[float] = None # 热限制因子
    
    # 额外参数
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典，用于 grid2op.make() 调用
        
        返回:
            包含环境创建参数的字典
        """
        params = {
            "dataset": self.env_name,
        }
        if self.difficulty:
            params["difficulty"] = self.difficulty
        if self.extra_params:
            params.update(self.extra_params)
        return params


# =============================================================================
# NeurIPS 2020 比赛配置
# =============================================================================

NEURIPS_2020_TRACK1 = EnvConfig(
    name="NeurIPS 2020 - 赛道1（鲁棒性）",
    env_name="l2rpn_neurips_2020_track1_small",  # 可选: _small 或 _large
    competition=Competition.NEURIPS_2020_TRACK1,
    description="""
    鲁棒性赛道：开发能够应对突发事件的智能体。
    对抗性对手每天随机攻击电网线路。
    目标：克服攻击，尽可能长时间地运行电网，
    同时最小化运营成本（线路损耗、再调度成本、停电惩罚）。
    
    电网：IEEE 118 节点系统
    动作：拓扑变更、再调度
    挑战：处理对抗性线路攻击
    """,
    use_lightsim=True,
    action_class="TopologyAndDispatchAction",
    has_storage=False,       # 无储能
    has_renewable=False,     # 无可再生能源
    has_curtailment=False,   # 无弃风
    has_redispatch=True,     # 支持再调度
    has_alarm=False,         # 无告警
    max_episode_steps=7 * 288,  # 7天，每5分钟一步
)

NEURIPS_2020_TRACK2 = EnvConfig(
    name="NeurIPS 2020 - 赛道2（适应性）",
    env_name="l2rpn_neurips_2020_track2_small",  # 可选: _small 或 _large
    competition=Competition.NEURIPS_2020_TRACK2,
    description="""
    适应性赛道：开发能够适应新能源生产的智能体。
    重点关注可再生能源比例增加（可控性较差）。
    目标：适应可变的可再生能源发电，同时最小化运营成本。
    
    电网：IEEE 118 节点系统（针对可再生能源修改）
    动作：拓扑变更、再调度
    挑战：处理可变的可再生能源发电
    """,
    use_lightsim=True,
    action_class="TopologyAndDispatchAction",
    has_storage=False,       # 无储能
    has_renewable=True,      # 有可再生能源
    has_curtailment=False,   # 无弃风
    has_redispatch=True,     # 支持再调度
    has_alarm=False,         # 无告警
    max_episode_steps=7 * 288,  # 7天
)


# =============================================================================
# ICAPS 2021 比赛配置
# =============================================================================

ICAPS_2021 = EnvConfig(
    name="ICAPS 2021 - L2RPN 信任赛道",
    env_name="l2rpn_icaps_2021_small",  # 可选: _small 或 _large
    competition=Competition.ICAPS_2021,
    description="""
    L2RPN 信任赛道：关注电网运营中可信赖的人工智能。
    引入告警机制，实现人机协作。
    目标：可靠地运营电网，同时向操作员提供有意义的告警。
    
    电网：IEEE 118 节点系统（36个变电站，59条线路）
    动作：拓扑变更、再调度、告警
    挑战：平衡自动化与人工监督
    """,
    use_lightsim=True,
    action_class="TopologyAndDispatchAction",
    has_storage=False,       # 无储能
    has_renewable=True,      # 有可再生能源
    has_curtailment=False,   # 无弃风
    has_redispatch=True,     # 支持再调度
    has_alarm=True,          # 有告警系统
)


# =============================================================================
# WCCI 2022 比赛配置
# =============================================================================

WCCI_2022 = EnvConfig(
    name="WCCI 2022 - 未来能源与碳中和",
    env_name="l2rpn_wcci_2022",
    competition=Competition.WCCI_2022,
    description="""
    未来能源与碳中和赛道。
    引入储能单元和弃风功能，用于可再生能源管理。
    目标：处理可再生能源生产的不确定性，实现碳中和。
    
    电网：扩展电网（带储能单元）
    动作：
        - 离散动作：线路连接、变电站拓扑
        - 连续动作：再调度、弃风、储能控制
    挑战：利用储能和弃风管理可再生能源的不确定性
    """,
    use_lightsim=True,
    action_class=None,  # 使用默认的 PlayableAction
    has_storage=True,        # 有储能
    has_renewable=True,      # 有可再生能源
    has_curtailment=True,    # 支持弃风
    has_redispatch=True,     # 支持再调度
    has_alarm=False,         # 无告警
)


# =============================================================================
# 开发/测试配置
# =============================================================================

SANDBOX_CASE14 = EnvConfig(
    name="沙盒环境 - Case 14",
    env_name="l2rpn_case14_sandbox",
    competition=Competition.WCCI_2022,  # 使用最新特性
    description="""
    用于开发和测试的小型沙盒环境。
    基于 IEEE 14 节点系统。
    适用于快速原型设计和调试。
    """,
    use_lightsim=True,
    has_storage=False,       # 无储能
    has_renewable=False,     # 无可再生能源
    has_curtailment=False,   # 无弃风
    has_redispatch=True,     # 支持再调度
    has_alarm=False,         # 无告警
)


# =============================================================================
# 配置注册表
# =============================================================================

# 所有可用配置的字典
ENV_CONFIGS: Dict[str, EnvConfig] = {
    "neurips_2020_track1": NEURIPS_2020_TRACK1,
    "neurips_2020_track2": NEURIPS_2020_TRACK2,
    "icaps_2021": ICAPS_2021,
    "wcci_2022": WCCI_2022,
    "sandbox_case14": SANDBOX_CASE14,
}


def get_config(name: str) -> EnvConfig:
    """
    根据名称获取环境配置
    
    参数:
        name: 配置名称（如 "wcci_2022", "neurips_2020_track1"）
        
    返回:
        EnvConfig 对象
        
    异常:
        KeyError: 如果配置名称未找到
    """
    if name not in ENV_CONFIGS:
        available = list(ENV_CONFIGS.keys())
        raise KeyError(f"配置 '{name}' 未找到。可用配置: {available}")
    return ENV_CONFIGS[name]


def list_configs() -> List[str]:
    """
    列出所有可用的配置名称
    
    返回:
        配置名称列表
    """
    return list(ENV_CONFIGS.keys())


def print_config_info(config: EnvConfig) -> None:
    """
    打印配置的详细信息
    
    参数:
        config: 要打印的 EnvConfig 对象
    """
    print(f"\n{'='*60}")
    print(f"配置名称: {config.name}")
    print(f"{'='*60}")
    print(f"环境标识: {config.env_name}")
    print(f"所属比赛: {config.competition.value}")
    print(f"\n描述:{config.description}")
    print(f"\n功能特性:")
    print(f"  - 储能单元: {'是' if config.has_storage else '否'}")
    print(f"  - 可再生能源: {'是' if config.has_renewable else '否'}")
    print(f"  - 弃风功能: {'是' if config.has_curtailment else '否'}")
    print(f"  - 再调度: {'是' if config.has_redispatch else '否'}")
    print(f"  - 告警系统: {'是' if config.has_alarm else '否'}")
    print(f"\n后端: {'LightSim2Grid（快速）' if config.use_lightsim else 'PandaPower'}")
    if config.max_episode_steps:
        print(f"最大回合步数: {config.max_episode_steps}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 演示：打印所有可用配置
    print("可用的 L2RPN 环境配置:")
    print("-" * 40)
    for name in list_configs():
        config = get_config(name)
        print(f"  {name}: {config.name}")
    
    print("\n" + "="*60)
    print("详细配置信息:")
    for name in list_configs():
        print_config_info(get_config(name))
