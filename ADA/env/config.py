# -*- coding: utf-8 -*-
"""
Grid2Op 环境配置
定义不同 L2RPN 比赛环境的配置

支持的环境：
- NeurIPS 2020 Track 1 (鲁棒性)
- NeurIPS 2020 Track 2 (适应性)
- ICAPS 2021 (信任)
- WCCI 2022 (碳中和)
- Sandbox (开发测试)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class Competition(Enum):
    """L2RPN 比赛枚举"""
    NEURIPS_2020_TRACK1 = "neurips_2020_track1"
    NEURIPS_2020_TRACK2 = "neurips_2020_track2"
    ICAPS_2021 = "icaps_2021"
    WCCI_2022 = "wcci_2022"
    SANDBOX = "sandbox"


@dataclass
class EnvConfig:
    """
    Grid2Op 环境配置
    
    Attributes:
        name: 配置名称
        env_name: Grid2Op 环境标识符（用于 grid2op.make()）
        competition: 所属比赛
        description: 环境描述
        use_lightsim: 是否使用 LightSim2Grid 后端（更快）
        action_class: 动作空间类名称
        has_storage: 是否有储能单元
        has_renewable: 是否有可再生能源发电机
        has_curtailment: 是否支持弃风操作
        has_redispatch: 是否支持再调度操作
        has_alarm: 是否有告警系统
        max_episode_steps: 最大回合步数
        extra_params: grid2op.make() 的额外参数
    """
    name: str
    env_name: str
    competition: Competition
    description: str = ""
    
    # 后端配置
    use_lightsim: bool = True
    action_class: Optional[str] = None
    
    # 功能标志
    has_storage: bool = False
    has_renewable: bool = False
    has_curtailment: bool = False
    has_redispatch: bool = True
    has_alarm: bool = False
    
    # 仿真参数
    max_episode_steps: Optional[int] = None
    
    # 额外参数
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为 grid2op.make() 参数
        
        Returns:
            包含环境创建参数的字典
        """
        params = {"dataset": self.env_name}
        if self.extra_params:
            params.update(self.extra_params)
        return params


# ============= 预定义配置 =============

# NeurIPS 2020 比赛配置
NEURIPS_2020_TRACK1 = EnvConfig(
    name="NeurIPS 2020 - 赛道1（鲁棒性）",
    env_name="l2rpn_neurips_2020_track1_small",
    competition=Competition.NEURIPS_2020_TRACK1,
    description="鲁棒性赛道：开发能够应对突发事件的智能体。对抗性对手每天随机攻击电网线路。",
    use_lightsim=True,
    action_class="TopologyAndDispatchAction",
    has_storage=False,
    has_renewable=False,
    has_curtailment=False,
    has_redispatch=True,
    has_alarm=False,
    max_episode_steps=7 * 288,  # 7天，每5分钟一步
)

NEURIPS_2020_TRACK2 = EnvConfig(
    name="NeurIPS 2020 - 赛道2（适应性）",
    env_name="l2rpn_neurips_2020_track2_small",
    competition=Competition.NEURIPS_2020_TRACK2,
    description="适应性赛道：开发能够适应新能源生产的智能体。重点关注可再生能源比例增加。",
    use_lightsim=True,
    action_class="TopologyAndDispatchAction",
    has_storage=False,
    has_renewable=True,
    has_curtailment=False,
    has_redispatch=True,
    has_alarm=False,
    max_episode_steps=7 * 288,
)

# ICAPS 2021 比赛配置
ICAPS_2021 = EnvConfig(
    name="ICAPS 2021 - 信任赛道",
    env_name="l2rpn_icaps_2021_small",
    competition=Competition.ICAPS_2021,
    description="L2RPN 信任赛道：关注电网运营中可信赖的人工智能。引入告警机制，实现人机协作。",
    use_lightsim=True,
    action_class="TopologyAndDispatchAction",
    has_storage=False,
    has_renewable=True,
    has_curtailment=False,
    has_redispatch=True,
    has_alarm=True,
)

# WCCI 2022 比赛配置
WCCI_2022 = EnvConfig(
    name="WCCI 2022 - 未来能源与碳中和",
    env_name="l2rpn_wcci_2022",
    competition=Competition.WCCI_2022,
    description="未来能源与碳中和赛道。引入储能单元和弃风功能，用于可再生能源管理。",
    use_lightsim=True,
    action_class=None,  # 使用默认的 PlayableAction
    has_storage=True,
    has_renewable=True,
    has_curtailment=True,
    has_redispatch=True,
    has_alarm=False,
)

# 开发/测试配置
SANDBOX_CASE14 = EnvConfig(
    name="沙盒环境 - Case 14",
    env_name="l2rpn_case14_sandbox",
    competition=Competition.SANDBOX,
    description="用于开发和测试的小型沙盒环境。基于 IEEE 14 节点系统。",
    use_lightsim=True,
    has_storage=False,
    has_renewable=False,
    has_curtailment=False,
    has_redispatch=True,
    has_alarm=False,
)

# 配置注册表
ENV_CONFIGS: Dict[str, EnvConfig] = {
    "neurips_2020_track1": NEURIPS_2020_TRACK1,
    "neurips_2020_track2": NEURIPS_2020_TRACK2,
    "icaps_2021": ICAPS_2021,
    "wcci_2022": WCCI_2022,
    "sandbox_case14": SANDBOX_CASE14,
}


def get_env_config(name: str) -> EnvConfig:
    """
    根据名称获取环境配置
    
    Args:
        name: 配置名称（如 "wcci_2022", "neurips_2020_track1"）
        
    Returns:
        EnvConfig 对象
        
    Raises:
        KeyError: 如果配置名称未找到
    """
    if name not in ENV_CONFIGS:
        available = list(ENV_CONFIGS.keys())
        raise KeyError(f"配置 '{name}' 未找到。可用配置: {available}")
    return ENV_CONFIGS[name]


def list_env_configs() -> List[str]:
    """
    列出所有可用的配置名称
    
    Returns:
        配置名称列表
    """
    return list(ENV_CONFIGS.keys())


def print_config_info(config: EnvConfig) -> None:
    """
    打印配置的详细信息
    
    Args:
        config: 要打印的 EnvConfig 对象
    """
    print(f"\n{'='*60}")
    print(f"配置名称: {config.name}")
    print(f"{'='*60}")
    print(f"环境标识: {config.env_name}")
    print(f"所属比赛: {config.competition.value}")
    print(f"\n描述: {config.description}")
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
    print("可用的 Grid2Op 环境配置:")
    for name in list_env_configs():
        config = get_env_config(name)
        print(f"  - {name}: {config.name}")

