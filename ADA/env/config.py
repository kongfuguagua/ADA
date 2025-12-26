# -*- coding: utf-8 -*-
"""
Grid2Op 环境配置
定义不同 L2RPN 比赛环境的配置
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
        env_name: Grid2Op 环境标识符
        competition: 所属比赛
        description: 环境描述
        use_lightsim: 是否使用 LightSim2Grid 后端
        action_class: 动作空间类
        has_storage: 是否有储能
        has_renewable: 是否有可再生能源
        has_curtailment: 是否支持弃风
        has_redispatch: 是否支持再调度
        max_episode_steps: 最大回合步数
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
    
    # 仿真参数
    max_episode_steps: Optional[int] = None
    
    # 额外参数
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为 grid2op.make() 参数"""
        params = {"dataset": self.env_name}
        if self.extra_params:
            params.update(self.extra_params)
        return params


# ============= 预定义配置 =============

WCCI_2022 = EnvConfig(
    name="WCCI 2022 - 碳中和",
    env_name="l2rpn_wcci_2022",
    competition=Competition.WCCI_2022,
    description="未来能源与碳中和赛道，包含储能单元和弃风功能",
    has_storage=True,
    has_renewable=True,
    has_curtailment=True,
    has_redispatch=True,
)

ICAPS_2021 = EnvConfig(
    name="ICAPS 2021 - 信任赛道",
    env_name="l2rpn_icaps_2021_small",
    competition=Competition.ICAPS_2021,
    description="关注电网运营中可信赖的人工智能",
    action_class="TopologyAndDispatchAction",
    has_renewable=True,
    has_redispatch=True,
)

NEURIPS_2020_TRACK1 = EnvConfig(
    name="NeurIPS 2020 - 鲁棒性",
    env_name="l2rpn_neurips_2020_track1_small",
    competition=Competition.NEURIPS_2020_TRACK1,
    description="鲁棒性赛道：应对对抗性攻击",
    action_class="TopologyAndDispatchAction",
    has_redispatch=True,
    max_episode_steps=7 * 288,
)

SANDBOX_CASE14 = EnvConfig(
    name="沙盒环境 - Case 14",
    env_name="l2rpn_case14_sandbox",
    competition=Competition.SANDBOX,
    description="用于开发和测试的小型沙盒环境",
    has_redispatch=True,
)

# 配置注册表
ENV_CONFIGS: Dict[str, EnvConfig] = {
    "wcci_2022": WCCI_2022,
    "icaps_2021": ICAPS_2021,
    "neurips_2020_track1": NEURIPS_2020_TRACK1,
    "sandbox_case14": SANDBOX_CASE14,
}


def get_env_config(name: str) -> EnvConfig:
    """获取环境配置"""
    if name not in ENV_CONFIGS:
        available = list(ENV_CONFIGS.keys())
        raise KeyError(f"配置 '{name}' 未找到。可用配置: {available}")
    return ENV_CONFIGS[name]


def list_env_configs() -> List[str]:
    """列出所有可用配置"""
    return list(ENV_CONFIGS.keys())


if __name__ == "__main__":
    print("可用的 Grid2Op 环境配置:")
    for name in list_env_configs():
        config = get_env_config(name)
        print(f"  - {name}: {config.name}")

