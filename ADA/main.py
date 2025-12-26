# -*- coding: utf-8 -*-
"""
ADA (Agile Dispatch Agent) 系统入口
知识驱动的复杂系统敏捷调度智能体

功能：
1. 启动配置 - 解析命令行参数、加载 YAML 配置
2. 环境初始化 - 创建 Grid2Op 环境（可选）
3. Agent 初始化 - 创建并配置各智能体
4. 日志/进度打印 - 统一日志输出和进度显示
5. SwanLab 绘图 - 实验指标可视化

使用方式:
    python main.py -f yaml/default.yaml
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入配置
from ADA.env import Grid2OpEnvironment
from config import SystemConfig, LLMConfig
from config.yaml_loader import (
    load_config_from_yaml,
    create_system_config_from_yaml,
    create_llm_config_from_yaml,
)

# 导入数据契约
from utils.const import EnvironmentState
from utils.logger import get_logger

# 导入编排器
from orchestrator import ADAOrchestrator

# 导入环境（可选）
try:
    from env import create_grid2op_env
    HAS_GRID2OP = True
except ImportError as e:
    HAS_GRID2OP = False
    logger.warning(f"Grid2Op 环境模块未找到: {e}，将使用模拟环境")

logger = get_logger("Main")

# SwanLab 初始化（可选）
try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False
    print("提示: SwanLab 未安装，将跳过实验可视化。安装: pip install swanlab")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ADA - Agile Dispatch Agent 系统启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py -f yaml/default.yaml
  python main.py -f yaml/default.yaml --verbose
  python main.py -f yaml/default.yaml --no-swanlab
        """
    )
    
    parser.add_argument(
        "-f", "--config",
        type=str,
        default=r'C:\Users\a2550\Desktop\ADA\ADA\yaml\default.yaml',
        help="YAML 配置文件路径（如: yaml/default.yaml）"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细信息"
    )
    
    parser.add_argument(
        "--no-swanlab",
        action="store_true",
        help="禁用 SwanLab 可视化"
    )
    
    return parser.parse_args()


def init_swanlab(config_dict: Dict[str, Any], enabled: bool = True) -> Optional[Any]:
    """
    初始化 SwanLab
        
        Args:
        config_dict: YAML 配置字典
        enabled: 是否启用
        
        Returns:
        SwanLab 实例（如果启用）或 None
    """
    if not enabled or not HAS_SWANLAB:
        return None
    
    experiment_config = config_dict.get("experiment", {})
    swanlab_config = experiment_config.get("swanlab", {})
    
    if not swanlab_config.get("enabled", True):
        return None
    
    try:
        run = swanlab.init(
            project=swanlab_config.get("project", "ada"),
            experiment_name=swanlab_config.get("experiment_name", experiment_config.get("name", "ada_experiment")),
            description=swanlab_config.get("description", "ADA 系统实验"),
            tags=swanlab_config.get("tags", ["ada"]),
        )
        logger.info("SwanLab 已初始化")
        return run
    except Exception as e:
        logger.warning(f"SwanLab 初始化失败: {e}")
        return None


def init_environment(config_dict: Dict[str, Any])->Grid2OpEnvironment:
    """
    初始化环境
    
    Args:
        config_dict: YAML 配置字典
    
    Returns:
        环境实例（Grid2Op 或 None）
    """
    env_config = config_dict.get("env", {})
    env_type = env_config.get("type", "none")
    
    if env_type == "grid2op" and HAS_GRID2OP:
        env_name = env_config.get("env_name", "l2rpn_wcci_2022")
        logger.info(f"初始化 Grid2Op 环境: {env_name}")
        try:
            env = create_grid2op_env(env_name)
            # 创建环境后立即 reset，确保有初始观测
            env.reset()
            logger.info("Grid2Op 环境初始化成功并已重置")
            return env
        except Exception as e:
            logger.error(f"Grid2Op 环境初始化失败: {e}")
            logger.info("将使用模拟环境")
            return None
    elif env_type == "mock":
        logger.info("使用模拟环境")
        return None
    else:
        logger.info("未配置环境，使用默认模拟环境")
        return None


def print_progress(episode: int, step: int, result: Dict[str, Any], verbose: bool = False):
    """
    打印进度信息
    
    Args:
        episode: 当前回合数
        step: 当前步数
        result: 运行结果
        verbose: 是否打印详细信息
    """
    if result["success"]:
        status = "✓"
        score = result.get("feedback", {}).score if result.get("feedback") else 0.0
        print(f"[Episode {episode}, Step {step}] {status} 成功 | 评分: {score:.4f}")
        
        if verbose and result.get("solution"):
            solution = result["solution"]
            print(f"  算法: {solution.algorithm_used}")
            print(f"  目标值: {solution.objective_value:.4f}")
    else:
        status = "✗"
        diagnosis = result.get("feedback", {}).diagnosis if result.get("feedback") else "未知错误"
        print(f"[Episode {episode}, Step {step}] {status} 失败 | {diagnosis[:50]}")


def log_to_swanlab(swanlab_run: Any, metrics: Dict[str, float], step: int):
    """
    记录指标到 SwanLab
    
    Args:
        swanlab_run: SwanLab 实例
        metrics: 指标字典
        step: 当前步数
    """
    if swanlab_run is None:
        return
    
    try:
        swanlab.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"SwanLab 记录失败: {e}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 打印启动信息
    print("=" * 60)
    print("  ADA - Agile Dispatch Agent")
    print("  知识驱动的复杂系统敏捷调度智能体")
    print("=" * 60)
    print()
    
    # 加载 YAML 配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return None
    
    logger.info(f"加载配置文件: {config_path}")
    try:
        config_dict = load_config_from_yaml(str(config_path))
        system_config = create_system_config_from_yaml(config_dict)
        llm_config = create_llm_config_from_yaml(config_dict)
    except Exception as e:
        print(f"错误: 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 验证配置
    try:
        llm_config.validate()
        llm_config.validate_embedding()
    except ValueError as e:
        print(f"错误: 配置验证失败: {e}")
        return None
    
    print(f"✓ 配置加载成功")
    print(f"  Chat 模型: {llm_config.model_name}")
    print(f"  Embedding 模型: {llm_config.embedding_model}")
    print()
    
    # 初始化 SwanLab
    swanlab_run = None
    if not args.no_swanlab:
        swanlab_run = init_swanlab(config_dict)
    
    # 初始化环境
    env = init_environment(config_dict)
    
    # 创建编排器
    try:
        orchestrator = ADAOrchestrator(
            system_config=system_config,
            llm_config=llm_config,
            env=env
        )
        logger.info("ADA 系统初始化完成")
    except Exception as e:
        print(f"错误: 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 获取实验配置
    experiment_config = config_dict.get("experiment", {})
    max_episodes = experiment_config.get("max_episodes", 1)
    max_steps_per_episode = experiment_config.get("max_steps_per_episode", 100)
    verbose = args.verbose or experiment_config.get("verbose", False)
    
    print(f"开始运行实验:")
    print(f"  最大回合数: {max_episodes}")
    print(f"  每回合最大步数: {max_steps_per_episode}")
    print()
    
    # 运行实验
    all_results = []
    total_success = 0
    total_steps = 0
    
    for episode in range(max_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {episode + 1}/{max_episodes}")
        print(f"{'=' * 60}")
        
        episode_results = []
        episode_success = 0
        
        for step in range(max_steps_per_episode):
            total_steps += 1
            
            # 创建环境状态
            # 如果 env 为 None，orchestrator.run() 会自动使用默认状态
            # 如果 env 存在，orchestrator.run() 会从环境获取状态
            env_state = None  # 让 orchestrator 自动处理
            
            # 运行主循环
            result = orchestrator.run(env_state=env_state)
            episode_results.append(result)
            
            # 打印进度
            print_progress(episode + 1, step + 1, result, verbose)
            
            # 记录到 SwanLab
            if swanlab_run is not None:
                metrics = {
                    "success": 1.0 if result["success"] else 0.0,
                    "score": result.get("feedback", {}).score if result.get("feedback") else 0.0,
                    "attempts": result.get("attempts", 0),
                }
                if result.get("solution"):
                    metrics["objective_value"] = result["solution"].objective_value
                log_to_swanlab(swanlab_run, metrics, total_steps)
            
            if result["success"]:
                episode_success += 1
                total_success += 1
                
                # 如果使用 Grid2Op 环境，执行动作
                if env is not None:
                    # TODO: 将 solution 转换为 Grid2Op action
                    pass
        
        all_results.append(episode_results)
        
        # 打印回合统计
        success_rate = episode_success / max_steps_per_episode if max_steps_per_episode > 0 else 0
        print(f"\nEpisode {episode + 1} 完成:")
        print(f"  成功次数: {episode_success}/{max_steps_per_episode}")
        print(f"  成功率: {success_rate:.2%}")
    
    # 打印总体统计
    print(f"\n{'=' * 60}")
    print("实验完成")
    print(f"{'=' * 60}")
    overall_success_rate = total_success / total_steps if total_steps > 0 else 0
    print(f"总步数: {total_steps}")
    print(f"总成功次数: {total_success}")
    print(f"总体成功率: {overall_success_rate:.2%}")
    
    # 系统状态
    status = orchestrator.get_status()
    print(f"\n系统状态:")
    print(f"  知识库条目: {status['knowledge_count']}")
    print(f"  可用算法: {status['algorithms']}")
    print(f"  可用工具: {status['tools']}")
    print(f"  环境状态: {'已连接' if status['has_env'] else '未连接'}")
    
    # 记录最终指标到 SwanLab
    if swanlab_run is not None:
        final_metrics = {
            "final_success_rate": overall_success_rate,
            "total_steps": total_steps,
            "total_success": total_success,
            "knowledge_count": status['knowledge_count'],
        }
        log_to_swanlab(swanlab_run, final_metrics, total_steps)
        swanlab.finish()
    
    return {
        "results": all_results,
        "total_steps": total_steps,
        "total_success": total_success,
        "success_rate": overall_success_rate,
        "status": status,
    }


if __name__ == "__main__":
    result = main()
    
    if result:
        print()
        print("=" * 60)
        print("✓ 运行完成")
        print("=" * 60)
