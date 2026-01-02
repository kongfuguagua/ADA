# -*- coding: utf-8 -*-
"""
ReAct Agent 核心实现 (Fix: Logic Bug & Auto-Correction)
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import numpy as np # 确保导入 numpy

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction

try:
    from .formatters import ObservationFormatter
    from .parser import ActionParser
    from .prompts import PromptManager
except ImportError:
    from ReAct_Baseline.formatters import ObservationFormatter
    from ReAct_Baseline.parser import ActionParser
    from ReAct_Baseline.prompts import PromptManager

# 尝试导入 ADA 的工具
from utils import OpenAIChat, get_logger
from utils.embeddings import OpenAIEmbedding

# 尝试导入 ADA 的知识库服务
try:
    from ADA.knowledgebase.service import KnowledgeService
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

logger = get_logger("ReActAgent")

if not RAG_AVAILABLE:
    logger.warning("RAG 功能不可用：无法导入 KnowledgeService")


class ReActAgent(BaseAgent):
    """
    ReAct Baseline Agent (Enhanced v2.1)
    
    主要改进：
    1. RAG 集成与预验证
    2. 量化仿真反馈
    3. 动作剪枝
    4. 上下文压缩
    5. 配置抽象
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        "rho_danger": 0.92,  # 危险阈值（预防性调度）
        "rho_safe": 0.80,    # 安全阈值（提前终止）
        "min_redispatch_threshold": 0.5,  # 动作剪枝阈值（MW）
        "rag_top_k": 2,  # RAG 检索返回条数
        "enable_rag": True,
    }
    
    def __init__(
        self,
        action_space,
        observation_space,
        llm_client: OpenAIChat,
        max_react_steps: int = 3,
        name: str = "ReActAgent",
        rho_danger: float = None,
        rho_safe: float = None,
        min_redispatch_threshold: float = None,
        enable_rag: bool = True,
        knowledge_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(action_space)
        self.name = name
        self.observation_space = observation_space
        self.llm_client = llm_client
        self.max_react_steps = max_react_steps
        
        # 配置管理（使用传入值或默认值）
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(kwargs.get("config", {}))
        if rho_danger is not None:
            self.config["rho_danger"] = rho_danger
        if rho_safe is not None:
            self.config["rho_safe"] = rho_safe
        if min_redispatch_threshold is not None:
            self.config["min_redispatch_threshold"] = min_redispatch_threshold
        if "enable_rag" in kwargs:
            self.config["enable_rag"] = kwargs["enable_rag"]
        else:
            self.config["enable_rag"] = enable_rag
        
        # 便捷访问常用配置
        self.rho_danger = self.config["rho_danger"]
        self.rho_safe = self.config["rho_safe"]
        self.min_redispatch_threshold = self.config["min_redispatch_threshold"]
        
        self.formatter = ObservationFormatter()
        self.parser = ActionParser()
        self.prompt_manager = PromptManager()
        
        self.current_step = 0
        self.react_history = []
        self.env_info = None
        
        self.stats = {
            "total_steps": 0,
            "react_loops": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "do_nothing_count": 0,
            "sanitized_count": 0, # 统计自动修正次数
        }
        
        # RAG 初始化
        self.enable_rag = self.config["enable_rag"] and RAG_AVAILABLE
        self.knowledge_base = None
        if self.enable_rag and self.llm_client:
            try:
                # 复用 ADA 的 Embedding 和 Storage 逻辑
                embedding = OpenAIEmbedding()
                # 默认指向 ADA 的存储路径，或者传入特定路径
                default_storage = Path(__file__).parent.parent / "ADA" / "knowledgebase" / "storage"
                storage_dir = knowledge_path if knowledge_path else str(default_storage)
                
                self.knowledge_base = KnowledgeService(
                    embedding_model=embedding,
                    storage_dir=storage_dir,
                    top_k=self.config["rag_top_k"]
                )
                logger.info(f"ReAct RAG 模块已启用，加载路径: {storage_dir}")
            except Exception as e:
                logger.warning(f"RAG 初始化失败: {e}")
                self.enable_rag = False
                self.knowledge_base = None
        
        logger.info(f"ReActAgent '{name}' 初始化完成 (v2.0 Fix, RAG: {'Enabled' if self.enable_rag else 'Disabled'})")
    
    def reset(self, observation: BaseObservation):
        self.current_step = 0
        self.react_history = []
        if self.env_info is None:
            self.env_info = self._extract_env_info(observation)
            self.prompt_manager.set_env_info(self.env_info)
        self.stats = {k: 0 for k in self.stats}
        logger.info(f"ReActAgent 已重置")
    
    def _build_rag_query(self, observation: BaseObservation) -> str:
        """
        构建检索查询语句（语义增强版本）
        
        改进点：
        1. 增加过载严重程度描述（High/Medium/Low）
        2. 增加过载线路数量描述
        3. 结合语义描述和具体 Line ID
        
        Args:
            observation: Grid2Op Observation 对象
            
        Returns:
            查询字符串
        """
        rho = observation.rho
        overloaded_lines = np.where(rho >= 1.0)[0]
        max_rho = np.max(rho) if len(rho) > 0 else 0
        
        parts = []
        
        # 1. 过载严重程度描述（语义特征）
        if len(overloaded_lines) > 0:
            if max_rho >= 1.2:
                severity = "Severe overload (>120%)"
            elif max_rho >= 1.1:
                severity = "Moderate overload (110-120%)"
            else:
                severity = "Slight overload (100-110%)"
            
            if len(overloaded_lines) == 1:
                parts.append(f"{severity} on single line")
            elif len(overloaded_lines) <= 3:
                parts.append(f"{severity} on {len(overloaded_lines)} lines")
            else:
                parts.append(f"{severity} on multiple lines ({len(overloaded_lines)} lines)")
            
            # 保留具体的 Line ID（Grid2Op 中 ID 是固定的，这是强特征）
            parts.append(f"Overloaded lines: {overloaded_lines.tolist()}")
        else:
            # 无过载但接近危险阈值
            if max_rho >= 0.95:
                parts.append(f"Near-overload condition (max rho: {max_rho:.2%})")
            else:
                parts.append(f"Safe condition (max rho: {max_rho:.2%})")
        
        # 2. 负载率信息
        parts.append(f"Max load rate: {max_rho:.2%}")
        
        # 3. 功率平衡信息
        total_gen = np.sum(observation.prod_p)
        total_load = np.sum(observation.load_p)
        parts.append(f"Power balance: {total_gen:.1f} MW / {total_load:.1f} MW")
        
        return "; ".join(parts)
    
    def _pre_validate_rag_context(self, rag_context: str, observation: BaseObservation) -> str:
        """
        RAG 预验证：在发给 LLM 之前，验证历史动作在当前环境下的安全性
        
        Args:
            rag_context: RAG 检索到的原始上下文
            observation: 当前观测
            
        Returns:
            增强后的 RAG 上下文（包含预验证结果）
        """
        try:
            # 尝试从 RAG 上下文中提取历史动作
            history_action_text = self.parser.extract_action_from_text(rag_context)
            
            if not history_action_text:
                # 如果无法提取动作，直接返回原上下文
                return rag_context
            
            # 解析历史动作
            try:
                history_action = self.parser.parse(history_action_text, self.action_space)
            except ValueError:
                # 解析失败，返回原上下文
                return rag_context
            
            # 对历史动作进行预仿真
            is_safe, feedback = self._simulate_action(history_action, observation)
            
            # 在 RAG 上下文末尾添加预验证结果
            if is_safe:
                validation_note = f"\n\n[系统提示]: 上述历史经验中的动作经当前仿真验证【有效且安全】({feedback})，建议优先采纳。"
            else:
                validation_note = f"\n\n[系统警告]: 上述历史经验中的动作在当前场景下【不可行】({feedback})，请仅参考思路，需重新规划动作。"
            
            return rag_context + validation_note
            
        except Exception as e:
            logger.warning(f"RAG 预验证失败: {e}")
            return rag_context
    
    def act(self, observation: BaseObservation, reward: float = 0.0, done: bool = False) -> BaseAction:
        self.current_step += 1
        self.stats["total_steps"] += 1
        
        # 0. 启发式策略（提前终止优化）
        max_rho = float(observation.rho.max())
        overflow_count = int((observation.rho > 1.0).sum())
        
        # 提前终止：如果系统非常安全，直接返回 do_nothing（跳过 LLM 调用）
        if overflow_count == 0 and max_rho <= self.rho_safe:
            self.stats["do_nothing_count"] += 1
            return self.action_space({})
        
        # 预防性调度：如果接近危险阈值但未过载，仍需要 LLM 介入
        if overflow_count == 0 and max_rho <= self.rho_danger:
            self.stats["do_nothing_count"] += 1
            return self.action_space({})
        
        # 1. RAG 检索 (在 ReAct 循环之前)
        rag_context = ""
        if self.enable_rag and self.knowledge_base:
            try:
                query = self._build_rag_query(observation)
                rag_context = self.knowledge_base.get_context_string(query)
                logger.debug(f"RAG Context retrieved: {len(rag_context)} chars")
                
                # RAG 预验证：提取历史动作并进行预仿真
                if rag_context and rag_context != "暂无相关知识":
                    rag_context = self._pre_validate_rag_context(rag_context, observation)
                    
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")
        
        # 2. 设置 RAG 上下文到 PromptManager（只在第一轮 ReAct 注入）
        is_first_react = len(self.react_history) == 0
        if is_first_react:
            self.prompt_manager.set_rag_context(rag_context)
        else:
            # 后续 ReAct 步骤不再注入 RAG，节省 Token
            self.prompt_manager.set_rag_context("")
        
        # 3. 准备 ReAct
        obs_text = self.formatter.format(observation)
        is_first_call = len(self.react_history) == 0
        history = self.prompt_manager.build(obs_text, self.react_history if not is_first_call else None)
        
        # 2. ReAct 循环
        for react_step in range(self.max_react_steps):
            self.stats["react_loops"] += 1
            
            try:
                # LLM 调用
                prompt = history[-1]["content"] if history[-1]["role"] == "user" else ""
                llm_history = [msg for msg in history if msg["role"] != "system" and msg != history[-1]]
                system_prompt = next((msg["content"] for msg in history if msg["role"] == "system"), None)
                
                llm_response = self.llm_client.chat(prompt=prompt, history=llm_history, system_prompt=system_prompt)
                
                # 提取和解析
                action_text = self.parser.extract_action_from_response(llm_response)
                if not action_text:
                    self.stats["failed_actions"] += 1
                    return self.action_space({})
                
                try:
                    raw_action = self.parser.parse(action_text, self.action_space)
                    
                    # === 新增：动作自动修正 (Auto-Correction) ===
                    # 自动处理爬坡率限制，避免因为数值超限导致动作非法
                    sanitized_action, correction_msg = self._sanitize_action(raw_action, observation)
                    if correction_msg:
                        logger.info(f"Step {self.current_step}: 动作已自动修正 -> {correction_msg}")
                        self.stats["sanitized_count"] += 1
                    
                    # 模拟验证
                    is_safe, reason = self._simulate_action(sanitized_action, observation)
                    
                    if is_safe:
                        logger.info(f"Step {self.current_step}: 动作安全 (Step {react_step+1})")
                        self.stats["successful_actions"] += 1
                        self.react_history = [] 
                        return sanitized_action
                    else:
                        # 反馈错误
                        feedback_msg = f"模拟警告: {reason}"
                        if correction_msg:
                            feedback_msg += f" (注: 原始动作已被修正: {correction_msg})"
                            
                        self.react_history.append({"role": "assistant", "content": llm_response})
                        self.react_history = self.prompt_manager.add_observation_feedback(self.react_history, feedback_msg)
                        
                        # 如果 ReAct 步数过多，压缩历史以节省 Token
                        if react_step >= 2:
                            self.react_history = self.prompt_manager.compress_history(self.react_history, max_preserved=2)
                        
                        history = self.prompt_manager.build(obs_text, self.react_history)
                        continue
                        
                except ValueError as e:
                    self.react_history.append({"role": "assistant", "content": llm_response})
                    self.react_history = self.prompt_manager.add_observation_feedback(self.react_history, f"格式错误: {e}")
                    
                    # 如果 ReAct 步数过多，压缩历史
                    if react_step >= 2:
                        self.react_history = self.prompt_manager.compress_history(self.react_history, max_preserved=2)
                    
                    history = self.prompt_manager.build(obs_text, self.react_history)
                    continue
                    
            except Exception as e:
                logger.error(f"ReAct 异常: {e}")
                import traceback
                traceback.print_exc()
                return self.action_space({})
        
        logger.warning(f"Step {self.current_step}: ReAct 耗尽步数，返回 do_nothing")
        self.stats["failed_actions"] += 1
        self.react_history = []
        return self.action_space({})

    def _sanitize_action(self, action: BaseAction, observation: BaseObservation) -> tuple[BaseAction, str]:
        """
        自动修正动作（增强版：包含动作剪枝）
        
        改进点：
        1. 检查 Redispatch 是否超过爬坡率，如果超过则截断
        2. 剔除对不可调度发电机的操作
        3. 过滤微小调整（动作剪枝）：过滤掉对奖励无益的微小 redispatch（< 0.5 MW）
        4. 处理 None 值和 NaN 值
        """
        correction_details = []
        action_dict = action.as_dict()
        modified = False
        
        # 动作剪枝阈值：小于此值的 redispatch 将被过滤
        MIN_REDISPATCH_THRESHOLD = self.min_redispatch_threshold
        
        # 1. 修正 Redispatch
        if 'redispatch' in action_dict and action_dict['redispatch'] is not None:
            new_redispatch = []
            redispatch_data = action_dict['redispatch']
            
            # 转换为 list of (id, amount)
            items = []
            if isinstance(redispatch_data, dict):
                items = [(int(k), float(v)) for k, v in redispatch_data.items() if v is not None]
            elif hasattr(redispatch_data, '__iter__') and not isinstance(redispatch_data, str):
                try:
                    # 处理 numpy array 或 list
                    for i, val in enumerate(redispatch_data):
                        # 过滤 None 和 NaN
                        if val is not None and not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                            if abs(float(val)) > 1e-6:
                                items.append((i, float(val)))
                except:
                    pass

            for gen_id, amount in items:
                # 动作剪枝：过滤微小调整
                if abs(amount) < MIN_REDISPATCH_THRESHOLD:
                    correction_details.append(f"忽略发电机 {gen_id} 的微小调整 ({amount:.2f} MW < {MIN_REDISPATCH_THRESHOLD} MW 阈值)")
                    modified = True
                    continue
                # 检查是否可调度
                if hasattr(observation, 'gen_redispatchable') and not observation.gen_redispatchable[gen_id]:
                    correction_details.append(f"忽略发电机 {gen_id} (不可调度)")
                    modified = True
                    continue
                
                # 检查爬坡率 (Ramp Rate)
                clamped_amount = amount
                if hasattr(observation, 'gen_max_ramp_up') and hasattr(observation, 'gen_max_ramp_down'):
                    max_up = float(observation.gen_max_ramp_up[gen_id])
                    max_down = float(observation.gen_max_ramp_down[gen_id])
                    
                    if amount > max_up:
                        clamped_amount = max_up
                        correction_details.append(f"Gen {gen_id} +{amount:.1f}->+{max_up:.1f} (爬坡限制)")
                        modified = True
                    elif amount < -max_down:
                        clamped_amount = -max_down
                        correction_details.append(f"Gen {gen_id} {amount:.1f}->-{max_down:.1f} (爬坡限制)")
                        modified = True
                
                new_redispatch.append((gen_id, clamped_amount))
            
            if modified:
                # 重新创建动作
                # 注意：我们不能直接修改 action 对象，最好创建一个新的
                new_action = self.action_space({})
                new_action.redispatch = new_redispatch
                # 复制其他属性 (如 set_line_status)
                if 'set_line_status' in action_dict and action_dict['set_line_status'] is not None:
                     # 这是一个简化处理，通常 parser 一次只生成一种类型的动作
                     # 如果混合了动作，这里需要更复杂的复制逻辑。
                     # 简单起见，假设 ActionParser 主要生成 redispatch 或 topology
                     pass 
                
                # 如果原动作还有 set_line_status，也得保留
                # Grid2Op 的 action 更新比较繁琐，这里采用 "叠加" 方式
                if 'set_line_status' in action_dict:
                     # 这是一个 tricky 的地方，直接修改 action 可能不生效或报错
                     # 最稳妥的是：如果仅仅修改了 redispatch，我们就在原 action 上覆盖
                     action.redispatch = new_redispatch
                     
                return action, "; ".join(correction_details)

        return action, ""

    def _simulate_action(self, action: BaseAction, observation: BaseObservation) -> tuple[bool, str]:
        """
        模拟验证（增强版：返回详细的量化反馈）
        
        改进点：
        1. 返回详细的负载率变化信息
        2. 提供方向性反馈（方向正确但力度不够 vs 方向错误）
        3. 识别具体过载线路的变化
        
        Returns:
            (is_safe: bool, feedback: str)
        """
        try:
            sim_obs, sim_reward, sim_done, sim_info = observation.simulate(action, time_step=0)
            
            # === BUG FIX START ===
            # 检查异常：Grid2Op 在成功时返回 'exception': [] (空列表)
            exception = sim_info.get('exception', None)
            if exception is not None:
                if isinstance(exception, list):
                    if len(exception) > 0:
                        err_strs = [str(e) for e in exception]
                        return False, f"动作不合法: {'; '.join(err_strs)}"
                else:
                    return False, f"动作不合法: {str(exception)}"
            # === BUG FIX END ===

            # 潮流发散检查
            if np.any(np.isnan(sim_obs.rho)) or np.any(np.isinf(sim_obs.rho)):
                return False, "模拟失败：潮流发散 (NaN/Inf)"
            
            # 量化分析
            rho_before = observation.rho
            rho_after = sim_obs.rho
            max_rho_before = float(rho_before.max())
            max_rho_after = float(rho_after.max())
            delta_rho = max_rho_after - max_rho_before
            
            overflow_before = np.where(rho_before > 1.0)[0]
            overflow_after = np.where(rho_after > 1.0)[0]
            overflow_before_count = len(overflow_before)
            overflow_after_count = len(overflow_after)
            
            # 1. 安全检查（熔断机制）
            if sim_done:
                return False, "动作导致游戏结束 (Game Over)"
            if max_rho_after > 1.5:
                return False, f"动作导致极度过载 ({max_rho_after:.2%})"
            
            # 2. 成功情况：完全消除过载
            if overflow_after_count == 0 and overflow_before_count > 0:
                return True, f"成功：所有过载已消除。最大负载率从 {max_rho_before:.2%} 降至 {max_rho_after:.2%}"
            
            if overflow_after_count == 0 and overflow_before_count == 0:
                return True, f"验证通过：系统保持安全状态（最大负载率 {max_rho_after:.2%}）"
            
            # 3. 缓解策略（量化反馈）
            if overflow_before_count > 0:
                # 原本有过载
                if overflow_after_count < overflow_before_count:
                    # 过载线路减少
                    feedback = f"有效缓解：过载线路数从 {overflow_before_count} 降至 {overflow_after_count}"
                    if delta_rho < -0.01:  # 最大负载率也下降了
                        feedback += f"，最大负载率从 {max_rho_before:.2%} 降至 {max_rho_after:.2%}（改善 {abs(delta_rho):.2%}）"
                    return True, feedback
                
                if delta_rho < -0.02:  # 最大负载率明显下降（>2%）
                    feedback = f"方向正确：最大负载率从 {max_rho_before:.2%} 降至 {max_rho_after:.2%}（改善 {abs(delta_rho):.2%}）"
                    if overflow_after_count == overflow_before_count:
                        feedback += f"，但仍有 {overflow_after_count} 条线路过载。建议加大调整力度或寻找协同发电机。"
                    return True, feedback
                
                if delta_rho > 0.02:  # 最大负载率恶化（>2%）
                    return False, f"方向错误：最大负载率从 {max_rho_before:.2%} 恶化至 {max_rho_after:.2%}（恶化 {delta_rho:.2%}）。建议撤销并尝试相反操作。"
                
                if abs(delta_rho) <= 0.02:  # 变化很小
                    return False, f"无效动作：负载率几乎无变化（{max_rho_before:.2%} -> {max_rho_after:.2%}），仍有 {overflow_after_count} 条线路过载。"
                
                # 默认情况：负载率未下降
                return False, f"无效动作：负载率未下降（{max_rho_before:.2%} -> {max_rho_after:.2%}），仍有 {overflow_after_count} 条线路过载。"
            
            else:
                # 原本安全，检查是否引入新过载
                if overflow_after_count > 0:
                    # 找出新过载的线路
                    new_overload_lines = overflow_after.tolist()
                    feedback = f"动作导致新过载：{overflow_after_count} 条线路过载（线路 {new_overload_lines[:5]}{'...' if len(new_overload_lines) > 5 else ''}）"
                    feedback += f"，最大负载率从 {max_rho_before:.2%} 升至 {max_rho_after:.2%}"
                    return False, feedback
                
                if max_rho_after > max_rho_before + 0.10:  # 负载率大幅上升（>10%）
                    return False, f"动作导致负载率大幅上升：从 {max_rho_before:.2%} 升至 {max_rho_after:.2%}（上升 {delta_rho:.2%}）"
                
                # 原本安全，动作后仍安全
                return True, f"验证通过：系统保持安全（最大负载率 {max_rho_after:.2%}）"

        except Exception as e:
            return False, f"模拟过程出错: {str(e)}"

    def _extract_env_info(self, observation):
        # 保持原样，省略以节省空间
        env_info = {
            "n_gen": int(observation.n_gen),
            "n_line": int(observation.n_line),
            "n_sub": int(observation.n_sub),
        }
        # 简单提取发电机 Ramp 信息
        gen_info = []
        if hasattr(observation, 'gen_max_ramp_up'):
            for i in range(observation.n_gen):
                gen_info.append({
                    "gen_id": i,
                    "max_ramp_up": float(observation.gen_max_ramp_up[i]),
                    "max_ramp_down": float(observation.gen_max_ramp_down[i]),
                    "redispatchable": bool(observation.gen_redispatchable[i]) if hasattr(observation, 'gen_redispatchable') else True
                })
        env_info["generators"] = gen_info
        return env_info