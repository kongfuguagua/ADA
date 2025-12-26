# -*- coding: utf-8 -*-
"""
ADA 日志工具
统一的日志格式，方便 Summarizer 后期解析日志进行复盘
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SystemConfig

# 导入 JSON 工具
from .json_utils import convert_to_serializable, safe_json_dumps


class ADAFormatter(logging.Formatter):
    """自定义日志格式化器，支持结构化输出"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m'       # 重置
    }
    
    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color
    
    def format(self, record: logging.LogRecord) -> str:
        # 基础信息
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname
        
        # 颜色处理
        if self.use_color and sys.stdout.isatty():
            color = self.COLORS.get(level, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            level_str = f"{color}{level:8s}{reset}"
        else:
            level_str = f"{level:8s}"
        
        # 模块信息
        module = record.name
        
        # 消息
        message = record.getMessage()
        
        # 额外数据（如果有）
        extra_str = ""
        if hasattr(record, 'extra_data') and record.extra_data:
            try:
                # 使用安全的 JSON 序列化
                extra_str = f" | data={safe_json_dumps(record.extra_data, ensure_ascii=False)}"
            except (TypeError, ValueError) as e:
                # 如果序列化失败，使用字符串表示
                extra_str = f" | data=<序列化失败: {str(e)}>"
        
        return f"[{timestamp}] {level_str} [{module}] {message}{extra_str}"


class ADALogger:
    """ADA 统一日志器"""
    
    def __init__(self, name: str, log_to_file: bool = True):
        """
        初始化日志器
        
        Args:
            name: 日志器名称（通常是模块名）
            log_to_file: 是否同时写入文件
        """
        self.name = name
        self.logger = logging.getLogger(f"ADA.{name}")
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加 handler
        if not self.logger.handlers:
            # 控制台输出
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(ADAFormatter(use_color=True))
            self.logger.addHandler(console_handler)
            
            # 文件输出
            if log_to_file:
                config = SystemConfig()
                log_path = config.get_log_path()
                log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
                
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(ADAFormatter(use_color=False))
                self.logger.addHandler(file_handler)
    
    def _log(self, level: int, message: str, extra_data: Dict[str, Any] = None):
        """内部日志方法"""
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "",
            0,
            message,
            (),
            None
        )
        if extra_data:
            # 转换 numpy 类型为可序列化类型
            record.extra_data = convert_to_serializable(extra_data)
        else:
            record.extra_data = None
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self._log(logging.DEBUG, message, kwargs if kwargs else None)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self._log(logging.INFO, message, kwargs if kwargs else None)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self._log(logging.WARNING, message, kwargs if kwargs else None)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self._log(logging.ERROR, message, kwargs if kwargs else None)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self._log(logging.CRITICAL, message, kwargs if kwargs else None)
    
    def agent_action(self, agent: str, action: str, details: Dict[str, Any] = None):
        """
        记录智能体动作（结构化日志，便于复盘）
        
        Args:
            agent: 智能体名称
            action: 动作名称
            details: 动作详情
        """
        self.info(
            f"[{agent}] {action}",
            agent=agent,
            action=action,
            **(details or {})
        )
    
    def trace_start(self, trace_id: str, env_state: Dict[str, Any]):
        """记录轨迹开始"""
        self.info(f"轨迹开始: {trace_id}", trace_id=trace_id, env=env_state)
    
    def trace_end(self, trace_id: str, success: bool, score: float):
        """记录轨迹结束"""
        self.info(
            f"轨迹结束: {trace_id} | 成功={success} | 评分={score:.4f}",
            trace_id=trace_id,
            success=success,
            score=score
        )


# 日志器缓存
_loggers: Dict[str, ADALogger] = {}


def get_logger(name: str) -> ADALogger:
    """
    获取日志器（单例模式）
    
    Args:
        name: 日志器名称
    
    Returns:
        ADALogger 实例
    """
    if name not in _loggers:
        _loggers[name] = ADALogger(name)
    return _loggers[name]


def log_function_call(logger_name: str = None):
    """
    函数调用日志装饰器
    
    Args:
        logger_name: 日志器名称，默认使用函数所在模块名
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = logger_name or func.__module__.split('.')[-1]
            logger = get_logger(name)
            
            # 记录调用
            logger.debug(f"调用 {func.__name__}", args=str(args)[:100], kwargs=str(kwargs)[:100])
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"完成 {func.__name__}", result_type=type(result).__name__)
                return result
            except Exception as e:
                logger.error(f"异常 {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    return decorator


# ============= 测试代码 =============
if __name__ == "__main__":
    # 测试日志器
    logger = get_logger("test")
    
    print("测试日志输出:")
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    
    print("\n测试结构化日志:")
    logger.agent_action("Planner", "状态增广", {"tool": "power_flow", "step": 1})
    logger.trace_start("trace_001", {"load": 100.0})
    logger.trace_end("trace_001", success=True, score=0.85)
    
    print("\n测试装饰器:")
    
    @log_function_call("test")
    def sample_function(x, y):
        return x + y
    
    result = sample_function(1, 2)
    print(f"函数结果: {result}")

