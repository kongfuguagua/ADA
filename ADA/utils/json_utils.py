# -*- coding: utf-8 -*-
"""
JSON 序列化工具
提供安全的 JSON 序列化，自动处理 numpy 类型等不可序列化的对象
"""

import json
from typing import Any

# 尝试导入 numpy（可选）
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def convert_to_serializable(obj: Any) -> Any:
    """
    递归转换对象为 JSON 可序列化的类型
    
    处理 numpy 类型、numpy 数组、集合等
    
    Args:
        obj: 要转换的对象
        
    Returns:
        可序列化的对象
    """
    if HAS_NUMPY and np is not None:
        # 处理 numpy 标量类型
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # 转换为 Python 原生类型
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # 转换为列表
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.complexfloating):
            return complex(obj)
    
    # 处理字典
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    
    # 处理列表和元组
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    
    # 处理集合
    if isinstance(obj, set):
        return list(obj)
    
    # 处理其他不可序列化类型（如 datetime）
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # 如果无法序列化，转换为字符串
        return str(obj)


def safe_json_dumps(
    obj: Any,
    ensure_ascii: bool = False,
    indent: int = None,
    default: Any = None,
    **kwargs
) -> str:
    """
    安全的 JSON 序列化函数
    
    自动处理 numpy 类型等不可序列化的对象
    
    Args:
        obj: 要序列化的对象
        ensure_ascii: 是否确保 ASCII 编码
        indent: 缩进级别
        default: 自定义默认序列化函数（可选）
        **kwargs: 传递给 json.dumps 的其他参数
        
    Returns:
        JSON 字符串
        
    Examples:
        >>> import numpy as np
        >>> data = {"value": np.int32(42), "array": np.array([1, 2, 3])}
        >>> safe_json_dumps(data)
        '{"value": 42, "array": [1, 2, 3]}'
    """
    # 先转换为可序列化类型
    serializable_obj = convert_to_serializable(obj)
    
    # 如果提供了自定义 default 函数，使用它
    if default is not None:
        return json.dumps(serializable_obj, ensure_ascii=ensure_ascii, indent=indent, default=default, **kwargs)
    else:
        return json.dumps(serializable_obj, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def safe_json_dump(
    obj: Any,
    fp: Any,
    ensure_ascii: bool = False,
    indent: int = None,
    default: Any = None,
    **kwargs
) -> None:
    """
    安全的 JSON 序列化并写入文件
    
    Args:
        obj: 要序列化的对象
        fp: 文件对象或文件路径
        ensure_ascii: 是否确保 ASCII 编码
        indent: 缩进级别
        default: 自定义默认序列化函数（可选）
        **kwargs: 传递给 json.dump 的其他参数
    """
    serializable_obj = convert_to_serializable(obj)
    
    if default is not None:
        json.dump(serializable_obj, fp, ensure_ascii=ensure_ascii, indent=indent, default=default, **kwargs)
    else:
        json.dump(serializable_obj, fp, ensure_ascii=ensure_ascii, indent=indent, **kwargs)

