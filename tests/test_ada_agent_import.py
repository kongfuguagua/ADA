#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基本导入与初始化测试

运行方式（在项目根目录，且已 conda activate ada）：

    python -m pytest -q

测试内容：
1. 能否成功导入 ADA.ADA_Agent 及其依赖模块
2. 能否在最小假环境下初始化 ADA_Agent（不真正调用 Grid2Op 环境）
"""

import types


def test_import_and_basic_init():
    # 延迟导入，避免在无 grid2op 环境下报错时影响其他测试
    try:
        from ADA.agent import ADA_Agent  # noqa: F401
    except Exception as exc:
        raise AssertionError(f"导入 ADA.agent.ADA_Agent 失败: {exc}")

    # 构造一个最小的伪 action_space / observation_space，用于 smoke test
    class DummyActionSpace:
        def __call__(self, *args, **kwargs):
            # 返回一个简单的对象，具有 as_dict 方法
            class _A:
                def as_dict(self_inner):
                    return {}

            return _A()

    class DummyObsSpace:
        # 仅提供 Solver / Planner 初始化中用到的最小属性
        n_line = 1
        n_sub = 1
        n_load = 0
        n_gen = 0
        n_storage = 0
        line_or_to_subid = [0]
        line_ex_to_subid = [0]
        load_to_subid = []
        gen_to_subid = []
        storage_to_subid = []

        def get_thermal_limit(self):
            return [100.0]

    action_space = DummyActionSpace()
    obs_space = DummyObsSpace()

    # 仅验证 __init__ 不抛异常（允许内部模块根据缺失依赖自行降级）
    from ADA.agent import ADA_Agent

    agent = ADA_Agent(
        action_space=action_space,
        observation_space=obs_space,
        llm_client=None,  # 关闭 LLM，避免外部依赖
        enable_knowledge_base=False,  # 测试时关闭 RAG，避免 API 依赖
    )

    assert agent is not None


