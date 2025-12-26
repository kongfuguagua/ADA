# -*- coding: utf-8 -*-
"""
ADA 系统完整测试
注意：此测试需要配置有效的 LLM API Key
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def check_api_config():
    """检查 API 配置"""
    api_key = os.getenv("CLOUD_API_KEY")
    if not api_key:
        print("=" * 60)
        print("错误: 未配置 LLM API Key")
        print("=" * 60)
        print()
        print("请在 .env 文件中设置以下环境变量:")
        print("  CLOUD_API_KEY=your-api-key-here")
        print("  CLOUD_BASE_URL=https://api.deepseek.com")
        print("  CLOUD_MODEL=deepseek-chat")
        print()
        return False
    return True


def test_utils_module():
    """测试 utils 模块"""
    print("\n" + "=" * 50)
    print("  测试1: Utils 模块")
    print("=" * 50)
    
    try:
        from utils import (
            VariableType, KnowledgeType, FeedbackType,
            VariableDefinition, OptimizationProblem, Solution, Feedback,
            BaseLLM, BaseEmbeddings, BaseTool,
            get_logger
        )
        
        # 测试数据结构
        var = VariableDefinition(name="x", lower_bound=0, upper_bound=10)
        print(f"✓ VariableDefinition: {var}")
        
        problem = OptimizationProblem(
            objective_function_latex=r"\min x^2",
            variables=[var]
        )
        print(f"✓ OptimizationProblem: {len(problem.variables)} 个变量")
        
        solution = Solution(
            is_feasible=True,
            algorithm_used="Test",
            decision_variables={"x": 5.0},
            objective_value=25.0
        )
        print(f"✓ Solution: 目标值={solution.objective_value}")
        
        logger = get_logger("Test")
        logger.info("日志系统正常")
        print("✓ Logger 正常")
        
        print("✓ Utils 模块测试通过")
        return True
    except Exception as e:
        print(f"✗ Utils 模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_module():
    """测试 LLM 模块"""
    print("\n" + "=" * 50)
    print("  测试2: LLM 模块")
    print("=" * 50)
    
    try:
        from utils.llm import OpenAIChat
        
        llm = OpenAIChat()
        print(f"✓ OpenAIChat 初始化成功: {llm.model}")
        
        # 测试简单对话
        response = llm.chat("用一句话介绍什么是优化问题")
        print(f"✓ LLM 响应: {response[:50]}...")
        
        print("✓ LLM 模块测试通过")
        return True
    except ValueError as e:
        print(f"✗ API 配置错误: {e}")
        return False
    except Exception as e:
        print(f"✗ LLM 模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_module():
    """测试 Embedding 模块"""
    print("\n" + "=" * 50)
    print("  测试3: Embedding 模块")
    print("=" * 50)
    
    try:
        from utils.embeddings import OpenAIEmbedding
        
        embedding = OpenAIEmbedding()
        print(f"✓ OpenAIEmbedding 初始化成功: {embedding.model}")
        
        # 测试向量生成
        v1 = embedding.get_embedding("电网调度优化")
        v2 = embedding.get_embedding("power grid dispatch optimization")
        
        sim = OpenAIEmbedding.cosine_similarity(v1, v2)
        print(f"✓ 向量维度: {len(v1)}")
        print(f"✓ 相似度: {sim:.4f}")
        
        print("✓ Embedding 模块测试通过")
        return True
    except ValueError as e:
        print(f"✗ API 配置错误: {e}")
        return False
    except Exception as e:
        print(f"✗ Embedding 模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_solver_module():
    """测试 Solver 模块"""
    print("\n" + "=" * 50)
    print("  测试4: Solver 模块")
    print("=" * 50)
    
    try:
        from Solver import SolverAgent
        from utils.const import OptimizationProblem, VariableDefinition
        
        solver = SolverAgent()
        print(f"✓ SolverAgent 初始化成功")
        print(f"  可用算法: {solver.list_algorithms()}")
        
        # 测试凸问题
        problem = OptimizationProblem(
            objective_function_latex=r"\min x_1^2 + x_2^2",
            objective_function_code="x1**2 + x2**2",
            variables=[
                VariableDefinition(name="x1", lower_bound=-10, upper_bound=10),
                VariableDefinition(name="x2", lower_bound=-10, upper_bound=10),
            ]
        )
        
        solution = solver.solve(problem)
        print(f"✓ 求解完成: 算法={solver.get_selected_algorithm()}")
        print(f"  可行: {solution.is_feasible}")
        print(f"  目标值: {solution.objective_value:.6f}")
        print(f"  解: {solution.decision_variables}")
        
        print("✓ Solver 模块测试通过")
        return True
    except Exception as e:
        print(f"✗ Solver 模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_planner_tools():
    """测试 Planner 工具"""
    print("\n" + "=" * 50)
    print("  测试5: Planner 分析工具")
    print("=" * 50)
    
    try:
        from Planner.tools.registry import create_default_registry
        
        registry = create_default_registry()
        print(f"✓ 工具注册表创建成功")
        print(f"  工具列表: {registry.list_tools()}")
        
        # 测试各工具
        result = registry.execute("grid_status_analysis")
        print(f"✓ grid_status_analysis: {result.get('summary', result)}")
        
        result = registry.execute("overflow_risk_analysis", threshold=0.9)
        print(f"✓ overflow_risk_analysis: {result.get('total_high_risk', 0)} 条高风险线路")
        
        result = registry.execute("generator_capacity_analysis")
        print(f"✓ generator_capacity_analysis: {result.get('summary', result)}")
        
        result = registry.execute("load_trend_analysis", forecast_hours=6)
        print(f"✓ load_trend_analysis: 趋势={result.get('trend', 'N/A')}")
        
        print("✓ Planner 工具测试通过")
        return True
    except Exception as e:
        print(f"✗ Planner 工具测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_service():
    """测试知识库服务"""
    print("\n" + "=" * 50)
    print("  测试6: 知识库服务")
    print("=" * 50)
    
    try:
        from knowledgebase.service import KnowledgeService
        from utils.embeddings import OpenAIEmbedding
        from utils.const import KnowledgeType
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            embedding = OpenAIEmbedding()
        kb = KnowledgeService(
                embedding_model=embedding,
                storage_path=temp_dir
        )
            print(f"✓ KnowledgeService 初始化成功: {kb}")
            
            # 添加知识
            tk_id = kb.add_knowledge(
                "电网调度优化问题通常建模为最小化发电成本",
                KnowledgeType.TK
            )
            print(f"✓ 添加 TK: {tk_id[:8]}...")
            
            ak_id = kb.add_knowledge(
                "当负载不确定时，建议先调用负荷预测工具",
                KnowledgeType.AK
            )
            print(f"✓ 添加 AK: {ak_id[:8]}...")
            
            # 检索知识
            tk_results = kb.query_task_knowledge("电网优化")
            print(f"✓ TK 检索: {len(tk_results)} 条结果")
            
            ak_results = kb.query_action_knowledge("负载预测")
            print(f"✓ AK 检索: {len(ak_results)} 条结果")
            
            print("✓ 知识库服务测试通过")
            return True
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"✗ 知识库服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    
def test_full_pipeline():
        """测试完整流程"""
    print("\n" + "=" * 50)
    print("  测试7: 完整流程")
    print("=" * 50)
    
    try:
        from main import ADAOrchestrator
        from utils.const import EnvironmentState
        
        # 创建临时知识库目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            orchestrator = ADAOrchestrator(kb_storage_path=temp_dir)
            print("✓ ADAOrchestrator 初始化成功")
            
            # 获取系统状态
            status = orchestrator.get_status()
            print(f"  知识库: {status['knowledge_count']} 条")
            print(f"  算法: {status['algorithms']}")
            print(f"  工具: {status['tools']}")
            
            # 运行简单任务
            state = EnvironmentState(
                user_instruction="优化发电调度，最小化成本",
                real_data={"load": 100.0}
        )
        
            result = orchestrator.run(state, max_retries=2)
        
            print(f"\n运行结果:")
            print(f"  成功: {result['success']}")
            print(f"  尝试次数: {result['attempts']}")
            
            if result['success'] and result['solution']:
                print(f"  算法: {result['solution'].algorithm_used}")
                print(f"  目标值: {result['solution'].objective_value:.4f}")
            
            if result['feedback']:
                print(f"  评估: {result['feedback'].feedback_type.value}")
                print(f"  评分: {result['feedback'].score:.4f}")
            
            print("✓ 完整流程测试通过")
            return True
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"✗ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("  ADA 系统测试")
    print("=" * 60)
    
    # 检查 API 配置
    if not check_api_config():
        return
    
    results = {}
    
    # 运行测试
    results["utils"] = test_utils_module()
    results["llm"] = test_llm_module()
    results["embedding"] = test_embedding_module()
    results["solver"] = test_solver_module()
    results["planner_tools"] = test_planner_tools()
    results["knowledge"] = test_knowledge_service()
    results["pipeline"] = test_full_pipeline()
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    print()
    print(f"总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n✓ 所有测试通过！")
    else:
        print(f"\n✗ {total - passed} 个测试失败")


if __name__ == "__main__":
    main()
