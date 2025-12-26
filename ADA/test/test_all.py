# -*- coding: utf-8 -*-
"""
ADA 系统综合测试
测试所有模块的集成功能
"""

import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest


class TestConfig(unittest.TestCase):
    """配置模块测试"""
    
    def test_llm_config(self):
        """测试 LLM 配置"""
        from config import get_llm_config
        
        config = get_llm_config()
        self.assertIsNotNone(config.model_name)
        self.assertIsNotNone(config.base_url)
        self.assertGreater(config.temperature, 0)
    
    def test_system_config(self):
        """测试系统配置"""
        from config import get_system_config
        
        config = get_system_config()
        self.assertGreater(config.max_retries, 0)
        self.assertGreater(config.judger_alpha, 0)
        self.assertLess(config.judger_alpha, 1)


class TestUtils(unittest.TestCase):
    """工具模块测试"""
    
    def test_environment_state(self):
        """测试环境状态"""
        from utils.const import EnvironmentState
        
        state = EnvironmentState(
            user_instruction="测试指令",
            real_data={"load": 100.0}
        )
        
        self.assertEqual(state.user_instruction, "测试指令")
        self.assertIn("load", state.real_data)
        
        prompt_str = state.to_prompt_string()
        self.assertIn("测试指令", prompt_str)
    
    def test_optimization_problem(self):
        """测试优化问题定义"""
        from utils.const import OptimizationProblem, VariableDefinition
        
        problem = OptimizationProblem(
            objective_function_latex=r"\min x^2",
            variables=[
                VariableDefinition(name="x", lower_bound=0, upper_bound=10)
            ]
        )
        
        self.assertEqual(len(problem.variables), 1)
        self.assertEqual(problem.get_variable_names(), ["x"])
    
    def test_feedback(self):
        """测试反馈"""
        from utils.const import Feedback, FeedbackType
        
        feedback = Feedback(
            feedback_type=FeedbackType.PASSED,
            score=0.8
        )
        
        self.assertFalse(feedback.needs_retry())
        
        feedback2 = Feedback(
            feedback_type=FeedbackType.PHYSICAL_ERROR,
            score=0.0
        )
        
        self.assertTrue(feedback2.needs_retry())


class TestKnowledgeBase(unittest.TestCase):
    """知识库测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_embeddings(self):
        """测试 Embedding"""
        from knowledgebase.Embeddings import MockEmbedding
        
        embedding = MockEmbedding(dimension=64)
        
        v1 = embedding.get_embedding("你好")
        v2 = embedding.get_embedding("世界")
        v3 = embedding.get_embedding("你好")  # 相同文本
        
        self.assertEqual(len(v1), 64)
        self.assertEqual(v1, v3)  # 相同文本应该有相同的向量
    
    def test_vector_store(self):
        """测试向量存储"""
        from knowledgebase.VectorBase import VectorStore
        from knowledgebase.Embeddings import MockEmbedding
        
        docs = ["文档1", "文档2", "文档3"]
        store = VectorStore(docs)
        embedding = MockEmbedding()
        
        store.get_vector(embedding, show_progress=False)
        
        self.assertEqual(len(store), 3)
        
        results = store.query("文档", embedding, k=2)
        self.assertEqual(len(results), 2)
    
    def test_knowledge_service(self):
        """测试知识服务"""
        from knowledgebase.service import KnowledgeService
        from knowledgebase.Embeddings import MockEmbedding
        from utils.const import KnowledgeType
        
        service = KnowledgeService(
            embedding_model=MockEmbedding(),
            storage_path=self.temp_dir
        )
        
        # 添加知识
        ak_id = service.add_knowledge(
            "测试动作知识",
            KnowledgeType.AK
        )
        
        self.assertIsNotNone(ak_id)
        self.assertEqual(len(service), 1)
        
        # 检索知识
        results = service.query_action_knowledge("测试")
        self.assertGreater(len(results), 0)


class TestPlanner(unittest.TestCase):
    """Planner 测试"""
    
    def test_tool_registry(self):
        """测试工具注册表"""
        from Planner.tools.registry import create_default_registry
        
        registry = create_default_registry()
        
        self.assertIn("weather_forecast", registry)
        self.assertIn("power_flow", registry)
        self.assertIn("load_forecast", registry)
        
        # 测试工具执行
        result = registry.execute("weather_forecast", location="北京")
        self.assertIn("summary", result)
    
    def test_planner_agent(self):
        """测试 Planner Agent"""
        from Planner.core import PlannerAgent
        from knowledgebase.LLM import MockLLM
        from utils.const import EnvironmentState
        
        # 创建 Mock LLM
        mock_llm = MockLLM()
        mock_llm.set_response("状态增广", "FINISH")
        mock_llm.set_response("建立数学优化", '''```json
{
    "objective_function_latex": "\\\\min x",
    "variables": [{"name": "x", "type": "continuous", "lower_bound": 0, "upper_bound": 10}],
    "parameters": {},
    "modeling_rationale": "测试"
}
```''')
        
        planner = PlannerAgent(llm=mock_llm, max_augmentation_steps=1)
        
        state = EnvironmentState(user_instruction="测试")
        problem = planner.plan(state)
        
        self.assertIsNotNone(problem)
        self.assertGreater(len(problem.variables), 0)


class TestSolver(unittest.TestCase):
    """Solver 测试"""
    
    def test_feature_extraction(self):
        """测试特征提取"""
        from Solver.feature import ProblemFeatureExtractor
        from utils.const import OptimizationProblem, VariableDefinition
        
        extractor = ProblemFeatureExtractor()
        
        problem = OptimizationProblem(
            objective_function_latex=r"\min x^2",
            objective_function_code="x**2",
            variables=[
                VariableDefinition(name="x", lower_bound=0, upper_bound=10)
            ]
        )
        
        features = extractor.extract(problem)
        
        self.assertEqual(features.variable_count, 1)
        self.assertFalse(features.is_linear)
    
    def test_convex_optimizer(self):
        """测试凸优化器"""
        from Solver.Template.convex import ConvexOptimizer
        from utils.const import OptimizationProblem, VariableDefinition
        
        problem = OptimizationProblem(
            objective_function_latex=r"\min x_1^2 + x_2^2",
            objective_function_code="x1**2 + x2**2",
            variables=[
                VariableDefinition(name="x1", lower_bound=-10, upper_bound=10),
                VariableDefinition(name="x2", lower_bound=-10, upper_bound=10),
            ]
        )
        
        optimizer = ConvexOptimizer()
        solution = optimizer.solve(problem)
        
        self.assertTrue(solution.is_feasible)
        self.assertLess(solution.objective_value, 1.0)  # 最优值接近 0
    
    def test_solver_agent(self):
        """测试 Solver Agent"""
        from Solver.core import SolverAgent
        from utils.const import OptimizationProblem, VariableDefinition
        
        solver = SolverAgent()
        
        problem = OptimizationProblem(
            objective_function_latex=r"\min x_1 + x_2",
            objective_function_code="x1 + x2",
            variables=[
                VariableDefinition(name="x1", lower_bound=0, upper_bound=10),
                VariableDefinition(name="x2", lower_bound=0, upper_bound=10),
            ]
        )
        
        solution = solver.solve(problem)
        
        self.assertIsNotNone(solution)
        self.assertTrue(solution.is_feasible)


class TestJudger(unittest.TestCase):
    """Judger 测试"""
    
    def test_physical_reward(self):
        """测试物理评分"""
        from Judger.Reward.phy_reward import PhysicalReward
        from utils.const import OptimizationProblem, Solution, VariableDefinition
        
        reward = PhysicalReward()
        
        problem = OptimizationProblem(
            objective_function_latex=r"\min x",
            variables=[
                VariableDefinition(name="x", lower_bound=0, upper_bound=10)
            ]
        )
        
        solution = Solution(
            is_feasible=True,
            decision_variables={"x": 5.0},
            objective_value=5.0
        )
        
        score, details = reward(problem, solution)
        
        self.assertGreater(score, 0)
        self.assertTrue(details.get("is_safe", False))
    
    def test_judger_agent(self):
        """测试 Judger Agent"""
        from Judger.core import JudgerAgent
        from utils.const import OptimizationProblem, Solution, VariableDefinition, FeedbackType
        
        judger = JudgerAgent()
        
        problem = OptimizationProblem(
            objective_function_latex=r"\min x",
            variables=[
                VariableDefinition(name="x", lower_bound=0, upper_bound=10)
            ]
        )
        
        solution = Solution(
            is_feasible=True,
            algorithm_used="TestAlgo",
            decision_variables={"x": 5.0},
            objective_value=5.0
        )
        
        feedback = judger.evaluate(problem, solution)
        
        self.assertIsNotNone(feedback)
        self.assertGreater(feedback.score, 0)


class TestSummarizer(unittest.TestCase):
    """Summarizer 测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_summarizer_agent(self):
        """测试 Summarizer Agent"""
        from Summarizer.core import SummarizerAgent
        from knowledgebase.service import KnowledgeService
        from knowledgebase.Embeddings import MockEmbedding
        from utils.const import (
            ExecutionTrace, EnvironmentState, OptimizationProblem,
            Solution, Feedback, FeedbackType, VariableDefinition
        )
        
        kb = KnowledgeService(
            embedding_model=MockEmbedding(),
            storage_path=self.temp_dir
        )
        
        summarizer = SummarizerAgent(kb=kb)
        
        trace = ExecutionTrace(
            trace_id="test_001",
            environment=EnvironmentState(user_instruction="测试"),
            problem=OptimizationProblem(
                objective_function_latex=r"\min x",
                variables=[VariableDefinition(name="x", lower_bound=0, upper_bound=10)]
            ),
            solution=Solution(
                is_feasible=True,
                algorithm_used="TestAlgo",
                decision_variables={"x": 5},
                objective_value=5
            ),
            feedback=Feedback(
                feedback_type=FeedbackType.PASSED,
                score=0.85
            )
        )
        
        summarizer.summarize(trace)
        
        stats = summarizer.get_statistics()
        self.assertIsNotNone(stats)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程"""
        from main import ADAOrchestrator
        from utils.const import EnvironmentState
        
        orchestrator = ADAOrchestrator(use_mock=True)
        
        env_state = EnvironmentState(
            user_instruction="测试优化",
            real_data={"value": 100.0}
        )
        
        result = orchestrator.run(env_state, max_retries=2)
        
        self.assertIn("success", result)
        self.assertIn("attempts", result)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeBase))
    suite.addTests(loader.loadTestsFromTestCase(TestPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestJudger))
    suite.addTests(loader.loadTestsFromTestCase(TestSummarizer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  ADA 系统综合测试")
    print("=" * 60)
    print()
    
    result = run_tests()
    
    print()
    print("=" * 60)
    print(f"测试完成: {result.testsRun} 个测试")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)

