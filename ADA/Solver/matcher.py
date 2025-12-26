# -*- coding: utf-8 -*-
"""
算法匹配模块
实现问题-算法对齐评分机制
"""

import math
from typing import List, Dict, Optional

from utils.interact import BaseSolverStrategy
from utils.logger import get_logger

from .feature import ProblemFeatures

logger = get_logger("Matcher")


class AlgorithmMatcher:
    """
    算法匹配器
    
    根据问题特征向量 φ 和算法能力向量 ψ(A)，
    计算对齐评分 G(A, φ) 选择最佳算法。
    """
    
    # 特征到能力的映射关系
    # phi[i] 需要的能力对应 psi[j]
    FEATURE_CAPABILITY_MAP = {
        0: 1,  # 非凸性 -> 非凸处理能力
        1: 1,  # 非线性 -> 非凸处理能力
        2: 2,  # 约束紧迫度 -> 约束处理能力
        3: 4,  # 离散性 -> 全局最优性（组合搜索）
        4: 3,  # 规模 -> 速度
    }
    
    # BM25 调节参数
    K1 = 1.2
    B = 0.75
    
    def __init__(self, algorithms: Dict[str, BaseSolverStrategy] = None):
        """
        初始化匹配器
        
        Args:
            algorithms: 算法名称到实例的映射
        """
        self._algorithms: Dict[str, BaseSolverStrategy] = algorithms or {}
        self._cached_weights: Optional[List[float]] = None
    
    def register(self, algorithm: BaseSolverStrategy) -> None:
        """注册算法"""
        self._algorithms[algorithm.meta.name] = algorithm
        self._cached_weights = None  # 清除缓存
        logger.debug(f"注册算法: {algorithm.meta.name}")
    
    def unregister(self, name: str) -> bool:
        """注销算法"""
        if name in self._algorithms:
            del self._algorithms[name]
            self._cached_weights = None
            return True
        return False
    
    def match(self, features: ProblemFeatures) -> BaseSolverStrategy:
        """
        匹配最佳算法
        
        Args:
            features: 问题特征
        
        Returns:
            最佳算法实例
        
        Raises:
            ValueError: 无可用算法
        """
        if not self._algorithms:
            raise ValueError("没有可用的求解算法")
        
        phi = features.to_vector()
        scores = self.compute_scores(phi)
        
        # 选择得分最高的算法
        best_name = max(scores, key=scores.get)
        best_algo = self._algorithms[best_name]
        
        logger.info(f"算法匹配完成: {best_name} (score={scores[best_name]:.4f})")
        
        return best_algo
    
    def compute_scores(self, phi: List[float]) -> Dict[str, float]:
        """
        计算所有算法的对齐评分
        
        Args:
            phi: 问题特征向量
        
        Returns:
            算法名称到评分的映射
        """
        weights = self._get_feature_weights(phi)
        scores = {}
        
        for name, algo in self._algorithms.items():
            scores[name] = self._alignment_score(algo, phi, weights)
        
        return scores
    
    def _alignment_score(
        self,
        algorithm: BaseSolverStrategy,
        phi: List[float],
        weights: List[float]
    ) -> float:
        """
        计算单个算法的对齐评分 G(A, φ)
        
        基于 BM25 风格的评分公式:
        G(A, φ) = Σ w(f_i) * (ψ_j(A) * (k1 + 1)) / (ψ_j(A) + k1 * (1 - b + b * f_i))
        
        Args:
            algorithm: 算法实例
            phi: 问题特征向量
            weights: 特征权重向量
        
        Returns:
            对齐评分
        """
        psi = algorithm.get_capability_vector()
        score = 0.0
        
        for i in range(len(phi)):
            # 获取该特征对应的能力维度
            j = self.FEATURE_CAPABILITY_MAP.get(i, i)
            if j >= len(psi):
                continue
            
            capability = psi[j]
            feature = phi[i]
            
            # BM25 风格评分
            numerator = capability * (self.K1 + 1)
            denominator = capability + self.K1 * (1 - self.B + self.B * feature)
            
            if denominator > 0:
                score += weights[i] * (numerator / denominator)
        
        return score
    
    def _get_feature_weights(self, phi: List[float]) -> List[float]:
        """
        计算特征权重（IDF 风格）
        
        w(f_i) = ln(1 + 总能力 / 该特征对应能力总和)
        
        Args:
            phi: 问题特征向量
        
        Returns:
            特征权重向量
        """
        if self._cached_weights is not None:
            return self._cached_weights
        
        # 计算所有算法的能力总和
        total_capability = 0.0
        capability_sums = [0.0] * 5  # 5 维能力
        
        for algo in self._algorithms.values():
            psi = algo.get_capability_vector()
            for i, cap in enumerate(psi):
                total_capability += cap
                if i < len(capability_sums):
                    capability_sums[i] += cap
        
        # 计算权重
        epsilon = 1e-6
        weights = []
        
        for i in range(len(phi)):
            j = self.FEATURE_CAPABILITY_MAP.get(i, i)
            if j < len(capability_sums):
                w = math.log(1 + total_capability / (capability_sums[j] + epsilon))
            else:
                w = 1.0
            weights.append(w)
        
        self._cached_weights = weights
        return weights
    
    def get_algorithm(self, name: str) -> Optional[BaseSolverStrategy]:
        """获取指定算法"""
        return self._algorithms.get(name)
    
    def list_algorithms(self) -> List[str]:
        """列出所有可用算法"""
        return list(self._algorithms.keys())
    
    def get_algorithm_info(self) -> str:
        """获取所有算法的描述信息"""
        lines = []
        for name, algo in self._algorithms.items():
            meta = algo.meta
            caps = algo.get_capability_vector()
            lines.append(f"- {name}: {meta.description}")
            lines.append(f"  能力向量: {[f'{c:.2f}' for c in caps]}")
        return "\n".join(lines)


# ============= 测试代码 =============
if __name__ == "__main__":
    from .Template import (
        ConvexOptimizer,
        PSOOptimizer,
        GeneticOptimizer,
    )
    
    print("测试 AlgorithmMatcher:")
    
    # 创建匹配器
    matcher = AlgorithmMatcher()
    matcher.register(ConvexOptimizer())
    matcher.register(PSOOptimizer())
    matcher.register(GeneticOptimizer())
    
    print(f"\n可用算法: {matcher.list_algorithms()}")
    
    # 测试凸问题特征
    convex_features = ProblemFeatures(
        variable_count=2,
        is_convex=True,
        is_linear=False,
        non_convexity_score=0.0,
        non_linearity_score=0.3,
    )
    
    print("\n凸问题匹配:")
    scores = matcher.compute_scores(convex_features.to_vector())
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.4f}")
    
    best = matcher.match(convex_features)
    print(f"  最佳算法: {best.meta.name}")
    
    # 测试非凸问题特征
    nonconvex_features = ProblemFeatures(
        variable_count=10,
        is_convex=False,
        is_linear=False,
        non_convexity_score=0.8,
        non_linearity_score=0.9,
    )
    
    print("\n非凸问题匹配:")
    scores = matcher.compute_scores(nonconvex_features.to_vector())
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.4f}")
    
    best = matcher.match(nonconvex_features)
    print(f"  最佳算法: {best.meta.name}")

