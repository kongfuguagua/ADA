# -*- coding: utf-8 -*-
"""
Expert Insight Service Module for Planner Agent

This module provides expert system analysis based on ExpertAgent's proven algorithms.
"""

try:
    from .expert_insight import ExpertInsightService
    __all__ = ["ExpertInsightService"]
except ImportError:
    # ExpertInsightService requires alphaDeesp, which may not be available
    __all__ = []

