# -*- coding: utf-8 -*-
"""
LLM 服务封装
支持 OpenAI 兼容的 Chat API

注意：此模块不包含 Mock 实现，仅提供真实 LLM 服务。
如果 API 配置无效，将抛出异常而非静默失败。
"""

import os
import json
from typing import List, Dict, Any, Type
from abc import ABC, abstractmethod

from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class BaseLLM(ABC):
    """LLM 服务基类"""
    
    def __init__(self, model: str = None) -> None:
        self.model = model
    
    @abstractmethod
    def chat(
        self, 
        prompt: str, 
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> str:
        """
        对话接口
        
        Args:
            prompt: 用户输入
            history: 对话历史
            system_prompt: 系统提示
        
        Returns:
            模型回复
        
        Raises:
            RuntimeError: API 调用失败
        """
        pass
    
    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> BaseModel:
        """
        结构化输出
        
        Args:
            prompt: 用户输入
            response_model: 期望的输出模型
            history: 对话历史
            system_prompt: 系统提示
        
        Returns:
            解析后的 Pydantic 对象
        
        Raises:
            RuntimeError: API 调用或解析失败
        """
        pass
    
    @abstractmethod
    def chat_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        带工具调用的对话
        
        Args:
            prompt: 用户输入
            tools: 工具定义列表
            history: 对话历史
            system_prompt: 系统提示
        
        Returns:
            包含 content 和 tool_calls 的字典
        
        Raises:
            RuntimeError: API 调用失败
        """
        pass


class OpenAIChat(BaseLLM):
    """OpenAI 兼容的 Chat 实现"""
    
    def __init__(
        self, 
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> None:
        """
        初始化 OpenAI Chat
        
        Args:
            model: 模型名称
            api_key: API Key (必须提供)
            base_url: Base URL
            temperature: 温度参数
            max_tokens: 最大生成长度
        
        Raises:
            ImportError: openai 模块未安装
            ValueError: API Key 未配置
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 模块未安装，请运行: pip install openai")
        
        self.model = model or os.getenv("CLOUD_MODEL", "deepseek-chat")
        self.api_key = api_key or os.getenv("CLOUD_API_KEY", "")
        self.base_url = base_url or os.getenv("CLOUD_BASE_URL", "https://api.deepseek.com")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 验证 API Key
        if not self.api_key:
            raise ValueError("API Key 未配置，请设置 CLOUD_API_KEY 环境变量或传入 api_key 参数")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 检测是否是需要处理 enable_thinking 的模型
        self._needs_thinking_fix = self._is_thinking_model(self.model)
    
    def _is_thinking_model(self, model_name: str) -> bool:
        """检测是否是支持深度思考的模型（需要特殊处理）"""
        if not model_name:
            return False
        model_lower = model_name.lower()
        # 检测 qwen3 系列或其他支持 thinking 的模型
        return "qwen3" in model_lower or "qwen" in model_lower
    
    def _build_messages(
        self, 
        prompt: str, 
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def chat(
        self, 
        prompt: str, 
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> str:
        """对话接口"""
        messages = self._build_messages(prompt, history, system_prompt)
        
        # 构建请求参数
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # 对于支持深度思考的模型，在非流式调用时必须设置 enable_thinking=False
        if self._needs_thinking_fix:
            request_params["extra_body"] = {"enable_thinking": False}
        
        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> BaseModel:
        """结构化输出（JSON Mode）"""
        from .json_utils import safe_json_dumps
        
        schema = response_model.model_json_schema()
        
        structured_prompt = f"""{prompt}

请严格按照以下 JSON Schema 格式输出：
```json
{safe_json_dumps(schema, ensure_ascii=False, indent=2)}
```

只输出 JSON，不要包含其他内容。
"""
        
        messages = self._build_messages(structured_prompt, history, system_prompt)
        
        # 构建请求参数
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # 结构化输出使用较低温度
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        # 对于支持深度思考的模型，在非流式调用时必须设置 enable_thinking=False
        if self._needs_thinking_fix:
            request_params["extra_body"] = {"enable_thinking": False}
        
        response = self.client.chat.completions.create(**request_params)
        
        content = response.choices[0].message.content
        data = json.loads(content)
        return response_model.model_validate(data)
    
    def chat_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """带工具调用的对话"""
        messages = self._build_messages(prompt, history, system_prompt)
        
        # 构建请求参数
        request_params = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # 对于支持深度思考的模型，在非流式调用时必须设置 enable_thinking=False
        if self._needs_thinking_fix:
            request_params["extra_body"] = {"enable_thinking": False}
        
        response = self.client.chat.completions.create(**request_params)
        
        message = response.choices[0].message
        
        result = {
            "content": message.content or "",
            "tool_calls": []
        }
        
        if message.tool_calls:
            for tc in message.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                })
        
        return result
    
    def chat_with_context(
        self,
        prompt: str,
        context: str,
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> str:
        """
        带 RAG 上下文的对话
        
        Args:
            prompt: 用户问题
            context: 检索到的上下文
            history: 对话历史
            system_prompt: 系统提示
        
        Returns:
            模型回复
        """
        rag_prompt = f"""使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。

问题: {prompt}

可参考的上下文：
```
{context}
```

有用的回答:"""
        
        return self.chat(rag_prompt, history, system_prompt)


# ============= 测试代码 =============
if __name__ == "__main__":
    print("测试 OpenAI Chat:")
    
    # 检查 API 配置
    api_key = os.getenv("CLOUD_API_KEY")
    if not api_key:
        print("错误: 未配置 CLOUD_API_KEY 环境变量")
        print("请在 .env 文件中设置 CLOUD_API_KEY")
        exit(1)
    
    try:
        llm = OpenAIChat()
        response = llm.chat("用一句话介绍什么是电力系统调度")
        print(f"响应: {response}")
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        exit(1)

