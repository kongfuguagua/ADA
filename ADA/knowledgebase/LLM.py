# -*- coding: utf-8 -*-
"""
LLM 服务封装
支持 OpenAI 兼容的 Chat API
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Type

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# RAG 提示模板
RAG_PROMPT_TEMPLATE = """
使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。

问题: {question}

可参考的上下文：
```
{context}
```

如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。

有用的回答:
"""


class BaseChatModel:
    """Chat 模型基类"""
    
    def __init__(self, model: str = None) -> None:
        self.model = model
    
    def chat(
        self, 
        prompt: str, 
        history: List[Dict[str, str]] = None,
        context: str = ""
    ) -> str:
        """
        对话接口
        
        Args:
            prompt: 用户输入
            history: 对话历史
            context: RAG 上下文
        
        Returns:
            模型回复
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        history: List[Dict[str, str]] = None
    ) -> BaseModel:
        """
        结构化输出
        
        Args:
            prompt: 用户输入
            response_model: 期望的输出模型
            history: 对话历史
        
        Returns:
            解析后的 Pydantic 对象
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def chat_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        带工具调用的对话
        
        Args:
            prompt: 用户输入
            tools: 工具定义列表
            history: 对话历史
        
        Returns:
            包含 content 和 tool_calls 的字典
        """
        raise NotImplementedError("子类必须实现此方法")


class OpenAIChat(BaseChatModel):
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
            api_key: API Key
            base_url: Base URL
            temperature: 温度参数
            max_tokens: 最大生成长度
        """
        self.model = model or os.getenv("CLOUD_MODEL", "deepseek-chat")
        self.api_key = api_key or os.getenv("CLOUD_API_KEY", "")
        self.base_url = base_url or os.getenv("CLOUD_BASE_URL", "https://api.deepseek.com")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _build_messages(
        self, 
        prompt: str, 
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []
        
        # 系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 历史消息
        if history:
            messages.extend(history)
        
        # 当前消息
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def chat(
        self, 
        prompt: str, 
        history: List[Dict[str, str]] = None,
        context: str = "",
        system_prompt: str = None
    ) -> str:
        """
        对话接口
        
        Args:
            prompt: 用户输入
            history: 对话历史
            context: RAG 上下文
            system_prompt: 系统提示
        
        Returns:
            模型回复
        """
        # 如果有上下文，使用 RAG 模板
        if context:
            prompt = RAG_PROMPT_TEMPLATE.format(question=prompt, context=context)
        
        messages = self._build_messages(prompt, history, system_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return f"Error: {str(e)}"
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> BaseModel:
        """
        结构化输出（JSON Mode）
        
        Args:
            prompt: 用户输入
            response_model: 期望的输出模型
            history: 对话历史
            system_prompt: 系统提示
        
        Returns:
            解析后的 Pydantic 对象
        """
        # 获取模型的 JSON Schema
        schema = response_model.model_json_schema()
        
        # 构建提示
        structured_prompt = f"""{prompt}

请严格按照以下 JSON Schema 格式输出：
```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

只输出 JSON，不要包含其他内容。
"""
        
        messages = self._build_messages(structured_prompt, history, system_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # 结构化输出使用较低温度
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # 解析 JSON
            data = json.loads(content)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            raise
        except Exception as e:
            print(f"结构化输出失败: {e}")
            raise
    
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
            tools: 工具定义列表（OpenAI 格式）
            history: 对话历史
            system_prompt: 系统提示
        
        Returns:
            包含 content 和 tool_calls 的字典
        """
        messages = self._build_messages(prompt, history, system_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
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
        except Exception as e:
            print(f"工具调用失败: {e}")
            return {"content": f"Error: {str(e)}", "tool_calls": []}


class MockLLM(BaseChatModel):
    """模拟 LLM（用于测试）"""
    
    def __init__(self):
        super().__init__("mock")
        self.responses = {}
    
    def set_response(self, pattern: str, response: str):
        """设置模拟响应"""
        self.responses[pattern] = response
    
    def chat(
        self, 
        prompt: str, 
        history: List[Dict[str, str]] = None,
        context: str = "",
        system_prompt: str = None
    ) -> str:
        """返回模拟响应"""
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response
        return f"Mock response for: {prompt[:50]}..."
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> BaseModel:
        """返回模型的默认实例"""
        return response_model()
    
    def chat_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """返回空的工具调用"""
        return {"content": "Mock response", "tool_calls": []}


# ============= 测试代码 =============
if __name__ == "__main__":
    print("测试 MockLLM:")
    mock = MockLLM()
    mock.set_response("你好", "你好！我是 Mock LLM。")
    
    response = mock.chat("你好，请介绍一下自己")
    print(f"响应: {response}")
    
    # 如果配置了 API，测试真实的 LLM
    if os.getenv("CLOUD_API_KEY"):
        print("\n测试 OpenAI Chat:")
        llm = OpenAIChat()
        response = llm.chat("用一句话介绍什么是电力系统调度")
        print(f"响应: {response}")
