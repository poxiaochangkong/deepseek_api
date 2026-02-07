import os
import logging
import uuid  # 用于生成唯一的会话 ID
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="tria1.env")

class DeepSeekChat:
    def __init__(self, model: str = "deepseek-chat", max_context_len: int = 10):
        self.api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("未找到 DEEPSEEK_API_KEY")

        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.model = model
        self.max_context_len = max_context_len
        
        # 初始化第一个会话
        self.session_id: str = ""
        self.history: List[Dict[str, str]] = []
        self.new_session() # 调用新建会话方法

    def new_session(self) -> str:
        """
        核心功能：开启一个全新的会话。
        清空历史记录，重新初始化 System Prompt，并生成新的 Session ID。
        """
        self.session_id = str(uuid.uuid4())[:8] # 取前8位方便观察
        self.history = [
            {"role": "system", "content": "你是一位精通技术的专业助手。"}
        ]
        logger.info(f"--- 已开启新对话 (Session ID: {self.session_id}) ---")
        return self.session_id

    def _manage_context(self):
        """滑动窗口逻辑"""
        if len(self.history) > self.max_context_len:
            excess = len(self.history) - self.max_context_len
            self.history = [self.history[0]] + self.history[1 + excess:]

    def chat(self, user_input: str) -> Optional[str]:
        self.history.append({"role": "user", "content": user_input})
        self._manage_context()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history, # type: ignore
                stream=False
            )
            answer = response.choices[0].message.content
            if answer:
                self.history.append({"role": "assistant", "content": answer})
                return answer
        except Exception as e:
            logger.error(f"对话发生错误: {e}")
        return None

# --- 执行入口 ---
if __name__ == "__main__":
    bot = DeepSeekChat()
    
    print(f"欢迎使用 DeepSeek 终端助手 | 当前会话: {bot.session_id}")
    print("指令说明: [new] 新建对话 | [exit] 退出程序")

    while True:
        prompt = input(f"\n[{bot.session_id}] 用户 >> ").strip()
        
        if not prompt: continue
        
        # 捕获新建对话指令
        if prompt.lower() == 'new':
            bot.new_session()
            print(f"\n>>> 历史记录已清理，新会话已开启 (ID: {bot.session_id})")
            continue
            
        if prompt.lower() in ['exit', 'quit']:
            break
            
        reply = bot.chat(prompt)
        if reply:
            print(f"\nDeepSeek >> {reply}")