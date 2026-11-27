"""
FastAPI + LangChain 多轮对话聊天服务（纯净版）

安装依赖：
    pip install fastapi uvicorn langchain langchain-openai pydantic

启动：
    python main.py
浏览器访问：
    http://127.0.0.1:8000
"""

import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# =====================================================================
# 顶部集中配置区：切换模型只改这里（ChatOpenAI 兼容所有 OpenAI 协议服务）
# ---------------------------------------------------------------------
# 1) DeepSeek：
#       BASE_URL   = "https://api.deepseek.com/v1"
#       MODEL_NAME = "deepseek-chat"
#       API_KEY    = "sk-xxxx"
#
# 2) 硅基流动：
#       BASE_URL   = "https://api.siliconflow.cn/v1"
#       MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
#       API_KEY    = "sk-xxxx"
#
# 3) 本地 Ollama（先执行 `ollama serve`，并 `ollama pull qwen2.5`）：
#       BASE_URL   = "http://127.0.0.1:11434/v1"   # 踩坑点：必须带 /v1 后缀
#       MODEL_NAME = "qwen2.5"
#       API_KEY    = "dummy"                        # Ollama 不校验，但 SDK 要求非空
# =====================================================================
MODEL_NAME = "qwen2.5:7b"
API_KEY = "123"
BASE_URL = "http://127.0.0.1:11434/v1"
TEMPERATURE = 0.7

SYSTEM_PROMPT = "你是一个友好、专业的中文 AI 助手，回答简洁准确，必要时分点说明。"

# =====================================================================
# 模型与提示词
# =====================================================================
# ChatOpenAI 采用 OpenAI 兼容协议，只要服务支持 /v1/chat/completions，
# 就能通过 base_url 接入，无需改动下面的调用逻辑。
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,        # 直接传字符串即可，内部会转成 SecretStr
    base_url=BASE_URL,      # 切换服务商的关键参数
    temperature=TEMPERATURE,
)

# 三段式提示词模板：
#   1. system      ：角色设定
#   2. MessagesPlaceholder：多轮对话历史占位符。它不会把历史拼成一段字符串，
#      而是把 HumanMessage / AIMessage 对象原样插入到该位置，保留角色信息。
#      变量名必须与 invoke 时传入的字典 key 严格一致（这里是 chat_history）。
#   3. human       ：本轮用户输入
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ]
)

# LangChain 标准管道写法：提示词 | 模型
chain = prompt | llm

app = FastAPI(title="LangChain 多轮对话 Demo")


class ChatRequest(BaseModel):
    user_input: str
    # 前端传来的历史是 [[用户文本, AI回复], ...]，这里用 tuple 承接即可
    chat_history: list[tuple[str, str]] = Field(default_factory=list)


# =====================================================================
# 接口
# =====================================================================
@app.post("/chat")
async def chat(req: ChatRequest):
    # 把 [(user, ai), ...] 循环转换成 LangChain 消息对象。
    # 踩坑点：绝不能把历史拼成一个字符串塞给模型，
    # 那样会丢失“谁说的”这一角色信息，多轮对话就会乱。
    history_messages = []
    for user_text, ai_text in req.chat_history:
        history_messages.append(HumanMessage(content=user_text))
        history_messages.append(AIMessage(content=ai_text))

    try:
        # chat_history 传给 MessagesPlaceholder，user_input 填进 human 占位
        result = await chain.ainvoke(
            {"chat_history": history_messages, "user_input": req.user_input}
        )
    except Exception as exc:
        # 把真实的模型调用错误返回给前端，方便排查（key 错误、base_url 写错等）
        raise HTTPException(status_code=500, detail=f"模型调用失败：{exc}") from exc

    return {"reply": result.content}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


# =====================================================================
# 内置聊天页面（纯原生 HTML + JS，无任何前端框架）
# =====================================================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LangChain 多轮对话</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; justify-content: center;
    font-family: system-ui, "Microsoft YaHei", sans-serif; background: #f4f5f7;
  }
  .app {
    width: 100%; max-width: 760px; height: 100vh; display: flex; flex-direction: column;
    background: #fff; box-shadow: 0 0 24px rgba(0, 0, 0, .06);
  }
  header {
    padding: 16px 20px; font-weight: 600; font-size: 16px;
    border-bottom: 1px solid #eee;
  }
  #messages {
    flex: 1; overflow-y: auto; padding: 20px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .msg {
    max-width: 78%; padding: 10px 14px; border-radius: 12px;
    line-height: 1.65; white-space: pre-wrap; word-break: break-word;
    font-size: 15px;
  }
  .user { align-self: flex-end; background: #2f6fed; color: #fff; border-bottom-right-radius: 4px; }
  .ai   { align-self: flex-start; background: #f0f2f5; color: #222; border-bottom-left-radius: 4px; }
  .tip  { align-self: center; color: #999; font-size: 13px; }
  .input-bar {
    display: flex; gap: 10px; padding: 14px 20px; border-top: 1px solid #eee;
  }
  #input {
    flex: 1; height: 48px; padding: 12px; resize: none; font-size: 15px;
    font-family: inherit; border: 1px solid #ddd; border-radius: 10px; outline: none;
  }
  #input:focus { border-color: #2f6fed; }
  #send {
    padding: 0 22px; border: none; border-radius: 10px; font-size: 15px;
    background: #2f6fed; color: #fff; cursor: pointer;
  }
  #send:disabled { background: #9db8f5; cursor: not-allowed; }
</style>
</head>
<body>
<div class="app">
  <header>AI 聊天（多轮对话）</header>
  <div id="messages"><div class="tip">你好，有什么可以帮你？</div></div>
  <div class="input-bar">
    <textarea id="input" placeholder="输入消息，Enter 发送，Shift+Enter 换行"></textarea>
    <button id="send">发送</button>
  </div>
</div>
<script>
  // chatHistory 保存 [[用户文本, AI回复], ...]，与后端接口格式严格一致。
  // 不能拼成字符串，否则后端无法还原成 HumanMessage / AIMessage。
  const chatHistory = [];
  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('input');
  const sendBtn = document.getElementById('send');

  function addBubble(text, role) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.textContent = text;   // 用 textContent 而非 innerHTML，避免 HTML 注入
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  async function send() {
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = '';
    addBubble(text, 'user');
    const thinking = addBubble('正在思考…', 'ai');
    sendBtn.disabled = true;

    try {
      // 注意：这里发送的是“本轮之前”的历史。
      // 当前这句话要等拿到回复后再 push，否则本轮问题会在历史里重复出现。
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: text, chat_history: chatHistory })
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || ('HTTP ' + resp.status));
      }

      const data = await resp.json();
      thinking.textContent = data.reply;
      chatHistory.push([text, data.reply]);   // 本轮问答完成后再写入历史
    } catch (e) {
      thinking.textContent = '出错了：' + e.message;
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  sendBtn.addEventListener('click', send);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # 踩坑点一：开了 reload=True 时，必须传 "模块名:app" 字符串，
    # 不能传 app 对象，否则 uvicorn 无法重新导入模块，reload 不生效。
    #
    # 踩坑点二：127.0.0.1 只允许本机访问；0.0.0.0 监听所有网卡（局域网可访问，
    # 本地开发用 127.0.0.1 更安全）。浏览器也必须用 127.0.0.1:8000 打开。
    #
    # 踩坑点三：模块名必须和当前文件名一致！原来写死 "main:app"，
    # 运行本文件（简单聊天工具.py）时 uvicorn 会去找名为 main 的模块，
    # 结果要么 ModuleNotFoundError，要么误加载别的 main.py。
    # 这里按当前文件名动态生成，以后改名也不用再动这一行。
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{module_name}:app", host="127.0.0.1", port=8080, reload=True)