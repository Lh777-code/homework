
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


MODEL_NAME = "qwen2.5:7b"
API_KEY = "123"
BASE_URL = "http://127.0.0.1:11434/v1"
TEMPERATURE = 0.7

SYSTEM_PROMPT = "你是一个友好、专业的中文 AI 助手，回答简洁准确，必要时分点说明。"


llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,        
    base_url=BASE_URL,      
    temperature=TEMPERATURE,
)


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


# 接口
@app.post("/chat")
async def chat(req: ChatRequest):
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


# 内置聊天页面（纯原生 HTML + JS）
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
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{module_name}:app", host="127.0.0.1", port=8080, reload=True)
