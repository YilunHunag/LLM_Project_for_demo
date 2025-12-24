import os
import json
from flask import Flask, request, abort # 引入 Flask 相關模組
import requests # 引入 requests 以呼叫外部 API
from openai import OpenAI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer # 引入 Hugging Face 函式庫
from dotenv import load_dotenv # 引入 dotenv 以讀取 .env 檔案

# 載入 .env 檔案中的環境變數
load_dotenv()

# LINE BOT 的基本設定
LINE_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# RAG 相關設定
embed_model = SentenceTransformer("all-MiniLM-L6-v2") # 使用較小的模型以減少資源消耗

# 測試用文件
test_documents = [
    "RAG 是一種檢索增強生成技術，它會先搜尋知識庫，再生成答案。",
    "LangChain 是一個用於構建 AI Agent 和 RAG 應用的框架。",
    "Semantic Kernel 是微軟推出的 AI agent 框架。",
    "FAISS向量資料庫用來儲存文件的 embedding，方便做語意搜尋。",
]

# 建立 embeddings 並儲存到 FAISS 向量資料庫
doc_embeddings = embed_model.encode(test_documents, convert_to_numpy=True) # 直接回傳 numpy 陣列
doc_embeddings = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12) # 正規化
doc_embeddings = doc_embeddings.astype("float32") # FAISS 要求 float32 格式
faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1]) # 使用內積 (IP) 作為相似度量
faiss_index.add(doc_embeddings) # 將 embeddings 加入索引

# search 函式
def search(query, k=2):
    q_emb = embed_model.encode([query], convert_to_numpy=True) 
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12) 
    q_emb = q_emb.astype("float32")
    _, indices = faiss_index.search(q_emb, k) # 搜尋相似文件
    return [test_documents[i] for i in indices[0]] # 取得對應的文件內容

# 使用檢索到的文件與使用者問題合在一起並呼叫 Chat Completions API
def rag_answer(query):
    retrieved_docs = search(query, k=2)
    system_prompt = "你是一個有用的助理。根據以下文件回答使用者的問題：\n" + "\n".join(retrieved_docs)
    prompt = f"使用者問題: {query}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# 建立 Flask Webhook
app = Flask(__name__)
# 設定 Webhook 路由
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True)  # 避免 JSON parse 失敗
    print("收到的資料:", data)

    if not data:
        # 可能是 LINE Verify 或其他空請求
        return "OK"

    events = data.get("events", [])
    if not events:
        return "OK"  # 沒有事件也回 200

    for event in events:
        # 只處理文字訊息
        if event.get("type") == "message" and event["message"]["type"] == "text":
            user_message = event["message"]["text"]
            print("使用者訊息:", user_message)
            answer = rag_answer(user_message)  # 取得 RAG 回答
            reply_message(event["replyToken"], answer)
        else:
            print("非文字訊息或其他事件，忽略")

    return "OK"   

# 回覆訊息給使用者
def reply_message(token, text):
    data = {
        "replyToken": token,
        "messages": [{"type": "text", "text": text}]
    }
    requests.post("https://api.line.me/v2/bot/message/reply", headers=LINE_HEADERS, data=json.dumps(data))

# 啟動 Flask 應用程式
if __name__ == "__main__":
    app.run(port=5000)
