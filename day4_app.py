#從day3切割出來的LINE BOT主程式
import os
import json
import requests
from flask import Flask, request
from dotenv import load_dotenv
from day4_rag_module import rag_answer  # 引用 RAG 邏輯

# 載入 .env 檔案
load_dotenv()

# LINE BOT 基本設定
LINE_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
}

# 建立 Flask app
app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True)
    print("收到的資料:", data)

    if not data:
        return "OK"

    events = data.get("events", [])
    if not events:
        return "OK"

    for event in events:
        if event.get("type") == "message" and event["message"]["type"] == "text":
            user_message = event["message"]["text"]
            print("使用者訊息:", user_message)
            answer = rag_answer(user_message)
            reply_message(event["replyToken"], answer)
        else:
            print("非文字訊息或其他事件，忽略")

    return "OK"

# 回覆訊息
def reply_message(token, text):
    data = {
        "replyToken": token,
        "messages": [{"type": "text", "text": text}]
    }
    requests.post("https://api.line.me/v2/bot/message/reply", headers=LINE_HEADERS, data=json.dumps(data))

# 啟動 Flask
if __name__ == "__main__":
    app.run(port=5000, debug=True)
