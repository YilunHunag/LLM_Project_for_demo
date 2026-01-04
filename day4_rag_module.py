# 從day3切割出來的RAG模組 
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()

# 初始化 OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 初始化 embedding 模型
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 測試用文件
test_documents = [
    "RAG 是一種檢索增強生成技術，它會先搜尋知識庫，再生成答案。",
    "LangChain 是一個用於構建 AI Agent 和 RAG 應用的框架。",
    "Semantic Kernel 是微軟推出的 AI agent 框架。",
    "FAISS向量資料庫用來儲存文件的 embedding，方便做語意搜尋。",
    "黃義倫是一位AI工程師, 專注於自然語言處理和機器學習領域。",
    "舒果農企業有限公司: 無毒農成立於2015年，起因於食安問題，致力於從改變台灣農業出發，透過自有技術、行銷、物流與客服團隊發展電商平台，建立公正公開的制度，減少農產銷中間剝削，推動第二方驗證與農友認證補助基金，讓台灣農產品安全地走向世界並促進農業永續發展。"
]

# 建立 embeddings 並儲存到 FAISS 向量資料庫
doc_embeddings = embed_model.encode(test_documents, convert_to_numpy=True)
doc_embeddings = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12)  # 正規化
doc_embeddings = doc_embeddings.astype("float32")
faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
faiss_index.add(doc_embeddings)

# search 函式
def search(query, k=2):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    q_emb = q_emb.astype("float32")
    _, indices = faiss_index.search(q_emb, k)
    return [test_documents[i] for i in indices[0]]

# RAG 問答
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
