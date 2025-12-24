import os
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer # 引入 Hugging Face 函式庫

# 建立 client 並設定環境變數 OPENAI_API_KEY
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 初始化本地 embedding 模型
embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# 測試用文件
test_documents = [
    "RAG 是一種檢索增強生成技術，它會先搜尋知識庫，再生成答案。",
    "LangChain 是一個用於構建 AI Agent 和 RAG 應用的框架。",
    "Semantic Kernel 是微軟推出的 AI agent 框架。",
    "FAISS向量資料庫用來儲存文件的 embedding，方便做語意搜尋。",
]

# --- Step 1: 建立文件的 embeddings 並儲存到 FAISS 向量資料庫 ---

def embed_documents(documents):
    # 使用 embed_model 的 encode 方法來建立文件的 embeddings
    # batch_size=32 可以在處理大量文件時提高效率
    embeddings = embed_model.encode(documents, batch_size=32, normalize_embeddings=True)
    # 將回傳的 NumPy 陣列轉換為 float32 格式，以符合 FAISS 要求
    return np.array(embeddings).astype("float32")

def create_faiss_index(embeddings):
    # 取得 embedding 的維度
    dim = embeddings.shape[1]
    # 建立 FAISS 索引
    index = faiss.IndexFlatL2(dim)
    # 將 embeddings 加入索引
    index.add(embeddings)
    return index

# 先建立 embeddings 和 FAISS 索引
embeddings = embed_documents(test_documents)
faiss_index = create_faiss_index(embeddings)

# --- Step 2: 使用 FAISS 向量資料庫進行語意搜尋 ---

def search(query, k=2):
    # 使用 embed_model 的 encode 方法建立查詢的 embedding
    query_embedding = embed_model.encode(query, normalize_embeddings=True)
    # 轉換為 numpy 陣列並調整形狀以符合 FAISS 搜尋要求
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
    
    # 在 FAISS 索引中搜尋最相似的 k 個文件
    distances, indices = faiss_index.search(query_embedding, k)
    return indices[0]

# --- Step 3: 使用檢索到的文件與使用者問題合在一起並呼叫 Chat Completions API ---
def rag_anser(query):
    # 搜尋最相似的文件
    indices = search(query, k=2)
    
    # 取得對應的文件內容
    retrieved_docs = [test_documents[i] for i in indices]
    
    # 建立系統提示詞，包含檢索到的文件
    system_prompt = "你是一個有用的助理。根據以下文件回答使用者的問題：\n" + "\n".join(retrieved_docs)
    
    # 加入使用者的問題
    query = f"使用者問題: {query}"
    
    # 呼叫 Chat Completions API 生成答案
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_input = input("請輸入您的問題(輸入exit或q離開): ")
        if user_input.lower() in ["exit", "q"]:
            print("離開問答系統")
            break
        answer = rag_anser(user_input)
        print("AI回覆:", answer)