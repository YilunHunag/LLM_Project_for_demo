from openai import OpenAI
import os

#建立client並設定環境變數OPENAI_API_KEY
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def simple_chat():
    print("===簡單的問答系統===")
    #持續詢問使用者問題直到輸入exit或q離開
    while True:
        use_input = input("請輸入您的問題(輸入exit或q離開): ")
        if use_input.lower() in ["exit", "q"]:
            print("離開問答系統")
            break
        
        #呼叫chat.completions.create建立對話
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一個有用的助理"},
                {"role": "user", "content": use_input}
            ]
        )
        #印出AI的回覆
        print("AI回覆:", response.choices[0].message.content)

#執行簡單問答系統
if __name__ == "__main__":
    simple_chat()
