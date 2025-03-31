import aiohttp
import asyncio

async def fetch_chat_data(session, url, uid, query, apiKey):
    try:
        async with session.post(url, json={
            "query": query,
            "uid": uid,
            "streaming": True  # 确保这个字段与你的服务端期望的字段匹配
        }, headers={
            "Authorization": f"Bearer {apiKey}"
        }) as response:
            if response.status == 200:
                # 处理流式响应
                data = await response.json()
                return data
            else:
                print(f"Error: {response.status}")
    except aiohttp.ClientError as e:
        print(f"Client error: {e}")

async def main():
    # 你的 API 密钥和请求参数
    apiKey = "your_api_key"
    uid = "aa"
    query = "fuck"

    # 你的服务端 URL
    url = "http://localhost:8084/llm/v1/chat"  # 确保使用正确的协议

    async with aiohttp.ClientSession() as session:
        data = await fetch_chat_data(session, url, uid, query, apiKey)
        if data:
            print(data)

# 运行主函数
asyncio.run(main())