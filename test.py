import requests

# url = 'http://localhost:8084/llm/flux-data'
# response = requests.get(url, stream=True)
#
# if response.status_code == 200:
#     for line in response.iter_lines():
#         if line:
#             print(line.decode('utf-8'))



# url = 'http://localhost:8084/llm/v1/completion'
url = 'https://km-dev.myvu.cn/llm/llm/v1/completion'
headers = {'Content-Type': 'application/json'}
data = {
    'uid': 'xxx',
    'query': '学习一下灰太狼说话的语气，责怪一下老婆没有早点回家',
    'streaming': True,
    'params': {}
}

response = requests.post(url, json=data, headers=headers,stream=True)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))
            print("\n")
