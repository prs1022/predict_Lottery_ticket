from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = '8z1S8JrCHc9wcQnAsE_W'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
api.push_model(
    model_id="rensong/pipi",
    model_dir="/Users/upuphone/modelscope/nlp_gpt3_text-generation_1.3B" # 本地模型目录，要求目录中必须包含configuration.json
)