# RAG_Azure
Azureを使ったRAGにより、独自データに基づいた応答ができるChatを作成する。  
OpenAIのAPIを使った書き方と若干違いがある。  

Azureを利用するメリットは、セキュアな環境を確保できること。  

# 環境
LinuxベースOS（Mac他）  
Docker利用  

# 事前準備
## Docker構築
[`RAG_Azure/docker/README.md`](./docker/README.md)に従い、Docker環境を構築する。  

## データ準備
任意のPDFを入手し、[`RAG_Azure/data/`](./data/)に保存する。  

## `.ENV`ファイルの準備
以下フォーマットで、[`RAG_Azure/src/`](./src/)に`.ENV`ファイルを作成する。  
```.ENV
API_TYPE=azure
API_KEYS={YOUR API KEY}
API_ENDPOINT={YOUR API ENDPOINT}
API_VERSION=2024-02-15-preview
MODEL_NAME={YOUR DEPLOYED EMBEDDING MODEL NAME}
EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_MODEL={YOUR DEPLOYED MODEL NAME}
```

# 実行方法
Dockerコンテナ内に入り、`main.py`を実行する。  
コード内のプロンプトは随時変更し、応答の変化を見るのも良い。  

# Webアプリ使い方
`Streamlit`を使ったWebアプリも実行可能。  
Dockerコンテナに入り、以下実行する。  
```bash
$ streamlit run chat_bot_st.py
```
その後、[`http://0.0.0.0:8501`](http://0.0.0.0:8501)へアクセスすると、チャット画面が立ち上がる。  
クエリをいれると、応答ができる。  

なお、アクセスするアドレスのIPとポートは、[`./src/.streamlit/config.toml`](./src/.streamlit/config.toml)内で設定可能。  

また、スクリプトは以下リポジトリを参考にした。  
<https://github.com/joshuel09/chatbot-langChain/blob/main/langChainExp.py>  
