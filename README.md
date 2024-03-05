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

# 実行方法
Dockerコンテナ内に入り、`main.py`を実行する。  
コード内のプロンプトは随時変更し、応答の変化を見るのも良い。  
