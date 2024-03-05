# Docker
## Dockerイメージの構築
以下の`bash`ファイルを実行する。
```bash
sh docker_builder.sh
```
## Dockerコンテナ立ち上げ
以下の`bash`ファイルを実行する。
```bash
sh docker_container.sh
```
## コンテナ内でjupyter notebookを実行する
`docker-compose.yml`でコンテナを立ち上げる。  
`RAG_Azure/docker`ディレクトリで、以下を実行する。  
```bash
docker compose up -d
```
続いて、コンテナ内でjupyter notebookを立ち上げる  
```bash
docker compose exec app bash
```
以下コマンドでjupyter notebookを起動、  
```
jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''
```
ブラウザを立ち上げ、[`http://localhost:8888`](http://localhost:8888)へアクセスし、jupyter notebookを実行する。  
