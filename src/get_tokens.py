import os
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken

GPT_MODEL = "gpt-3.5-turbo"

def num_tokens(text, model=GPT_MODEL):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def save_fig(df, colum_name, path_save, title='token', xlabel='index', ylabel='token'):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[colum_name], marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(path_save)

if __name__ == '__main__':
    # data.jsonlをDataFrameで読み込む
    path_data = os.path.join('data', 'data.jsonl')
    df_data = pd.read_json(path_data, orient='records', lines=True)

    # tokenを計算
    df_data['token'] = df_data['Question'].apply(num_tokens)

    # 可視化
    save_fig(df=df_data, colum_name='token', path_save='temp_Q.png')

    # QuestionとAnswerを結合
    df_data['QnA'] = df_data['Question'] + " " + df_data['Answer']

    df_data['token_QnA'] = df_data['QnA'].apply(num_tokens)
    # 可視化
    save_fig(df=df_data, colum_name='token_QnA', path_save='temp_QnA.png')

