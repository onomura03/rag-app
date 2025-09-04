import streamlit as st
import os
import re
from pathlib import Path
import shutil
from llama_parse import LlamaParse
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ---
# 1. APIキーとパスワードの設定
# ---
if "LLAMA_CLOUD_API_KEY" not in os.environ:
    if "LLAMA_CLOUD_API_KEY" in st.secrets:
        os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets["LLAMA_CLOUD_API_KEY"]
    else:
        st.error("エラー: `LLAMA_CLOUD_API_KEY`が設定されていません。`secrets.toml`ファイルをご確認ください。")
        st.stop()

if "OPENAI_API_KEY" not in os.environ:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("エラー: `OPENAI_API_KEY`が設定されていません。`secrets.toml`ファイルをご確認ください。")
        st.stop()

# パスワード認証機能の追加
if "APP_PASSWORD" not in st.secrets:
    st.error("エラー: パスワード`APP_PASSWORD`が設定されていません。`secrets.toml`ファイルをご確認ください。")
    st.stop()
    
# セッション状態の初期化
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# ---
# 2. パスワード認証UIとロジック
# ---
# Streamlitでは、ボタンが押されたときにページ全体が再実行される。
# 認証状態をセッションに保持し、認証済みでない場合のみ認証UIを表示する。
if not st.session_state.authenticated:
    st.title("ログイン")
    password_input = st.text_input("パスワードを入力してください", type="password")
    
    if st.button("ログイン"):
        if password_input == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("パスワードが間違っています。")
    # 認証されていない場合は、ここでスクリプトの実行を停止し、メインUIを表示しない
    st.stop()

# ---
# 3. メインUIコンポーネント
# ---

@st.cache_resource
def get_db():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)


st.set_page_config(page_title="PDFチャット", layout="wide")
st.title("PDFドキュメント Q&Aチャット")

# ユーザーへの指示メッセージをタイトルの下に配置
if not Path("./chroma_db").exists():
    st.markdown("""
    **質問の前に、以下の対応を行ってください。**
    
    ① 左側の「ファイルを閲覧する」ボタンを押下し、PDFファイルをアップロード
    
    ② 「データベースを更新」を押下
    
    ③ データベースの構築が完了した後に、質問してください
    
    *※構築までに数分かかります※*
    """)

# サイドバー
with st.sidebar:
    st.header("PDFファイルのアップロード")
    uploaded_files = st.file_uploader("", type="pdf", accept_multiple_files=True)
    if st.button("データベースを更新"):
        if uploaded_files:
            # データベースを初期化
            if Path("./chroma_db").exists():
                shutil.rmtree("./chroma_db")
            
            with st.spinner("PDFファイルを解析し、データベースを構築中..."):
                documents = []
                for uploaded_file in uploaded_files:
                    # PDFを一時ファイルとして保存
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # LlamaParseで解析
                    loader = LlamaParse(
                        api_key=os.environ["LLAMA_CLOUD_API_KEY"],
                        result_type="markdown",
                        system_prompt="""資料の見出し、セクションタイトル、表やグラフを、主要なメタデータもすべて含めて、厳密に抽出してください。
                                         ページ番号はフッターにありますが、ページによって左右に分かれていますので注意して確認してください。
                                         また、表やグラフなどは、主要な内容やポイントを抽出してください。"""
                    )
                    parsed_documents = loader.load_data(uploaded_file.name)
                    
                    # メタデータ正規化
                    langchain_documents = []
                    for doc in parsed_documents:
                        page_content = doc.get_content()
                        
                        page_number_match = re.search(r'ページ番号:\s*(\d+)', page_content)
                        if not page_number_match:
                            page_number_match = re.search(r'\b\s*(\d+)\s*$', page_content[-200:])
                        
                        page_number = int(page_number_match.group(1)) if page_number_match else None
                        
                        metadata = {"source": uploaded_file.name}
                        if page_number:
                            metadata["page_number"] = page_number
                        
                        langchain_documents.append(Document(page_content=page_content, metadata=metadata))
                    
                    documents.extend(langchain_documents)
                
                # テキストをチャンクに分割
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.split_documents(documents)
                
                # ベクトル化とベクトルデータベースへの保存
                embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
                db = Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_db")
                db.persist()
            st.success("データベースの更新が完了しました！")
        else:
            st.warning("PDFファイルをアップロードしてください。")

# ---
# 4. チャットUIとロジック
# ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの入力を受け付ける
if query := st.chat_input("質問を入力してください..."):
    # ユーザーのメッセージをUIに表示
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # データベースがなければ警告
    if not Path("./chroma_db").exists():
        with st.chat_message("assistant"):
            st.warning("データベースがありません。PDFをアップロードしてから質問してください。")
    else:
        # LLMとRAGチェーンのセットアップ
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            """あなたはPDFの内容に基づいて質問に回答するAIアシスタントです。
            以下のコンテキスト情報と質問に基づいて、ユーザーに役立つように、専門家として丁寧かつ自然な日本語で回答してください。

            回答のルール:
            1. 質問がコンテキストに**関連しない**場合、**「検索しましたが、関連情報が見つかりませんでした。」**と回答してください。
            2. 回答は簡潔に、しかし必要な情報を網羅するようにしてください。
            3. 回答の根拠となる情報（ファイル名と該当ページ番号）を必ず明記してください。
            4. 情報を正確に伝えることを最優先とし、情報が不確かな場合は、その旨を正直に伝えてください。
            5. 回答は、適宜箇条書きや改行を行い、分かりやすい構成してください。

            コンテキスト:
            {context}

            質問:
            {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        db = get_db()
        retriever = db.as_retriever(search_kwargs={"k": 5})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # AIの回答を生成して表示
        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                try:
                    response = retrieval_chain.invoke({"input": query})
                    
                    full_response = response["answer"]
                    st.markdown(full_response)
                    
                    # 根拠となるドキュメント情報を表示
                    st.divider()
                    st.caption("以下より情報抽出しています。必要に応じて該当ファイル・ページを確認してください。")
                    for i, doc in enumerate(response["context"][:2]):
                        source = doc.metadata.get('source', '不明')
                        page = doc.metadata.get('page_number', '不明')
                        st.text(f"ファイル名: {source}, ページ番号: {page}")

                except Exception as e:
                    full_response = f"エラーが発生しました: {e}"
                    st.error(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
