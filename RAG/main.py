from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
import os

from langchain_core.prompts import ChatPromptTemplate


def get_chat_model():
    os.environ["OPENAI_API_KEY"] = "None"
    os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
    # llm_completion = OpenAI(model_name="Qwen2.5-14B")
    llm_chat = ChatOpenAI(model_name="Qwen2.5-14B")

    return llm_chat


def get_embedding():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return embedding


def get_db():
    db = Milvus(embedding_function=get_embedding(),
                collection_name="arXiv",
                connection_args={"uri": "http://10.58.0.2:19530"}
                )
    return db


def fix_user_input(llm, query):
    # prompt = "请你帮我更正以下查询，确保没有拼写错误、笔误或表述错误等。" \
    #          "请根据你已有的知识，注意你一定要完全理解查询中每个词的意思，都有可能出错，一定要结合你自己已有的知识将用户输入中存在错误的地方更正，特别关注计算机领域的术语和表达方式。" \
    #          "请直接输出且只输出更正后的查询，不要输出任何其他多余文字："
    prompt = "请检查并更正以下查询，确保没有拼写错误、笔误或表述错误。" \
             "特别注意计算机领域的术语和表达方式，确保每个词语的准确性，并结合你现有的知识进行纠正。" \
             "请直接输出且只输出更正后的查询，不要输出任何其他多余文字："
    fixed_query = llm.invoke(prompt + query).content
    print(fixed_query)
    return fixed_query


def optimize_user_input(llm, query):
    prompt = "请你帮我润色该查询，优化表达方式，使其更清晰、更能帮助找到相关的文档或信息。" \
             "你需要根据用户的查询，重新组织句子，并提炼出重要的关键词，保证查询更具针对性。" \
             "请注意用户的输入可能存在拼写错误、笔误、表述错误等问题，你应该根据你已掌握的知识自行更正之。" \
             "请直接告诉我优化后的查询，不要输出任何其他多余文字："
    optimized_query = llm.invoke(prompt + query).content
    print(optimized_query)
    return optimized_query


def translate_user_input(llm, query):
    prompt = '将以下中文句子翻译成英文，确保翻译准确，特别关注计算机领域的术语和表达方式。' \
             '请使用正确的专业术语，并确保英文翻译符合技术文献或文档的常用表达。' \
             '请直接告诉我翻译后的结果，不要输出任何其他多余文字：'
    translated_query = llm.invoke(prompt + query).content
    print(translated_query)
    return translated_query


if __name__ == '__main__':
    # os.putenv("TOKENIZERS_PARALLELISM", 'False')
    llm = get_chat_model()

    db = get_db()
    retriever = db.as_retriever(search_kwargs={"k": 5})
    # retriever = db.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={'k': 5, 'fetch_k': 50}
    # )

    # 创建带有 system 消息的模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """你是一个只能用中文回答用户问题的助手，请严格按照下面的已知信息并回答问题，不要提其他东西。
                   请记住你的问答中绝对不能出现'已知信息'这四个字，也就是说你绝对不能向用户透露出'已知信息'的存在，要把'已知信息'当作你自己的知识。
                   如果你不知道答案就说你不知道，不要说已知信息中没有，也不要提及你知道的其他信息，请简洁明了地说你不知道即可，也不要试图编造答案。
                   请记住无论什么情况下，你都应该且只能用中文回答问题。
                   已知信息: {context} """),
        ("user", "{question}")
    ])

    # 自定义的提示词参数
    chain_type_kwargs = {
        "prompt": prompt_template,
    }

    # 定义RetrievalQA链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 使用stuff模式将上下文拼接到提示词中
        chain_type_kwargs=chain_type_kwargs,
        retriever=retriever,
        return_source_documents=True
    )

    while True:
        question = input("请输入：")
        question = fix_user_input(llm, question)
        question = optimize_user_input(llm, question)
        question = translate_user_input(llm, question)
        answer = qa_chain.invoke(question)
        print(answer['result'])
