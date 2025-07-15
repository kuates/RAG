import chainlit as cl

from rag_langchain import rag_chain  # Import your pre-defined RAG chain

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="📚 Welcome! Ask me something based on the documents.").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Invoke the RAG chain with the user's question
    result = rag_chain.invoke({"question": message.content})

    # Just send the final answer — no retrieved context
    await cl.Message(content=result["answer"]).send()