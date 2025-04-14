from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import ollama
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
TOKEN = '7423907422:AAGsjRh2b4MP-TjPGdONFkuGfEI0GnqmX6A'

# Function to generate responses using Ollama's Ally model
def query_llm(user_prompt):
    try:
        embedding_model = OllamaEmbeddings(model="llama3")
        vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

        retriever = vectorstore.as_retriever()
        llm = Ollama(model="llama3")

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True  # Enable to check if context was found
        )

        system_prompt = (
            "You are MunnaBhai MBBS — a witty but expert medical assistant specializing in gut health. "
            "Give short, crisp, and medically accurate answers with a touch of humor, but skip the fluff.\n\n"
            "User: " + user_prompt
        )


        result = qa_chain(system_prompt)

        # If no documents were retrieved, fallback to just the LLM
        if not result.get('source_documents'):
            fallback_response = llm(system_prompt)
            return fallback_response

        return result['result']

    except Exception as e:
        return f"Error: {e}"

# Start command handler
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    system_prompt = (
        "You are MunnaBhai MBBS — a witty but expert medical assistant specializing in gut health. "
        "Give a short and fun welcome message to a user who just started chatting with you. "
        "Mention that they can ask any medical question related to gut health."
    )

    response = llm(system_prompt)
    await update.message.reply_text(response)


# Message handler to generate LLM responses
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    response = query_llm(user_message)
    await update.message.reply_text(response)

def main():
    # Create the Application
    app = Application.builder().token(TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    print("Bot is running...")
    app.run_polling()

if __name__ == '__main__': 
    main()
