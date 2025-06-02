import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from constant import CHROMA_SETTINGS

def main():
    all_documents = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Loading: {file}")
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Use persist_directory (do NOT pass client here)
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_SETTINGS.persist_directory
    )

    print("✅ Ingestion completed and saved successfully.")

if __name__ == "__main__":
    main()
