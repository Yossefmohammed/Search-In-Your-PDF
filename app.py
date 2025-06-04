import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from constant import CHROMA_SETTINGS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource()
def load_llm():
    import torch

    # Local model path
    checkpoint = r"C:\Users\youssef\Desktop\search_PDF\LaMini-T5-738M"
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map='cpu',
        torch_dtype=torch.float32
    )

    # LaMini-T5 max token limit = 512
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True,  # <-- Important to avoid token overflow
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )

    return HuggingFacePipeline(pipeline=gen_pipeline)

@st.cache_resource()
def load_qa():
    llm = load_llm()

    embeddings = SentenceTransformerEmbeddings(model_name="all-miniLM-L6-v2")

    db = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_SETTINGS.persist_directory
    )

    retriever = db.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain

def process_answer(instruction):
    qa = load_qa()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

def main():
    st.title("🔍 Search Your PDF")
    
    with st.expander("About the App"):
        st.markdown("""This is a Generative AI-powered question-and-answering app that responds to questions about your PDF files.""")

    question = st.text_area("Enter Your Question")

    if st.button("Search"):
        if question.strip() == "":
            st.warning("Please enter a question before searching.")
        else:
            st.info("Your question: " + question)
            answer, metadata = process_answer(question)
            st.subheader("Answer:")
            st.write(answer)
            st.subheader("Metadata:")
            st.write(metadata)

if __name__ == "__main__":
    main()
