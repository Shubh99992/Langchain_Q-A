import streamlit as st
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback
# with st.sidebar:
#     st.title('🤗💬 LLM Chat App')
#     st.markdown('''
#     ## About
#     This app is an LLM-powered chatbot built using:
#     - [Streamlit](https://streamlit.io/)
#     - [LangChain](https://python.langchain.com/)
#     - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
#     ''')
    
 
def main():
    st.set_page_config(page_title='Ask your PDF')
    st.title('💬 LLM Chat App')
    # set.header('Ask your PDF 💬')
    pdf=st.file_uploader('Upload your PDF',type='pdf')

    if pdf is not None:#Extract text from pdf
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()

        # st.write(text)

        text_splitter = CharacterTextSplitter( #split into small chunks
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
           length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.write(chunks)
        # embeddings = OpenAIEmbeddings()#create embeddings
        # knowledge_base=FAISS.from_texts(chunks,embeddings)#create document


        # user_question =  st.text_input('Ask me anything')#show input
        # if user_question:
        #     docs=knowledge_base.simmilarity_search(user_question)
        #     # st.write(docs)
        #     llm=OpenAI()
        #     chain=load_qa_chain(llm,chain_type='stuff')
        #     with get_openai_callback() as cb:
        #         response = chain.run(input_documents=docs,question=user_question)
        #         print(cb)
        #     st.write(response)
           


    


if __name__=='__main__':
    main()