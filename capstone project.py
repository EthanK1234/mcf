import openai
import fitz
import os
from annoy import AnnoyIndex
import numpy as np

# initiate the openai key and Annoy
openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings_index = AnnoyIndex(1536, 'angular') # size of the gpt 3.5 embeddings

MAX_TOKENS = 8192 # maximum number of tokens for the model

# extract text from pdf
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# generates the embeddings form the given text using openai
def generate_embeddings(text):
    if len(text) > MAX_TOKENS:
        text = text[:MAX_TOKENS] # truncate text to fit within token limits
    response = openai.Embedding.create(input=text, engine = "text-embedding-ada-002")
    return response['data'][0]['embedding']

# builds an annoy vector store for the extracted pdf texts
def build_vector_store(file_paths):
    for i, file_path in enumerate(file_paths):
       text = extract_text_from_pdf(file_path)
       embedding = generate_embeddings(text)
       embeddings_index.add_item(i, embedding)
    embeddings_index.build(10) # 10 tress
    embeddings_index.save("text_embeddings.ann")


# loads the vector store
def load_vector_store():
    embeddings_index.load("text_embeddings.ann")

# search the vector store for documents similar to the query
def search_documents(query, k=1):
    query_embedding = generate_embeddings(query)
    nearest_ids = embeddings_index.get_nns_by_vector(query_embedding, k) # retrive the top 5 similar items
    return nearest_ids


# paths to the pdfs
pdf_files = ['/content/15DayExpressTerms.pdf',
             '/content/808677.pdf',
             '/content/AV_Express_Terms.pdf',
             '/content/AV_Regulations_Summary (1).pdf',
             '/content/AV_Regulations_Summary-1.pdf',
             '/content/AV_Regulations_Summary.pdf',
             '/content/AV_Second15Day_Notice_Express_Terms.pdf',
             '/content/CDL_testing_learners_stds.pdf',
             '/content/Commercial_dl_DocsList.pdf',
             '/content/DL-1010-D-English-R12-2022-WWW.pdf',
             '/content/DL_1010e_English.pdf',
             '/content/DMV Commerical Request.pdf',
             '/content/DMV Recreational Handbook.pdf',
             '/content/DMV commerical.pdf',
             '/content/DMV motorcycle.pdf',
             '/content/FederalCompliantInfographic.pdf',
             '/content/INF-1128-R9-2006-AS-WWW.pdf',
             '/content/SOFIT-Policy-Papers_C_OpenDataOpenGovernment_PR_v2.pdf',
             '/content/avexpressterms_31017.pdf',
             '/content/ca-drivers-handbook.pdf',
             '/content/disc_guide.pdf',
             '/content/eir_cardspecifications.pdf',
             '/content/iid_pilot_program_q-a.pdf',
             '/content/laps_table.pdf',
             '/content/mc522i.pdf',
             '/content/reg360.pdf',
             '/content/reg361.pdf',
             '/content/reg490b (1).pdf',
             '/content/reg490b.pdf',
             '/content/residency_docslist.pdf',
             '/content/sr104.pdf']


# build the vector store with embeddings
build_vector_store(pdf_files)

# load vector store
load_vector_store()

# example query
query = "How to apply for a driver's license?"
document_ids = search_documents(query)
print("Top documents for query:", document_ids)

# generate answer
def generate_answer(documents_ids):
    context = ' '.join([extract_text_from_pdf(pdf_files[i]) for i in document_ids])
    # truncate the contect to fit within the token limit
    max_context_length = 4097 - len(query) - 200 # reserve space for query  and completion
    if len(context) > max_context_length :
        context = context[:max_context_length]
    prompt = f"Based on the following documents : {context}\n\nAnswer the query :{query}"
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

print(generate_answer(document_ids))


