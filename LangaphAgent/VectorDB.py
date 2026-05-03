# # %%
# %pip install chromadb langchain langchain-community

# # %%
# %pip install langchain-groq

# # %%
# %pip install -U langchain-ollama

# # %%
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Any

llm = ChatGroq(
    model="llama-3.3-70b-versatile",api_key="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"
)
# print(llm.invoke("Explain vector databases"))

# # %%
# %pip install sentence-transformers

# # %%
# # from langchain_ollama import OllamaEmbeddings

# # embedding = OllamaEmbeddings(
# #     model="nomic-embed-text"
# # )


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# # %%
employees = [
    {
        "id": 1,
        "name": "Anil Kumar",
        "role": "Software Engineer",
        "skills": ["Python", "AI", "LangChain"],
        "location": "Hyderabad"
    },
    {
        "id": 2,
        "name": "Ravi Teja",
        "role": "Data Scientist",
        "skills": ["ML", "NLP", "Pandas"],
        "location": "Bangalore"
    },
    {
        "id": 3,
        "name": "Sneha Reddy",
        "role": "DevOps Engineer",
        "skills": ["AWS", "Docker", "Kubernetes"],
        "location": "Chennai"
    }
]

# # %%
docs = []

# for emp in employees:
#     text = f"""
#     Employee ID: {emp['id']}
#     Name: {emp['name']}
#     Role: {emp['role']}
#     Skills: {', '.join(emp['skills'])}
#     Location: {emp['location']}
#     """
    
#     docs.append(Document(
#         page_content=text,
#         metadata={"id": emp["id"], "name": emp["name"]}
#     ))
for emp in employees:
    text = f"""
    Employee ID: {emp['id']}
    Name: {emp['name']}
    Role: {emp['role']}
    Skills: {', '.join(emp['skills'])}
    Location: {emp['location']}
    """

    docs.append(
        Document(
            page_content=text,
            metadata={
                "id": emp["id"],
                "name": emp["name"],
                "role": emp["role"],
                "location": emp["location"],   # ✅ REQUIRED
                "skills": emp["skills"]        # ✅ useful for future filters
            }
        )
    )

# # %%


vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./employee_db"
)

# # %%
# vectorstore.persist()

# # %%
# results = vectorstore.similarity_search("Who knows AI?")

# for r in results:
#     print(r.page_content)

# # %%
# def search_employee(query: str):
#     results = vectorstore.similarity_search_with_score(query, k=5)

#     best = []
#     for doc, score in results:
#         print(f"Score: {score}")  # debug
#         print(doc)
#         if score < 1.4:
#             best.append(doc.page_content)

#     return "\n\n".join(best) if best else "No relevant employees found"


def normalize_score(distance: float, metric: str = "cosine"):
    if metric == "cosine":
        return 1 - distance
    elif metric == "l2":
        return 1 / (1 + distance)   # safe normalization
    else:
        return distance  # fallback
    

def search_employees_tool(
    query: str,
    top_k: int = 5,
    filters: Dict[str, Any] = None
) -> Dict[str, Any]:

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k,
        filter=filters
    )

    formatted_results = []

    for doc, distance in results:
        metadata = doc.metadata or {}

        similarity = normalize_score(distance, metric="cosine")

        formatted_results.append({
            "employee_id": metadata.get("employee_id"),
            "name": metadata.get("name"),
            "department": metadata.get("department"),
            "role": metadata.get("role"),
            "score": round(similarity, 4),
            "content": doc.page_content,
            "source": "chroma"
        })

        MIN_SCORE = 0.6  # tune this

    filtered_results = [
        r for r in formatted_results if r["score"] >= MIN_SCORE
    ]
    if not filtered_results:
        return {
            "results": [],
            "message": "No relevant employees found"
        }
    else:
        return {
            "results": formatted_results
        }


# # %%
# query={
#   "query": "string",
#   "top_k": 5,
#   "filters": {
#     "department": "Engineering"
#   }
# }

response = search_employees_tool(
        query="python",
        top_k=5,
        filters={"location": "Hyderabad"})

print (response)


