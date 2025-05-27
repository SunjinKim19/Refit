from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import json
import numpy as np
import os

# 1. 임베딩 모델 설정
embedding_model_name = "jhgan/ko-sbert-sts"
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

category = input("카테고리를 입력해 주세요\nAS, business, change, order, payment, return, shipping\n: ")
print()

# 2. 저장된 JSON 파일 불러오기
with open(f'backend/embeddings/{category}_docs_with_intent.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 3. Document 객체 재생성 및 임베딩 벡터 분리
docs = []
embeddings = []

for item in data:
    metadata = item["metadata"].copy()
    embedding = np.array(metadata.pop("embedding"), dtype=np.float32)
    doc = Document(
        page_content=item["page_content"],
        metadata=metadata
    )
    docs.append(doc)
    embeddings.append(embedding)

embeddings = np.array(embeddings)

# 4. FAISS 인덱스 생성 또는 불러오기
save_path = f"backend/embeddings/{category}_faiss_index"

if os.path.exists(save_path):
    print("FAISS 인덱스가 존재합니다. 불러오는 중...")
    vectorstore = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
else:
    print("FAISS 인덱스가 없습니다. 새로 생성 중...")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print("FAISS 인덱스 저장 완료!")

# 5. Retriever 생성 (k=10)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 6. 질의어 입력
query = "포장 어떻게 해서 보내야 하나요?"

# 7. 후보 문서 가져오기
candidate_docs = retriever.invoke(query)

# 8. CrossEncoder 로 유사도 직접 계산
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
pairs = [(query, doc.page_content) for doc in candidate_docs]
scores = cross_encoder.score(pairs)

# 9. 점수를 문서에 붙이고 정렬
scored_docs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)

# 10. 상위 3개 문서 출력
print("\n🎯 [상위 3개 후보 출력 - 유사도 포함]")
for i, (doc, score) in enumerate(scored_docs[:3], 1):
    print(f"{i}. 질문: {doc.metadata.get('question', '없음')}")
    print(f"   ↳ 답변: {doc.metadata.get('answer', '없음')}")
    print(f"   🧭 고객의도: {doc.metadata.get('intent', '없음')}")
    print(f"   📊 유사도 점수: {score:.4f}")
    print("-" * 50)

