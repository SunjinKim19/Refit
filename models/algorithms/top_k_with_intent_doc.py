from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
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
    embedding = np.array(metadata.pop("embedding"), dtype=np.float32)  # ✅ embedding만 분리
    doc = Document(
        page_content=item["page_content"],
        metadata=metadata  # ✅ question, answer, intent는 유지
    )
    docs.append(doc)
    embeddings.append(embedding)

embeddings = np.array(embeddings)

# 4. FAISS 인덱스 생성 (from_embeddings 사용)
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
query = "결제 도중 시스템 오류가 났어요."

# 모델 초기화
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

# 상위 3개의 문서 선택
compressor = CrossEncoderReranker(model=model, top_n=3)

# 문서 압축 검색기 초기화
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(query)
# 8. 상위 3개 후보 문서 출력 (점수 없이)

print("\n🎯 [상위 3개 후보 출력]")
for i, doc in enumerate(compressed_docs[:3], 1):
    print(f"{i}. 질문: {doc.metadata.get('question', '없음')}")
    print(f"   ↳ 답변: {doc.metadata.get('answer', '없음')}")
    print(f"   🧭 고객의도: {doc.metadata.get('intent', '없음')}")
    print("-" * 50)
