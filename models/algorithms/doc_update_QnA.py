from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import json

# 모델 로드
model = SentenceTransformer('jhgan/ko-sbert-sts')

# 기존 docs JSON 파일 경로
existing_docs_path = 'C:\\Users\\효원\\Desktop\\my_project\\backend\\embeddings\\payment_docs_with_intent.json'

# 새로운 데이터 (예: 새로운 질문-답변-고객의도 리스트)
new_data = [
    {
        "질문": "결제한 금액이 왜 다르게 청구됐나요?",
        "답변": "결제 금액 오류에 대해 죄송합니다. 고객센터로 문의주시면 확인 후 조치해드리겠습니다.",
        "고객의도": "결제금액오류"
    },
    {
        "질문": "결제 후 승인 취소가 가능한가요?",
        "답변": "결제 승인 취소는 결제 상황에 따라 다르니 고객센터로 문의 바랍니다.",
        "고객의도": "결제승인취소문의"
    }
    # 여기에 더 추가 가능
]

# 1. 기존 docs 불러오기
with open(existing_docs_path, 'r', encoding='utf-8') as f:
    existing_docs = json.load(f)

# 2. 기존 docs를 Document 객체 리스트로 변환 (임베딩은 metadata에 있으므로 그대로 활용 가능)
docs = []
for d in existing_docs:
    doc = Document(
        page_content=d['page_content'],
        metadata=d['metadata']
    )
    docs.append(doc)

# 3. 새로운 데이터에 대해 임베딩 생성 및 docs에 추가
for item in new_data:
    question = item["질문"]
    intent = item.get("고객의도", "").strip() or "기타"
    answer = item.get("답변", "").strip()

    content = f"{question} [SEP] {intent}"
    embedding = model.encode(content).tolist()

    new_doc = Document(
        page_content=content,
        metadata={
            "question": question,
            "answer": answer,
            "intent": intent,
            "embedding": embedding
        }
    )
    docs.append(new_doc)

# 4. 다시 저장 가능한 dict 리스트로 변환
serializable_docs = []
for d in docs:
    serializable_docs.append({
        "page_content": d.page_content,
        "metadata": d.metadata
    })

# 5. JSON 파일 덮어쓰기
with open(existing_docs_path, 'w', encoding='utf-8') as f:
    json.dump(serializable_docs, f, ensure_ascii=False, indent=4)

print("기존 docs에 새로운 질문-답변-의도 임베딩 업데이트 완료!")
