from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import json

# 모델 로드
model = SentenceTransformer('jhgan/ko-sbert-sts')

category = "payment"

# 예시 데이터셋 로드
with open(f"C:\\Users\\효원\\Desktop\\my_project\\backend\\data\\{category}_cleaned_with_intent.json", "r", encoding="utf-8") as f:
    cleaned_data = json.load(f)

# Document 리스트
docs = []

for item in cleaned_data:
    question = item["질문"]
    intent = item.get("고객의도", "").strip() or "기타"
    answer = item.get("답변", "").strip()

    content = f"{question} [SEP] {intent}"
    embedding = model.encode(content).tolist()

    doc = Document(
        page_content=content,
        metadata={
            "question": question,
            "answer": answer,
            "intent": intent,
            "embedding": embedding  # 메타데이터로 임베딩 저장 (필요에 따라)
        }
    )
    docs.append(doc)

# Document 객체는 JSON 직렬화가 안 되니까, 저장 가능한 형태로 변환
serializable_docs = []
for d in docs:
    serializable_docs.append({
        "page_content": d.page_content,
        "metadata": d.metadata
    })

# JSON 저장
with open('C:\\Users\\효원\\Desktop\\my_project\\backend\\embeddings\\{category}_docs_with_intent.json', 'w', encoding='utf-8') as f:
    json.dump(serializable_docs, f, ensure_ascii=False, indent=4)

print("Document 형식으로 임베딩 포함 JSON 저장 완료!")
