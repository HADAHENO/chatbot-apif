from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

# إعداد الموديلات
SentenceTransformer_model = SentenceTransformer("intfloat/multilingual-e5-large")

# Pinecone - إنشاء كائن جديد
pc = Pinecone(api_key="pcsk_3ax4D8_PH7vWF1KWAMRpyjmEnXhwxswmHSjvqgwovna3xGGbfsgZsMRtRyFi9uCpPyi4B9")
index = pc.Index("quickstart")

# Gemini
genai.configure(api_key="AIzaSyAThgYMlxd1FFg67pMnnBtxBbgDPk7qIkQ")
model = genai.GenerativeModel("gemini-2.0-flash")

# Flask app
app = Flask(__name__)
chat_history = []

# البحث في قاعدة البيانات
def get_answer_from_pinecone(user_question):
    question_vector = SentenceTransformer_model.encode(user_question).tolist()
    search_result = index.query(vector=question_vector, top_k=3, include_metadata=True)
    best_match = max(search_result.matches, key=lambda x: x.score, default=None)

    if best_match and best_match.score > 0.7:
        return best_match.metadata.get('answer', 'لا توجد إجابة متوفرة')
    else:
        return "لا توجد إجابة دقيقة في قاعدة البيانات."

# التواصل مع Gemini
def ask_gemini_with_combined_answer(user_question, pinecone_answer, history=[]):
    context = "\n".join([f"👤 المستخدم: {q}\n🤖 الرد: {a}" for q, a in history])
    prompt = f"""
❗ هام: لا تستخدم إلا المعلومات الموجودة في قاعدة البيانات فقط. لا تُخمِّن.

📜 المحادثة السابقة:
{context if context else 'لا يوجد سياق سابق.'}

👤 المستخدم يسأل: {user_question}
📚 المعلومة المستخرجة من قاعدة البيانات: "{pinecone_answer or 'لا توجد'}"

✅ المطلوب: 
- إذا كان السؤال الحالي مرتبطًا بالمحادثة السابقة، استخدم السياق والإجابة المقتبسة من قاعدة البيانات معًا.
- إذا لم يكن مرتبطًا، تجاهل السياق السابق تمامًا.
- ❌ لا تضف أي معلومة من خارج قاعدة البيانات حتى لو كنت تعرفها.
- ✅ أجب بجملة واحدة واضحة.

📌 الرد:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# Endpoint
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    user_q = data.get("question", "")
    if not user_q:
        return jsonify({"error": "لم يتم توفير سؤال"}), 400

    pinecone_answer = get_answer_from_pinecone(user_q)
    final_answer = ask_gemini_with_combined_answer(user_q, pinecone_answer, chat_history)
    chat_history.append((user_q, final_answer))

    return jsonify({"answer": final_answer})

# ملحوظة: لا نستخدم app.run() في Hugging Face Spaces
