import streamlit as st
import textwrap
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoTokenizer

# ======================
# 🔧 Caching Models and Clients
# ======================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("multi-qa-mpnet-base-dot-v1")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("google/flan-t5-base")

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    if "dvaita-index" not in pc.list_indexes().names():
        pc.create_index(
            name="dvaita-index",
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index("dvaita-index")

# ======================
# 📦 Load Resources
# ======================
embedder = load_embedder()
reranker = load_reranker()
generator = load_generator()
tokenizer = load_tokenizer()
index = init_pinecone()

# ======================
# 🔍 Retrieval + Rerank
# ======================
def retrieve_and_rerank(query, top_k=5, final_k=3):
    query_emb = embedder.encode([query]).tolist()[0]
    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    raw_chunks = [m["metadata"]["text"] for m in results["matches"]]

    # st.info(f"Retrieved {len(raw_chunks)} chunks from Pinecone")  # 👈 hidden debug

    reranker_max = getattr(reranker.tokenizer, "model_max_length", 512)
    query_len = len(reranker.tokenizer.encode(query, add_special_tokens=False))
    safe_chunks = []
    for chunk in raw_chunks:
        tokens = reranker.tokenizer.encode(chunk, add_special_tokens=False)
        allowed = reranker_max - query_len - 3
        if len(tokens) > allowed:
            chunk = reranker.tokenizer.decode(tokens[:allowed], skip_special_tokens=True)
            # st.warning(f"Truncated chunk to {allowed} tokens")  # 👈 hidden debug
        safe_chunks.append(chunk)

    pairs = [[query, c] for c in safe_chunks]
    scores = reranker.predict(pairs)
    reranked = [c for _, c in sorted(zip(scores, safe_chunks), reverse=True)]
    return reranked[:final_k]

# ======================
# ✂️ Truncate Context
# ======================
def build_truncated_context(chunks, question, max_model_tokens=512, max_output_tokens=200):
    max_input_tokens = max_model_tokens - max_output_tokens
    q_tokens = tokenizer.encode(question, add_special_tokens=False)
    instr = (
        "You are a knowledgeable teacher of Dvaita Vedānta. "
        "Explain the following question using the provided context.\n\n"
    )
    instr_tokens = tokenizer.encode(instr, add_special_tokens=False)
    sep_tokens = tokenizer.encode("\n\nQuestion:\n\nAnswer:\n", add_special_tokens=False)
    overhead = len(q_tokens) + len(instr_tokens) + len(sep_tokens) + 10
    available = max_input_tokens - overhead

    selected = []
    total = 0
    for ch in chunks:
        t = tokenizer.encode(ch, add_special_tokens=False)
        if total + len(t) <= available:
            selected.append(ch)
            total += len(t)
        else:
            remain = available - total
            if remain > 40:
                trunc = tokenizer.decode(t[:remain], skip_special_tokens=True)
                selected.append(trunc + "...")
                total += remain
            break

    context = "\n\n".join(selected)
    prompt = f"{instr}Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n"
    # st.info(f"Final prompt token count: {len(tokenizer.encode(prompt))}/{max_input_tokens}")  # 👈 hidden debug
    return context

# ======================
# 🧠 Generate Answer
# ======================
def generate_answer(question, context):
    instr = (
        "You are a revered teacher of Dvaita Vedānta, known for your clarity, depth, and ability to convey complex "
        "philosophical truths in a simple manner.\n\n"
        "Using the context below, answer the question thoroughly, as if teaching a sincere student. "
        "If the context lacks information, explain the concept based on traditional Dvaita philosophy.\n\n"
        "Your answer must follow this structure:\n"
        "1. A brief explanation of related concepts (e.g., guṇas, tattvas, moksha)\n"
        "2. A simple example or analogy if appropriate\n"
        "3. A gentle conclusion or reflection\n\n"
        "Using the following context, answer the question precisely, providing explanations, examples, and reflections.\n"
        "Avoid vague language. Be concise yet insightful. Maintain a calm, knowledgeable tone — "
        "like a traditional scholar guiding a student on their spiritual journey.\n\n")


    prompt = f"{instr}Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n"
    # st.info(f"Prompt length: {len(tokenizer.encode(prompt))}")  # 👈 hidden debug

    outputs = generator(
        prompt,
        max_new_tokens=256,
        #do_sample=True,
        top_p=0.9,                  # nucleus sampling to keep quality and diversity
        num_beams=3,  # 👈 improved beam width
        no_repeat_ngram_size=3,
        early_stopping=False 
    )
    return textwrap.fill(outputs[0]["generated_text"].strip(), width=100)

# ======================
# 🖼️ Streamlit Interface
# ======================
st.title("🧘 Gita-Tatvavāda Ask a Question on Bhagavad Gita")
st.markdown("Enter your question below and receive a detailed explanation grounded in the commentry of Shree Madhvacharya")

question = st.text_input("📩 Your Question", "")

if question:
    with st.spinner("🔍 Thinking..."):
        chunks = retrieve_and_rerank(question)
        context = build_truncated_context(chunks, question)
        answer = generate_answer(question, context)

    st.success("✅ Answer generated!")

    st.markdown("### 🧠 Answer")
    st.write(answer)

    st.markdown("### 📜 Context used in answer")
    for i, chunk in enumerate(chunks, start=1):
      st.markdown(f"""
           **Context {i}:**  
           {chunk.strip()}
           """)

