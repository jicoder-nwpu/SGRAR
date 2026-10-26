import json
import pickle
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# ====================== 配置 ======================
INPUT_JSONL = "./ESC-database-processed.jsonl"
FAISS_INDEX_PATH = "./ESC-database-faiss.index"
META_PATH = "./ESC-database-meta.pkl"
MODEL_NAME = "./Qwen-3B"
DEVICE = "cuda"

# -------- VAD情绪三维得分 替换旧EMOTION_NEIGHBOR --------
EMOTION_VAD_TARGET = {
    "surprised": {"vad": [0.568, 0.71, 0.078], "mean": 0.5107},
    "excited": {"vad": [0.816, 0.862, 0.418], "mean": 0.4380},
    "angry": {"vad": [-0.756, 0.66, 0.208], "mean": 0.7013},
    "proud": {"vad": [0.812, 0.4, 0.746], "mean": 0.3070},
    "sad": {"vad": [-0.55, -0.334, -0.702], "mean": 0.6530},
    "annoyed": {"vad": [-0.792, 0.566, -0.31], "mean": 0.7780},
    "grateful": {"vad": [0.916, -0.294, 0.12], "mean": 0.2783},
    "lonely": {"vad": [-0.5, -0.548, -0.524], "mean": 0.5793},
    "afraid": {"vad": [-0.977, 0.352, -0.529], "mean": 0.8097},
    "terrified": {"vad": [-0.82, 0.804, -0.226], "mean": 0.8083},
    "guilty": {"vad": [-0.73, 0.54, -0.296], "mean": 0.7610},
    "impressed": {"vad": [0.612, 0.54, -0.142], "mean": 0.5117},
    "disgusted": {"vad": [-0.898, 0.546, -0.452], "mean": 0.8160},
    "hopeful": {"vad": [0.894, -0.286, 0.254], "mean": 0.2610},
    "confident": {"vad": [0.53, -0.352, 0.446], "mean": 0.2787},
    "furious": {"vad": [-0.876, 0.906, 0.196], "mean": 0.7643},
    "anxious": {"vad": [-0.438, 0.75, -0.132], "mean": 0.7200},
    "anticipating": {"vad": [0.244, -0.062, 0.334], "mean": 0.3933},
    "joyful": {"vad": [0.99, 0.24, 0.296], "mean": 0.3257},
    "nostalgic": {"vad": [-0.084, -0.298, -0.632], "mean": 0.5697},
    "disappointed": {"vad": [-0.858, -0.056, -0.518], "mean": 0.7200},
    "prepared": {"vad": [0.42, -0.04, 0.466], "mean": 0.3457},
    "jealous": {"vad": [-0.654, 0.71, -0.31], "mean": 0.7790},
    "content": {"vad": [0.528, -0.408, 0.118], "mean": 0.3243},
    "devastated": {"vad": [-0.76, 0.088, -0.472], "mean": 0.7200},
    "embarrassed": {"vad": [-0.632, 0.12, -0.47], "mean": 0.7037},
    "caring": {"vad": [0.27, -0.062, 0.0], "mean": 0.4447},
    "sentimental": {"vad": [0.166, -0.244, -0.376], "mean": 0.4943},
    "trusting": {"vad": [0.714, 0.016, 0.5], "mean": 0.3003},
    "ashamed": {"vad": [-0.688, 0.176, -0.544], "mean": 0.7347},
    "apprehensive": {"vad": [-0.188, 0.184, -0.138], "mean": 0.5850},
    "faithful": {"vad": [0.776, -0.334, 0.254], "mean": 0.2727},
    "guilt": {"vad": [-0.584, 0.134, -0.588], "mean": 0.6843},
    "anger": {"vad": [-0.666, 0.73, 0.314], "mean": 0.6600},
    "nervousness": {"vad": [-0.674, 0.83, -0.518], "mean": 0.7903},
    "fear": {"vad": [-0.854, 0.68, -0.414], "mean": 0.8080},
    "pain": {"vad": [-0.918, 0.53, -0.351], "mean": 0.8015},
    "anxiety": {"vad": [-0.708, 0.73, -0.316], "mean": 0.7623},
    "disgust": {"vad": [-0.896, 0.55, -0.366], "mean": 0.8020},
    "jealousy": {"vad": [-0.708, 0.566, -0.038], "mean": 0.7187},
    "depression": {"vad": [-0.938, 0.04, -0.536], "mean": 0.7423},
    "sadness": {"vad": [-0.896, -0.424, -0.672], "mean": 0.6840},
    "shame": {"vad": [-0.88, 0.34, -0.69], "mean": 0.7850}
}

STRATEGY_LEVEL_MAP = {
    "gratitude prompting": 1,
    "restatement or paraphrasing": 1,
    "others": 1,
    "information": 1,
    "question":1,
    "neutral validation": 2,
    "positive reinforcement": 2,
    "self-disclosure": 2,
    "exploring actions and intentions": 2,
    "affirmation and reassurance": 2,
    "reflection of feelings": 3,
    "cognitive reframing": 3,
    "exploring feelings and emotions": 3,
    "exploring thoughts and cognition": 3,
    "providing suggestions": 3
}

W_SEMANTIC = 0.45
W_EMOTION = 0.35
W_STRATEGY = 0.2

# ===================== Qwen3‑0.6B‑Base Mean‑Pool Embedding =====================
class Qwen3Embedding:
    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            output_hidden_states=True
        ).to(device).eval()
    @torch.no_grad()
    def encode(self, texts, batch_size=16, normalize_embeddings=True):
        all_embeds = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start+batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            ).to(self.device)
            out = self.model(**enc)
            last_hidden = out.hidden_states[-1]
            attn_mask = enc["attention_mask"].unsqueeze(-1)
            sum_hidden = torch.sum(last_hidden * attn_mask, dim=1)
            lengths = torch.sum(attn_mask, dim=1)
            emb = sum_hidden / lengths
            if normalize_embeddings:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embeds.append(emb.cpu().numpy())
        return np.vstack(all_embeds)

# ====================== 打分函数（calc_emotion_score已替换为VAD余弦） ======================
def calc_emotion_score(query_emotion: str, candidate_emotion: str) -> float:
    """
    使用VAD三维向量余弦相似度计算情绪匹配分数
    cos_sim ∈ [-1,1] → 映射到 [0,1]
    """
    q = query_emotion.lower().strip()
    c = candidate_emotion.lower().strip()
    if q not in EMOTION_VAD_TARGET or c not in EMOTION_VAD_TARGET:
        return 0.0
    v_q = np.array(EMOTION_VAD_TARGET[q]["vad"], dtype=np.float32)
    v_c = np.array(EMOTION_VAD_TARGET[c]["vad"], dtype=np.float32)
    # 余弦相似度
    dot = np.dot(v_q, v_c)
    norm_q = np.linalg.norm(v_q)
    norm_c = np.linalg.norm(v_c)
    if norm_q < 1e-8 or norm_c <1e-8:
        return 0.0
    cos_sim = dot/(norm_q * norm_c)
    # [-1,1] -> [0,1]
    score = (cos_sim + 1.0) / 2.0
    return float(np.clip(score, 0.0, 1.0))

def get_strategy_level(s: str):
    return STRATEGY_LEVEL_MAP.get(s.strip().lower(), None)

def calc_strategy_score(query_strategy: str, cand_strategy: str) -> float:
    qs = query_strategy.lower().strip()
    cs = cand_strategy.lower().strip()
    qlv = get_strategy_level(qs)
    clv = get_strategy_level(cs)
    if qlv is None or clv is None:
        return 0.0
    if qs == cs:
        return 1.0
    diff = abs(qlv - clv)
    if diff == 0:
        return 0.5
    elif diff == 1:
        return 0.2
    else:
        return 0.0

def calc_total_score(
    faiss_distance: float,
    q_emotion: str,
    c_emotion: str,
    q_strategy: str,
    c_strategy: str,
):
    # 前提：索引为 IndexFlatL2，库向量与查询向量均已 L2 归一化
    squared_l2 = float(faiss_distance)
    if not np.isfinite(squared_l2):
        raise ValueError("Non-finite FAISS distance")

    cosine = float(np.clip(1.0 - squared_l2 / 2.0, -1.0, 1.0))
    semantic_score = (cosine + 1.0) / 2.0

    emotion_score = calc_emotion_score(q_emotion, c_emotion)
    strategy_score = calc_strategy_score(q_strategy, c_strategy)

    total_score = (
        W_SEMANTIC * semantic_score
        + W_EMOTION * emotion_score
        + W_STRATEGY * strategy_score
    )

    # 排序前不四舍五入
    return {
        "faiss_squared_l2": squared_l2,
        "semantic_cosine": cosine,
        "semantic_score": semantic_score,
        "emotion_score": emotion_score,
        "strategy_score": strategy_score,
        "total_score": total_score,
    }

# ====================== 构建FAISS索引 ======================
def build_faiss_db(jsonl_path, embedder: Qwen3Embedding):
    meta_list = []
    text_list = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            doc_text = f"{sample['context']}\n{sample['action']}"
            text_list.append(doc_text)
            example = (
                f"Context: {sample.get('abstract', '').strip()} "
                f"Emotion: {sample.get('emotion', '').strip()}. "
                f"Action: {sample.get('action', '').strip()} "
                f"Answer: {sample.get('answer', '').strip()}"
            )

            meta_list.append({
                "example": example,
                "emotion": sample["emotion"],
                "strategy": sample["strategy"],
                "answer": sample["answer"],
                "hash_id": sample["hash_id"]
            })
    print(f"embedding total samples: {len(text_list)}")
    embeddings = embedder.encode(text_list, batch_size=16, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta_list, f)
    print(f"saved index:{FAISS_INDEX_PATH}, meta:{META_PATH}")
    return index, meta_list

# ====================== 检索+Rerank ======================
def load_faiss_resources():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta_list = pickle.load(f)
    return index, meta_list

def faiss_retrieve_rerank(
    query_context: str,
    query_action: str,
    query_emotion: str,
    query_strategy: str,
    embedder: Qwen3Embedding,
    coarse_top=50,
    top_k_final=5
):
    index, meta_list = load_faiss_resources()
    query_text = f"{query_context}\n{query_action}"
    q_emb = embedder.encode([query_text], normalize_embeddings=True)
    distances, indices = index.search(q_emb, coarse_top)
    raw_result = []
    for l2_dist, idx in zip(distances[0], indices[0]):
        if idx >= len(meta_list):
            continue
        meta = meta_list[idx]
        score_info = calc_total_score(
            faiss_distance=l2_dist,
            q_emotion=query_emotion,
            c_emotion=meta["emotion"],
            q_strategy=query_strategy,
            c_strategy=meta["strategy"]
        )
        raw_result.append({"meta": meta, "example": meta["example"], **score_info})
    raw_result.sort(key=lambda x:x["total_score"], reverse=True)
    return raw_result[:top_k_final]

if __name__ == "__main__":
    embedder = Qwen3Embedding(MODEL_NAME, DEVICE)
    # ========= 构建索引只跑一次，跑完注释 =========
    # build_faiss_db(INPUT_JSONL, embedder)
    
    # ========== 样例7：grateful 感恩，Gratitude Prompting 感恩引导 ==========
    candidates = faiss_retrieve_rerank(
        query_context="speaker: Yes of course I didn't hear it go off, my apologies. I really appreciate that, it really is helpful knowing that I'm not the only one who feels this way I really know that that's what I need to do, and I am trying my best to get over my fears and just do it ya know. listener: What level of alcohol consumption do you have? speaker: no matter how hard it seems, maybe it wont be when I actual do it. listener: Trust me everyone will be very sympathetic :).  Everyone there will be in the same place. speaker: and quite a bit, about half a gallon or more a day . And thank you, I really do appreciate your help. I know that and I know its time for me to better my life because its no way to live ya know. Id like to grow up and have a normal life (:. listener: I am not sure how much half a gallon is to you ... I am in the UK and I know it differs between the US and UK :). speaker: thank you, i do feel better about going now. I just dont know what to expect and it can be scary sometimes.",
        query_action="Information",
        query_emotion="shame",
        query_strategy="Information",
        embedder=embedder,
        coarse_top=50,
        top_k_final=5
    )

    for i, item in enumerate(candidates):
        print(f"\n--- candidate {i+1} total_score:{item['total_score']} ---")
        print(f"semantic:{item['semantic_score']}, emo:{item['emotion_score']}, strategy:{item['strategy_score']}")
        print(item["example"])
