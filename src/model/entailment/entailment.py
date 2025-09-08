import pandas as pd
from sentence_transformers import SentenceTransformer, util


class EntailmentChecker:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", threshold=0.75):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def check_entailment(self, knowledge: str, answer: str) -> dict:
        """So sánh 1 cặp knowledge - answer"""
        emb_k = self.model.encode(knowledge, convert_to_tensor=True)
        emb_a = self.model.encode(answer, convert_to_tensor=True)

        sim = util.cos_sim(emb_k, emb_a).item()
        is_entailed = sim >= self.threshold

        return {
            "Knowledge": knowledge,
            "Generated Answer": answer,
            "Cosine Similarity": sim,
            "Entailed": is_entailed
        }

    def check_entailment_loop(self, knowledge_list, generated_answer_list, output_file="entailment_results.csv"):
        """So sánh nhiều cặp knowledge - answer"""
        data = []
        for k, a in zip(knowledge_list, generated_answer_list):
            result = self.check_entailment(k, a)
            data.append(result)

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        return df
