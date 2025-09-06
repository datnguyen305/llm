from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch



class NLI: 
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
        self.model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")

    def predict(self, premise, hypothesis):
        # Encode dữ liệu
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt")
        
        # Dự đoán
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        # Nhãn của MNLI: 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
        labels = ["INTRINSIC", "EXTRINSIC", "NO"]
        pred = labels[torch.argmax(probs)]
        
        return pred, probs