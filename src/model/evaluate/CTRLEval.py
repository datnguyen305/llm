import sys
import os

# Thêm đường dẫn đến CTRLEval folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'CTRLEval'))

from ctrleval import CTRLEval as OriginalCTRLEval

class TextEvaluator: 
    def __init__(self, task="senti", question="", generated_answer=""): 
        self.task = task
        self.question = question
        self.generated_answer = generated_answer

        self.prefix = [question]
        self.data = [question + '\n' + generated_answer]  # Sửa lại format data
        
        # Tạo đường dẫn đúng đến các file
        ctrleval_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'CTRLEval')
        
        self.scorer = OriginalCTRLEval(
            iwf_dir=os.path.join(ctrleval_dir, "iwf_full.txt"),
            prompt_dir=os.path.join(ctrleval_dir, f"prompt/prompt_{self.task}.txt"),
            verbal_dir=os.path.join(ctrleval_dir, f"prompt/verbal_{self.task}.txt"),
            model_name_or_path="google/pegasus-large",
            device="cpu"
        )

    def consistency(self):
        """Tính điểm consistency"""
        return self.scorer.score(aspect="cons", data=self.data, prefix=self.prefix)
    
    def fluency(self):
        """Tính điểm fluency"""
        return self.scorer.score(aspect="flu", data=self.data)
    
    def evaluate_all(self):
        """Đánh giá tất cả metrics"""
        return {
            'consistency': self.consistency(),
            'fluency': self.fluency()
        }
