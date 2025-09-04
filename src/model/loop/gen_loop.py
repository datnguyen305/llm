from src.model.text_gen import TextGenerator
from src.model.evaluate.CTRLEval import TextEvaluator

def text_gen_loop(context, question):
    text_generator = TextGenerator(context, question, threshold_cons=-1.5, max_retries=3)

    def evaluator_func(question, answer):
        """
        Wrapper function để đánh giá consistency
        Args:
            question (str): Câu hỏi
            answer (str): Câu trả lời
            
        Returns:
            float: Consistency score
        """
        evaluator = TextEvaluator(task="senti", question=question, generated_answer=answer)
        score_list = evaluator.consistency()
        return float(score_list[0]) if isinstance(score_list, list) else float(score_list)
    
    response, score, retries = text_generator.generate_with_quality_check(evaluator_func=evaluator_func)

    return response, score, retries