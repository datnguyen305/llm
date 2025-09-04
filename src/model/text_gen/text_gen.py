from transformers import AutoTokenizer, AutoModelForCausalLM

class TextGenerator:
    def __init__(self, context, question, threshold_cons=-2, max_retries=3):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        self.context = context
        self.question = question
        self.THRESHOLD_CONS = threshold_cons  # Ngưỡng consistency
        self.MAX_RETRIES = max_retries        # Số lần thử lại tối đa
        self.response = None
        self.consistency_score = None

    def _create_messages(self, is_regenerate=False, previous_score=None):
        """Tạo messages cho chat template"""
        system_message = "Bạn là một trợ lý hữu ích. Trả lời câu hỏi của người dùng bằng tiếng Việt, chỉ dựa trên thông tin trong phần Context được cung cấp."
        
        if is_regenerate and previous_score is not None:
            user_content = f"""Ngữ cảnh: {self.context}
Câu hỏi: {self.question}

Lần trả lời trước có điểm consistency thấp ({previous_score:.4f} < {self.THRESHOLD_CONS}). 
Hãy trả lời lại một cách chính xác và nhất quán hơn, dựa chặt chẽ vào thông tin trong ngữ cảnh."""
        else:
            user_content = f"Ngữ cảnh: {self.context}\nCâu hỏi: {self.question}\nHãy trả lời câu hỏi dựa trên ngữ cảnh"

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

    def _generate_text(self, messages):
        """Sinh text từ messages"""
        # Tạo input từ template chat
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2042,         # Giảm xuống để tránh quá dài
            do_sample=True,            
            top_p=0.9,                 
            temperature=0.7            
        )

        # Lấy phần text mới sinh
        generated_answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return generated_answer.strip()

    def generate_response(self):
        """Sinh response lần đầu tiên"""
        messages = self._create_messages(is_regenerate=False)
        self.response = self._generate_text(messages)
        return self.response

    def regenerate_response(self, consistency_score):
        """
        Regenerate response nếu consistency score thấp
        
        Args:
            consistency_score (float): Điểm consistency hiện tại
            
        Returns:
            str: Response mới được sinh ra
        """
        self.consistency_score = consistency_score
        
        # Kiểm tra nếu điểm đã đủ tốt
        if consistency_score >= self.THRESHOLD_CONS:
            print(f"✅ Consistency score ({consistency_score:.4f}) đã đạt yêu cầu!")
            return self.response
        
        print(f"🔄 Regenerating... (Score: {consistency_score:.4f} < {self.THRESHOLD_CONS})")
        
        # Tạo messages mới cho regeneration
        messages = self._create_messages(is_regenerate=True, previous_score=consistency_score)
        new_response = self._generate_text(messages)
        
        self.response = new_response
        return new_response

    def generate_with_quality_check(self, evaluator_func=None):
        """
        Sinh response với kiểm tra chất lượng tự động
        
        Args:
            evaluator_func: Function để đánh giá consistency score
                          Signature: func(question, answer) -> float
                          
        Returns:
            tuple: (final_response, final_score, num_retries)
        """
        if evaluator_func is None:
            # Nếu không có evaluator, chỉ sinh response thường
            return self.generate_response(), None, 0
        
        # Sinh response đầu tiên
        current_response = self.generate_response()
        retries = 0

        while retries < self.MAX_RETRIES:
            # Đánh giá consistency
            score = evaluator_func(self.question, current_response)
            print(f"Attempt {retries + 1}: Consistency = {score:.4f}")
            
            # Nếu đạt yêu cầu, return
            if score >= self.THRESHOLD_CONS:
                self.consistency_score = score
                return current_response, score, retries
            
            # Nếu chưa đạt và còn lần thử, regenerate
            if retries < self.MAX_RETRIES - 1:
                current_response = self.regenerate_response(score)
                retries += 1
                print(f"Regenerated Response: {current_response}\n Retries: {retries}")
            else:
                # Hết lần thử, return response cuối cùng
                print(f"⚠️ Đã thử {self.MAX_RETRIES} lần, giữ response cuối cùng")
                self.consistency_score = score
                return current_response, score, retries
        
        return current_response, self.consistency_score, retries
