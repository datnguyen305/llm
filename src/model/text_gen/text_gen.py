from transformers import AutoTokenizer, AutoModelForCausalLM

class TextGenerator:
    def __init__(self, context, question):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        self.messages = [
            {
                "role": "system",
                "content": "Bạn là một trợ lý hữu ích. Trả lời câu hỏi của người dùng bằng tiếng Việt, chỉ dựa trên thông tin trong phần Context được cung cấp."
            },
            {
                "role": "user",
                "content": f"Ngữ cảnh: {context}\nCâu hỏi: {question}\nHãy trả lời ngắn gọn trong một đoạn văn."
            }
        ]

    def generate_response(self):
        # Tạo input từ template chat
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2042,        # giới hạn token sinh
            do_sample=True,            # cho random sampling (nếu muốn sáng tạo hơn)
            top_p=0.9,                 # nucleus sampling
            temperature=0.7            # kiểm soát độ đa dạng
        )

        # Lấy phần text mới sinh
        generated_answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True   # bỏ token đặc biệt
        )

        return generated_answer