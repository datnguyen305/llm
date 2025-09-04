import re

def remove_stars(generated_answer: str) -> str:
    generated_answer = re.sub(r'\*+', '', generated_answer)   # bỏ tất cả dấu *
    return generated_answer