from src.model.text_gen import TextGenerator
from src.model.evaluate.CTRLEval import TextEvaluator
from src.model.loop import text_gen_loop
import pandas as pd

if __name__ == "__main__":
    # Đọc file CSV với encoding phù hợp
    data = pd.read_csv('vihallu-warmup.csv')

    # Duyệt qua từng dòng để xử lý
    results = []

    for index, row in data.iterrows():
        # Lấy dữ liệu từ từng dòng
        context = row['context']
        prompt = row['prompt'] 
        
        print(f"Đang xử lý dòng {index + 1}/{len(data)}")
        
        # Gọi hàm xử lý của bạn
        response, score, retries = text_gen_loop(context, prompt)
        
        # Lưu kết quả
        results.append({
            'id': row['id'],
            'original_response': row['response'],
            'generated_response': response,
            'score': score,
            'retries': retries
        })

    # Chuyển kết quả thành DataFrame
    results_df = pd.DataFrame(results)

    # Save lại kết quả
    results_df.to_csv('submit.csv', index=False)
