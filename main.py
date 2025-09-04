from src.model.text_gen import TextGenerator
from src.model.evaluate.CTRLEval import TextEvaluator
from src.model.loop import text_gen_loop
import pandas as pd
import argparse
import os

def main():
    # Thiết lập argument parser
    parser = argparse.ArgumentParser(description='Xử lý dữ liệu CSV với LLM')
    parser.add_argument('--input', '-i', 
                        type=str, 
                        required=True,
                        help='Đường dẫn đến file CSV input (ví dụ: data/vihallu-warmup.csv)')
    parser.add_argument('--output', '-o',
                        type=str,
                        default='submit.csv',
                        help='Đường dẫn file output (mặc định: submit.csv)')
    parser.add_argument('--encoding', '-e',
                        type=str,
                        default='utf-8',
                        choices=['utf-8', 'cp1258', 'latin1', 'iso-8859-1'],
                        help='Encoding của file CSV (mặc định: utf-8)')

    args = parser.parse_args()
    
    # Kiểm tra file input có tồn tại không
    if not os.path.exists(args.input):
        print(f"❌ Lỗi: Không tìm thấy file {args.input}")
        return
    
    print(f"📂 Đọc file: {args.input}")
    print(f"🔤 Encoding: {args.encoding}")
    print(f"💾 Output: {args.output}")
    
    # Đọc file CSV với encoding được chỉ định
    try:
        data = pd.read_csv(args.input, encoding=args.encoding)
        print(f"✅ Đọc thành công! Kích thước: {data.shape}")
    except UnicodeDecodeError as e:
        print(f"❌ Lỗi encoding: {e}")
        print("💡 Thử các encoding khác: utf-8, latin1, iso-8859-1")
        return
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return
    
    # Duyệt qua từng dòng để xử lý
    results = []
    
    print(f"\n🚀 Bắt đầu xử lý {len(data)} dòng dữ liệu...")

    for index, row in data.iterrows():
        try:
            # Lấy dữ liệu từ từng dòng
            context = row['context']
            prompt = row['prompt'] 
            
            print(f"📋 Đang xử lý dòng {index + 1}/{len(data)}")
            
            # Gọi hàm xử lý của bạn
            response, score, retries = text_gen_loop(context, prompt)
            
            # Lưu kết quả
            results.append({
                'id': row.get('id', index),
                'original_response': row.get('response', ''),
                'generated_response': response,
                'score': score,
                'retries': retries
            })
            
            print(f"   ✅ Hoàn thành - Score: {score:.3f}, Retries: {retries}")
            
        except Exception as e:
            print(f"   ❌ Lỗi tại dòng {index + 1}: {str(e)[:100]}")
            # Lưu lỗi vào kết quả
            results.append({
                'id': row.get('id', index),
                'original_response': row.get('response', ''),
                'generated_response': f"ERROR: {str(e)}",
                'score': 0.0,
                'retries': 0
            })

    # Chuyển kết quả thành DataFrame
    results_df = pd.DataFrame(results)

    # Save lại kết quả
    try:
        results_df.to_csv(args.output, index=False, encoding='utf-8')
        print(f"\n💾 Đã lưu kết quả vào: {args.output}")
    except Exception as e:
        print(f"\n❌ Lỗi lưu file: {e}")

if __name__ == "__main__":
    main()
