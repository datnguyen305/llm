from src.model.text_gen import TextGenerator
from src.model.evaluate.CTRLEval import TextEvaluator
from src.model.loop import text_gen_loop

if __name__ == "__main__":
    context = 'Theo pháp lệnh Vincennes năm 1374, vương quốc được điều hành bởi Nhiếp chính vương cho đến khi Louis lên 13 tuổi. Danh hiệu Nhiếp chính được trao cho người bà con gần nhất là ông chú của nhà vua Philippe, Quận công xứ Orleans. Louis XIV, tuy nhiên, không tín nhiệm Philippe, một người lính kiệt xuất, nhưng bị nhà vua coi là kẻ không sùng đạo. Nhà vua gọi Philippe là Fanfaron des crimes ("đầu sỏ của tội ác)" Louis XIV muốn quyền điều hành Hội đồng Nhiếp chính phải giao cho người con ngoại hôn được ông rất thương yêu, Quận công xứ Maine (con triêng của Louis XIV với Madame de Montespan). Tháng 8 năm 1714, không lâu trước khi chết, nhà vua viết di chiếu lệnh hạn chế quyền hạn của người chấp chính; theo đó quốc gia sẽ được điều hành bởi Hội đồng Nhiếp chính gồm 14 thành viên cho đến năm tân vương 13 tuổi. Philippe là cháu gọi Louis XIV là bác, làm Chủ tịch Hội đồng, nhưng còn có các thành viên khác bao gồm Quận công xứ Maine cùng các đồng minh. Quyết định của triều đình được ban xuống theo chế độ đa số phiếu, nghĩa là quyền lực Nhiếp chính vương có thể bị bác bỏ bởi nhóm Maine. Orléans nhìn ra điều đó, và ngay sau khi nhà vua qua đời, ông đến Nghị viện Paris, một Hội đồng quý tộc gồm nhiều đồng minh của ông, và Nghị viện đã hủy bỏ tờ di chiếu. Để đổi lấy sự ủng hộ của họ, Orléans cho khôi phục droit de remontrance (quyền phản đối) của Nghị viện - vốn bị Louis XIV triệt bỏ từ trước, theo đó Nghị viện có quyền phản đối những quyết định của nhà vua mà họ cho là trái với lợi ích dân tộc. Quyền phản đối làm suy yếu quyền hành của quân chủ và đánh dấu khởi đầu xung đột giữa Nhà vua và Nghị viện mà đỉnh điểm là Cách mạng Pháp năm 1789.'
    question = 'Quyền phản đối được khôi phục bởi Orléans đã giúp cho Nghị viện có được quyền lực nào, mặc dù nhà vua luôn duy trì quyền lực tuyệt đối và không bao giờ cho phép bất kỳ sự can thiệp nào từ Nghị viện?'

    print("Starting text generation loop...")
    response, score, retries = text_gen_loop(context, question)
    
    print(f"Generated Response: {response}")
    print(f"Consistency Score: {score:.4f}")
    print(f"Number of Retries: {retries}")
