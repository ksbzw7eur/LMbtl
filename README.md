1. Nội dung 
1.1. Bài toán và động cơ nghiên cứu
Trong thiên văn học hiện đại, việc phân loại tự động các đối tượng thiên văn từ dữ liệu quan sát đóng vai trò quan trọng trong việc xử lý khối lượng dữ liệu ngày càng lớn từ các kính thiên văn khảo sát bầu trời. Một dạng dữ liệu phổ biến là light curve, biểu diễn sự thay đổi độ sáng của một đối tượng theo thời gian.
Bài toán trong MALLORN Astronomical Classification Challenge yêu cầu phân loại các đối tượng thiên văn dựa trên dữ liệu light curve thu thập không đều theo thời gian. Bài toán này gặp nhiều thách thức:
Chuỗi thời gian không đều, không có cùng số lượng quan sát


Dữ liệu chứa nhiều nhiễu và ngoại lai


Các lớp có số lượng mẫu rất khác nhau (mất cân bằng lớp)


Dữ liệu test không có nhãn, yêu cầu mô hình có khả năng tổng quát hóa tốt


Do đó, nhóm không tiếp cận bài toán như một bài toán chuỗi thời gian thuần túy, mà chuyển đổi nó thành bài toán phân loại dữ liệu bảng (tabular classification) thông qua trích xuất đặc trưng.

1.2. Quy trình tổng thể
Quy trình xây dựng mô hình được thiết kế theo các bước sau:
Tiền xử lý và chuẩn hóa dữ liệu light curve


Trích xuất đặc trưng mô tả hành vi biến thiên của đối tượng


Kết hợp dữ liệu quan sát với metadata thiên văn


Huấn luyện và tinh chỉnh các mô hình học máy


Kết hợp mô hình để cải thiện hiệu suất dự đoán


Quy trình này cho phép mô hình học được các đặc điểm quan trọng của light curve mà không phụ thuộc vào cấu trúc chuỗi thời gian đều.

1.3. Tiền xử lý dữ liệu
Giá trị Flux trong light curve có phân phối không đối xứng, biên độ lớn và chứa giá trị ngoại lai. Để giảm ảnh hưởng của các yếu tố này, nhóm áp dụng phép biến đổi log có giữ dấu:
Fluxlog=sign(Flux)⋅log⁡(1+∣Flux∣)\text{Flux}_{log} = \text{sign(Flux)} \cdot \log(1 + |\text{Flux}|)Fluxlog​=sign(Flux)⋅log(1+∣Flux∣)
Phép biến đổi này giúp:
Giảm độ lệch của phân phối


Hạn chế tác động của các giá trị cực đoan


Giữ lại thông tin vật lý về chiều biến thiên của độ sáng


Sau đó, dữ liệu được sắp xếp theo thời gian để đảm bảo việc tính toán các đặc trưng động học là chính xác.

1.4. Trích xuất đặc trưng (Feature Engineering)
Đây là bước quan trọng nhất trong toàn bộ phương pháp.
1.4.1. Đặc trưng thống kê cơ bản
Với mỗi đối tượng (object_id), nhóm trích xuất các đặc trưng thống kê như:
Giá trị trung bình và trung vị


Độ lệch chuẩn


Giá trị nhỏ nhất và lớn nhất


Độ lệch phân phối (skewness)


Số lượng điểm quan sát


Những đặc trưng này cung cấp cái nhìn tổng quát về mức độ và phạm vi biến thiên của light curve.

1.4.2. Đặc trưng phân vị và biên độ
Thay vì chỉ sử dụng min–max, nhóm sử dụng phân vị 10% và 90% để mô tả biên độ biến thiên một cách bền vững hơn. Từ đó xây dựng:
Amplitude = p90 − p10


Amplitude chuẩn hóa theo trung vị


Độ lệch chuẩn chuẩn hóa


Các đặc trưng này giúp mô hình phân biệt các nguồn có biên độ biến thiên khác nhau.

1.4.3. Đặc trưng động học
Để mô tả sự thay đổi ngắn hạn của light curve, nhóm tính:
Trung bình của sai khác Flux liên tiếp


Độ lệch chuẩn của sai khác Flux liên tiếp


Các đặc trưng này phản ánh mức độ “gồ ghề” của light curve, hữu ích trong việc phân biệt các loại nguồn biến thiên nhanh và chậm.

1.4.4. Xu hướng theo thời gian
Nhóm ước lượng hệ số góc của hồi quy tuyến tính Flux theo thời gian (slope). Đặc trưng này giúp mô hình nhận biết xu hướng tăng hoặc giảm độ sáng tổng thể của đối tượng.

1.4.5. Đặc trưng bền vững
Median Absolute Deviation (MAD) được sử dụng để đo mức độ biến thiên nhưng ít nhạy với ngoại lai, giúp mô hình ổn định hơn khi dữ liệu có nhiễu.

1.5. Kết hợp metadata
Ngoài dữ liệu light curve, nhóm sử dụng thêm hai đặc trưng vật lý:
Z (redshift) – liên quan đến khoảng cách và sự dịch chuyển quang phổ


EBV – mô tả mức suy giảm ánh sáng do bụi


Việc kết hợp metadata giúp mô hình học được các đặc điểm vật lý bổ sung mà light curve đơn thuần không thể hiện đầy đủ.

2. Kỹ thuật Machine Learning 
2.1. Lựa chọn mô hình
Nhóm lựa chọn ba thuật toán học máy có bản chất khác nhau:
LightGBM làm mô hình chính do khả năng xử lý dữ liệu tabular hiệu quả và mô hình hóa quan hệ phi tuyến


Random Forest giúp giảm phương sai và tăng tính ổn định


Logistic Regression đóng vai trò baseline tuyến tính và giúp đa dạng hóa ensemble


Việc kết hợp các mô hình khác nhau giúp giảm rủi ro phụ thuộc vào một thuật toán duy nhất.

2.2. Xử lý mất cân bằng lớp
Dữ liệu có sự chênh lệch lớn giữa các lớp. Nhóm sử dụng trọng số mẫu tỉ lệ nghịch với tần suất lớp để buộc mô hình quan tâm nhiều hơn đến các lớp hiếm, từ đó cải thiện Macro F1-score.

2.3. Ensemble learning
Dự đoán cuối cùng được tạo bằng cách kết hợp xác suất dự đoán của các mô hình theo soft voting. Trọng số được phân bổ dựa trên hiệu quả thực nghiệm, trong đó LightGBM và Random Forest được ưu tiên cao hơn.

3. Hiệu suất mô hình 
3.1. Chỉ số đánh giá
Hiệu suất mô hình được đánh giá bằng Macro F1-score, phù hợp với bài toán đa lớp mất cân bằng vì mỗi lớp đều có mức độ đóng góp như nhau vào kết quả cuối.

3.2. Kết quả thực nghiệm
Mô hình ensemble sau khi huấn luyện trên toàn bộ tập dữ liệu huấn luyện đạt:
Macro F1-score = 0.2896 trên Kaggle leaderboard


Kết quả này cho thấy mô hình có khả năng tổng quát hóa tốt trên tập test ẩn.

4. Cải tiến và tối ưu mô hình
Phần này trình bày chi tiết các chiến lược được sử dụng để cải thiện hiệu suất mô hình, bao gồm xây dựng đặc trưng, tinh chỉnh siêu tham số và các cải tiến bổ sung nhằm nâng cao khả năng tổng quát hóa của mô hình.

4.1. Tối ưu thông qua xây dựng đặc trưng (Feature Engineering)
Do dữ liệu đầu vào là các chuỗi light curve không đều theo thời gian, nhóm không sử dụng trực tiếp các mô hình chuỗi thời gian, mà tập trung vào việc thiết kế các đặc trưng có ý nghĩa vật lý và thống kê. Mục tiêu của bước này là biến đổi dữ liệu thô thành một không gian đặc trưng mà các mô hình học máy có thể khai thác hiệu quả.
4.1.1. Chuẩn hóa và ổn định phân phối Flux
Giá trị Flux có phân phối lệch và chứa nhiều giá trị ngoại lai. Nhóm áp dụng phép biến đổi log có giữ dấu:
Flux𝑙𝑜𝑔=sign(Flux)⋅log⁡(1+∣Flux∣)Fluxlog​=sign(Flux)⋅log(1+∣Flux∣)
Biến đổi này giúp:
Giảm độ lệch phân phối
Hạn chế ảnh hưởng của các giá trị cực đoan
Giữ lại thông tin về chiều biến thiên của độ sáng
Đây là bước tiền xử lý quan trọng giúp các đặc trưng thống kê phía sau trở nên ổn định và đáng tin cậy hơn.

4.1.2. Đặc trưng thống kê mô tả toàn cục
Với mỗi đối tượng thiên văn, nhóm trích xuất các đặc trưng thống kê toàn cục như trung bình, trung vị, độ lệch chuẩn, min, max, skewness và số lượng quan sát. Những đặc trưng này phản ánh:
Mức độ sáng trung bình
Độ phân tán của light curve
Quy mô và mật độ dữ liệu quan sát
Đặc biệt, số lượng quan sát được giữ lại như một đặc trưng độc lập vì nó gián tiếp phản ánh mức độ tin cậy của các thống kê.

4.1.3. Đặc trưng phân vị và biên độ biến thiên
Thay vì chỉ sử dụng giá trị cực trị (min–max), nhóm sử dụng phân vị 10% và 90% để mô tả biên độ biến thiên một cách bền vững trước nhiễu. Từ đó, các đặc trưng sau được xây dựng:
Biên độ biến thiên (amplitude)
Biên độ chuẩn hóa theo trung vị
Độ lệch chuẩn chuẩn hóa
Các đặc trưng này cho phép mô hình phân biệt rõ hơn giữa các loại nguồn có mức độ biến thiên thấp, trung bình hoặc mạnh.

4.1.4. Đặc trưng động học và biến thiên ngắn hạn
Để mô tả sự thay đổi ngắn hạn của light curve, nhóm tính:
Trung bình của sai khác Flux liên tiếp
Độ lệch chuẩn của sai khác Flux liên tiếp
Những đặc trưng này phản ánh tốc độ và cường độ biến thiên trong khoảng thời gian ngắn, giúp mô hình nhận diện các đối tượng có biến thiên nhanh so với các nguồn biến thiên chậm.

4.1.5. Đặc trưng xu hướng theo thời gian
Xu hướng tổng thể của light curve được mô tả thông qua hệ số góc của hồi quy tuyến tính Flux theo thời gian (slope). Đặc trưng này cung cấp thông tin về việc độ sáng của đối tượng có xu hướng tăng, giảm hay ổn định trong suốt thời gian quan sát.

4.1.6. Đặc trưng bền vững (Robust Features)
Nhóm sử dụng Median Absolute Deviation (MAD) để đo mức độ biến thiên nhưng ít nhạy với ngoại lai. Điều này giúp mô hình ổn định hơn khi xử lý các light curve có nhiễu mạnh hoặc giá trị bất thường.
4.2. Tinh chỉnh siêu tham số (Hyperparameter Tuning)
Sau khi xây dựng tập đặc trưng, nhóm tập trung tinh chỉnh các siêu tham số của mô hình nhằm cân bằng giữa độ chính xác và khả năng tổng quát hóa.
4.2.1. Tinh chỉnh mô hình LightGBM
LightGBM là mô hình chính của hệ thống. Các siêu tham số được lựa chọn dựa trên thử nghiệm thực nghiệm và hiểu biết về đặc điểm dữ liệu:
num_leaves và max_depth được giới hạn để tránh mô hình quá phức tạp
min_data_in_leaf được tăng nhằm giảm overfitting
feature_fraction và bagging_fraction được sử dụng để tăng tính ngẫu nhiên
learning_rate được giữ ở mức vừa phải để đảm bảo quá trình học ổn định
Các lựa chọn này giúp mô hình học được quan hệ phi tuyến quan trọng mà không bị phụ thuộc quá mức vào nhiễu trong dữ liệu.

4.2.2. Xử lý mất cân bằng lớp bằng trọng số mẫu
Do sự chênh lệch lớn về số lượng mẫu giữa các lớp, nhóm áp dụng trọng số mẫu tỉ lệ nghịch với tần suất lớp. Việc này giúp mô hình ưu tiên học tốt hơn cho các lớp hiếm, cải thiện hiệu suất Macro F1-score – chỉ số đánh giá chính của bài toán.

4.2.3. Tinh chỉnh các mô hình bổ trợ
Random Forest: giới hạn độ sâu cây và số mẫu tối thiểu ở lá để giảm overfitting
Logistic Regression: chuẩn hóa dữ liệu và tăng số vòng lặp để đảm bảo hội tụ
Mặc dù các mô hình này không được tinh chỉnh quá sâu, chúng đóng vai trò quan trọng trong việc tăng tính đa dạng của ensemble.

4.3. Các cải tiến bổ sung
4.3.1. Ensemble learning
Thay vì sử dụng một mô hình đơn lẻ, nhóm áp dụng ensemble soft voting để kết hợp dự đoán của nhiều mô hình. Cách tiếp cận này giúp:
Giảm phương sai của dự đoán
Tăng độ ổn định trên tập test
Khai thác điểm mạnh của từng mô hình
Trọng số ensemble được lựa chọn dựa trên hiệu quả thực nghiệm, ưu tiên các mô hình có hiệu suất cao hơn.

4.3.2. Huấn luyện trên toàn bộ tập dữ liệu
Sau khi hoàn thiện cấu hình mô hình, nhóm huấn luyện mô hình trên toàn bộ tập huấn luyện nhằm tận dụng tối đa dữ liệu sẵn có, từ đó cải thiện khả năng tổng quát hóa trên tập test ẩn.

4.3.3. Đảm bảo khả năng tổng quát hóa
Toàn bộ các bước tối ưu được thiết kế nhằm tránh overfitting:
Không sử dụng đặc trưng quá phức tạp
Không tuning quá sâu
Ưu tiên các đặc trưng có ý nghĩa vật lý và thống kê rõ ràng

4.4. Tổng kết
Thông qua việc kết hợp feature engineering có định hướng, tinh chỉnh siêu tham số hợp lý và ensemble learning, nhóm đã cải thiện đáng kể hiệu suất mô hình so với các phiên bản ban đầu. Các cải tiến này giúp mô hình đạt kết quả tốt hơn trên Kaggle leaderboard và đảm bảo tính ổn định khi áp dụng vào dữ liệu chưa từng thấy.

5. Kết luận
Báo cáo đã trình bày chi tiết quá trình xây dựng và tối ưu mô hình phân loại đối tượng thiên văn từ dữ liệu light curve không đều. Phương pháp dựa trên feature engineering thủ công kết hợp ensemble học máy cho thấy hiệu quả và phù hợp với bài toán thực tế.


