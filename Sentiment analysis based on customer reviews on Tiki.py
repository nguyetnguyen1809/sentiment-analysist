# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 13:56:07 2024

@author: THANH HA
"""

import re
import time
import emoji
import string
import codecs
import pandas as pd
import numpy as np
import seaborn as sns
from pyvi import ViTokenizer
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#---------CHUẨN BỊ DỮ LIỆU----------
file_path = r"D:\Sentiment Analysis on Tiki\comments_data_ncds.csv"
df = pd.read_csv(file_path)

# Hiển thị thông tin chung về dữ liệu
print("Thông tin dữ liệu:")
print(df.info())

# Số lượng missing value trên từng cột
print("\nSố lượng missing value trên từng cột:")
missing_values = df.isnull().sum()
print(missing_values)


#----------TIỀN XỬ LÝ----------
# Xóa các cột không cần thiết
columns_to_drop = ['thank_count', 'id', 'customer_id', 'created_at', 'customer_name', 'purchased_at', 'title']
df = df.drop(columns=columns_to_drop, errors='ignore')  # errors='ignore' để tránh lỗi nếu thiếu cột

# Xóa giá trị null và trùng lặp
df = df.dropna(subset=['content'])  # Chỉ giữ lại các dòng có nội dung
df = df.drop_duplicates(subset=['content'], keep='first')  # Giữ dòng đầu tiên của các nội dung trùng lặp

# Đọc từ điển tích cực, tiêu cực, phủ định và stopwords
def read_txt_file(path):
    with codecs.open(path, 'r', encoding='UTF-8') as f:
        return [line.strip() for line in f.readlines()]

path_nag = r"D:\Sentiment Analysis on Tiki\nag.txt"
path_pos = r"D:\Sentiment Analysis on Tiki\pos.txt"
path_not = r"D:\Sentiment Analysis on Tiki\not.txt"
path_stopwords = r"D:\Sentiment Analysis on Tiki\vietnamese-stopwords.txt"

nag_list = read_txt_file(path_nag)
pos_list = read_txt_file(path_pos)
not_list = read_txt_file(path_not)
stopwords = set(read_txt_file(path_stopwords))
#Chuẩn hóa tiếng Việt,chuẩn hóa tiếng Anh, thuật ngữ
replace_list = {
       'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ',
       'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ', 'uả': 'ủa',
       'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
       'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
       'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
#Chuẩn hóa 1 số sentiment words/English words
       'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
       'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
       ' tks ': u' cám ơn ', 'thks': u'cám ơn', 'thanks': u'cám ơn', 'ths': u'cám ơn', 'thank': u' cám ơn ',
       'kg ': ' không ','not': ' không ',' kg ': 'không', 'k ': 'không','kh':u'không','kô':'không',' kp ': u' không phải ',
       'ko': 'không', ' k ': u' không ', 'khong': u' không ', ' hok ': u' không ',
       'cute': u' dễ thương ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
       'sz ': u' cỡ ', 'size': u' cỡ ', ' đx ': u' được ', 'dc': u' được ',
       'đc': u' được ','shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
       'sấu': u' xấu ','gut': u' tốt ', 'tot': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
       'time': u' thời gian ', 'qá': u' quá ', ' ship ': u' giao hàng ', ' m ': u' mình ', ' mik ': u' mình ',
       'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',
       'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',' síp ': u' giao hàng ',
       'beautiful': u' đẹp tuyệt vời ', ' tl ': u' trả lời ', ' r ': u' rồi ', ' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
       'chất lg': u' chất lượng ',' sd ': u' sử dụng ',' dt ': u' điện thoại ',' nt ': u' nhắn tin ',' sài ': u' xài ','bjo':u' bao giờ ',
       'thik': u' thích ',' sop ': u' cửa hàng ',  ' very ': u' rất ','quả ng ':u' quảng  ',
       'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', 'hàg': u' hàng ', 'qủa': u' quả ',
       'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', 'thik' : 'thích',
       ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ','fback':' feedback ','fedback':' feedback ',
       }
# Hàm chuẩn hóa dữ liệu
def normalize_text(text):
    # Loại bỏ các ký tự kéo dài (đẹpppp -> đẹp)
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
    
    # Chuyển thành chữ thường
    text = text.lower()
    
    # Loại bỏ emoji
    text = emoji.replace_emoji(text, replace="")  
    
    # Loại bỏ dấu câu
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    
    # Chuẩn hóa từ viết tắt/sai chính tả
    text = " ".join([replace_list.get(word, word) for word in text.split()])  # Sử dụng get để tránh lỗi KeyError

    #Tách từ 
    words = ViTokenizer.tokenize(text).split()  # Tách từ sau khi chuẩn hóa


    # Loại bỏ stopwords
    words = [word for word in words if word not in stopwords]
    
    
    # Bước 3: Xử lý phủ định
    processed_words = []

    i = 0
    while i < len(words):
        word = words[i]

        # Kiểm tra nếu là từ phủ định
        if word in not_list:
            if i + 1 < len(words):  # Kiểm tra từ tiếp theo
                next_word = words[i + 1]
               
               # Nếu từ tiếp theo là từ tích cực
                if next_word in pos_list:
                    processed_words.append("not-positive")
                    i += 2  # Bỏ qua từ tiếp theo
                    continue

                # Nếu từ tiếp theo là từ tiêu cực
                elif next_word in nag_list:
                    processed_words.append("not-negative")
                    i += 2  # Bỏ qua từ tiếp theo
                    continue

       # Nếu không nằm trong cụm phủ định, thêm từ hiện tại
        processed_words.append(word)
        i += 1

   # Gộp lại thành chuỗi văn bản
    return " ".join(processed_words)

# Áp dụng chuẩn hóa và xử lý văn bản
df['tokenized_content'] = df['content'].apply(normalize_text)

# Loại bỏ các dòng có giá trị rỗng trong cột 'tokenized_content'
df = df[df['tokenized_content'].notnull() & (df['tokenized_content'] != '')]

# Hiển thị số lượng dòng sau xử lý
print(f"Số dòng sau khi xử lý: {len(df)}")


#----------PHÂN TÍCH DỮ LIỆU----------
# Gán nhãn sentiment dựa trên rating
def map_sentiment(rating):
    if rating >= 4:
        return "Positive"
    else:
        return "Negative"

df['sentiment'] = df['rating'].apply(map_sentiment)

# Biểu đồ mô tả số lượng "Positive" và "Negative" trong cột 'sentiment'
ax = df['sentiment'].value_counts().sort_index() \
    .plot(kind='bar', title='Count of Positive and Negative Sentiments', figsize=(10, 5))

ax.set_xlabel('Sentiment')  # Đặt tên trục X là 'Sentiment'
ax.set_ylabel('Count')      # Đặt tên trục Y là 'Count'

# Thêm số cụ thể lên trên từng cột
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black')
plt.show()

# Đếm tổng số lượng dữ liệu trong DataFrame
total_data = len(df)

# Gộp tất cả các từ đã tokenized thành một chuỗi duy nhất
all_words = ' '.join(df['tokenized_content']).split()

# Lọc bỏ các từ là số
all_words_no_numbers = [word for word in all_words if not word.isdigit()]

# Đếm tần suất xuất hiện của các từ (không bao gồm số)
token_counts_no_numbers = Counter(all_words_no_numbers)
# Lấy ra 20 từ xuất hiện nhiều nhất
most_common_words_no_numbers = token_counts_no_numbers.most_common(20)

# Tách từ và tần suất để biểu diễn biểu đồ
words_no_numbers, counts_no_numbers = zip(*most_common_words_no_numbers)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
plt.bar(words_no_numbers, counts_no_numbers, color='skyblue')
plt.xlabel('Từ', fontsize=12)
plt.ylabel('Tần suất xuất hiện', fontsize=12)
plt.title('20 từ xuất hiện nhiều nhất trong dữ liệu (không tính số)', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()

# Tạo WordCloud từ tần suất các từ 
wcloud_no_numbers = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(token_counts_no_numbers)

# Vẽ WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wcloud_no_numbers, interpolation="bilinear")
plt.axis("off")
plt.title('WordCloud từ tần suất các từ')
plt.show()

# Xuất dữ liệu đã chuẩn hóa ra file mới
print("File đã được xuất ra file D:\Sentiment Analysis on Tiki\comments_cleaned.csv")
output_path = r"D:\Sentiment Analysis on Tiki\comments_cleaned.csv"
df.to_csv(output_path, index=False, encoding='utf-8')



#----------TRÍCH XUẤT ĐẶC TRƯNG VÀ VECTO HÓA-----------
# Đọc file csv
file_path = r"D:\Sentiment Analysis on Tiki\comments_cleaned.csv"
df = pd.read_csv(file_path)

print('Dataset columns, rows:', df.shape)

print('Số lượng missing data ở từng cột trong bô dữ liệu sau khi được xử lý')
print(df.isnull().sum()) 


# Bước 1: Chuẩn bị dữ liệu
X = df['tokenized_content']  # Các đặc trưng (features)
y = df['sentiment']  # Nhãn (labels)

# Chia thành 80% cho huấn luyện và 20% còn lại cho kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#--- Lấy nội dung cần tạo TF-IDF ---
# Khởi tạo TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit và transform tập huấn luyện
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform các tập testing
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# Cân bằng dữ liệu bằng SMOTE trên tập huấn luyện
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_tfidf.toarray(), y_train)  # Chuyển đổi từ sparse matrix sang array


# Tạo đối tượng KFold (StratifiedKFold đảm bảo phân bố nhãn đồng đều)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#Bước 3:Hàm huấn luyện và đánh giá mô hình
#Hàm tính chỉ số
def calculate_results(y_true, y_pred):
    '''
    Calculate accuracy, precision, recall, f1 score for a model
    '''
    model_accuracy = accuracy_score(y_true, y_pred)   # Scale to 1-100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    
    model_results = {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1
    }
    
    return model_results

# Hàm đánh giá mô hình với KFold
def evaluate_model_kfold(model,param_grid, X, y, kfold):
    fold_results = []
    fold_result_each_kfold = []
   

    for train_ids, val_ids in  kfold.split(X, y):
        X_train_kfold, X_val = X[train_ids], X[val_ids]
        y_train_kfold, y_val = y.iloc[train_ids], y.iloc[val_ids]
        
        
        # Huấn luyện mô hình với GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=3,n_jobs=-1)
        grid_search.fit(X_train_kfold, y_train_kfold)
        
        # Dự đoán trên tập validation
        val_pred = grid_search.predict(X_val)
        
        
        # Tính toán trên cả 2 tập 
        val_metrics = calculate_results(y_val, val_pred)

        fold_result = {
            'fold': len(fold_results) + 1,
            'best_params':  grid_search.best_params_,
            'validation_metrics': val_metrics
        }
        fold_results.append(val_metrics)
        fold_result_each_kfold.append(fold_result)
        
        #Tính trung bình các chỉ số trên tất cả các folds 
        # So sánh kết quả
    model_results = {
        "accuracy": np.mean([result["accuracy"] for result in fold_results]),
        "precision": np.mean([result["precision"] for result in fold_results]),
        "recall": np.mean([result["recall"] for result in fold_results]),
        "f1": np.mean([result["f1"] for result in fold_results]),
        "best_params": grid_search.best_params_
    }
    # Huấn luyện lại mô hình với toàn bộ dữ liệu
    model.fit(X, y)

    return model_results
# Hàm in kết quả trung bình của các chỉ số
def print_results(fold_results):
    print(f"Average Results Across All Folds:")
    print(f" - Accuracy: {fold_results['accuracy']:.4f}")
    print(f" - Precision: {fold_results['precision']:.4f}")
    print(f" - Recall: {fold_results['recall']:.4f}")
    print(f" - F1 Score: {fold_results['f1']:.4f}")
    print(f" - Best Params: {fold_results['best_params']}\n")
    
    
#----------ĐÁNH GIÁ MÔ HÌNH----------
# -------------------------------------------
#  Xây dựng mô hình huấn luyện bằng Naive Bayes
# -------------------------------------------
param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]  # Tham số alpha cho Naive Bayes
}
#Thêm biến để kiểm tra thời gian huấn luyện
start_NB = time.time()

# Khởi tạo mô hình Naive Bayes
model_NB = MultinomialNB()
nb_results = evaluate_model_kfold(model_NB,param_grid_nb, X_resampled, y_resampled, kfold)

end_NB = time.time()
time_NB = end_NB - start_NB

# In thời gian huấn luyện và báo cáo phân loại
print("================Mô hình Navie Bayes==================")
print("Thời gian huấn luyện: ", str(time_NB), " giây")

# Hiển thị kết quả trung bình
print_results(nb_results)

#Đánh giá mô hình
y_pred_nb=model_NB.predict(X_test_tfidf)
report = classification_report(y_test, y_pred_nb)
print("Classification Report:\n", report)

# -------------------------------------------
#  Xây dựng mô hình Logistic Regression
# -------------------------------------------
param_grid_lr = {
    'C': [ 0.01,0.1, 1, 10] # Tham số C cho Logistic Regression  # Các solver khác nhau
}
start_lr=time.time()

logistic_model = LogisticRegression(max_iter=1000)
lr_results = evaluate_model_kfold(logistic_model,param_grid_lr, X_resampled, y_resampled, kfold)

end_lr=time.time()
time_lr=end_lr-start_lr
#Show ra thời gian huấn luyện của mô hình
print("================Mô hình Logistic Regression==================")
print("Thời gian huấn luyện: ", str(time_lr), " giây")

# Hiển thị kết quả trung bình
print_results(lr_results)

# Đánh giá mô hình hình
y_pred_lg = logistic_model.predict(X_test_tfidf)
report_lg = classification_report(y_test, y_pred_lg)
print("Classification Report:\n", report_lg)

# -------------------------------------------
#  Xây dựng mô hình Support Vector Machine (SVM)
# -------------------------------------------
param_grid_svm = {
    'C': [ 0.1, 1, 10] # Tham số C cho SVM
}

start_SVM = time.time()

model_SVM = LinearSVC(max_iter=10000)
SVM_results = evaluate_model_kfold(model_SVM,param_grid_svm, X_resampled, y_resampled, kfold)

end_SVM = time.time()
time_SVM=end_SVM - start_SVM


#Show ra thời gian huấn luyện của mô hình
print("===================Mô hình SVM====================")
print("Thời gian huấn luyện: ", str(time_SVM), " giây")

# Hiển thị kết quả trung bình
print_results(SVM_results)

# Đánh giá mô hình hình
y_pred_SVM = model_SVM.predict(X_test_tfidf)

report_SVM = classification_report(y_test, y_pred_SVM)
print("Classification Report:\n", report_SVM)


# Trích xuất các chỉ số từ kết quả
models = ['Naive Bayes', 'Logistic Regression', 'SVM']
accuracy = [nb_results['accuracy'], lr_results['accuracy'], SVM_results['accuracy']]
precision = [nb_results['precision'], lr_results['precision'], SVM_results['precision']]
recall = [nb_results['recall'], lr_results['recall'], SVM_results['recall']]
f1_score = [nb_results['f1'], lr_results['f1'], SVM_results['f1']]

# Tạo biểu đồ
x = np.arange(len(models))  # Vị trí trên trục x
width = 0.2 # Độ rộng của mỗi cột

fig, ax = plt.subplots(figsize=(10, 6))

# Vẽ các cột cho từng chỉ số
rects1 = ax.bar(x - width*1.5, accuracy, width, label='Accuracy', color='lightblue')
rects2 = ax.bar(x - width/2, precision, width, label='Precision', color='gold')
rects3 = ax.bar(x + width/2, recall, width, label='Recall', color='lightgreen')
rects4 = ax.bar(x + width*1.5, f1_score, width, label='F1 Score', color='salmon')

# Thêm nhãn và tiêu đề
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Hiển thị giá trị trên cột
for rects in [rects1, rects2, rects3, rects4]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    ha='center', va='bottom')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()


# Kết hợp 3 mô hình với phương pháp Major Voting
ensemble_model = VotingClassifier(
    estimators=[
        ('nb', model_NB),
        ('lr', logistic_model),
        ('svm', model_SVM)
    ],
    voting='hard'  # 'hard' dùng Major Voting
    )

# Huấn luyện mô hình Ensemble trên tập resampled
ensemble_model.fit(X_resampled, y_resampled)

# Dự đoán trên tập kiểm tra
y_pred_ensemble = ensemble_model.predict(X_test_tfidf)

# Báo cáo kết quả
ensemble_report = classification_report(y_test, y_pred_ensemble)
print("Classification Report for Ensemble Model:\n", ensemble_report)



from sklearn.metrics import ConfusionMatrixDisplay
# Hiển thị Confusion Matrix cho từng mô hình
models_to_evaluate = {
    'Naive Bayes': model_NB,
    'Logistic Regression': logistic_model,
    'SVM': model_SVM,
    'Ensemble': ensemble_model
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (model_name, model) in enumerate(models_to_evaluate.items()):
    y_pred = model.predict(X_test_tfidf)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=axes[idx], cmap='Blues', colorbar=False
    )
    axes[idx].set_title(f'Confusion Matrix: {model_name}', fontsize=12)

plt.tight_layout()
plt.show()


#----------ÁP DỤNG DỮ LIỆU MỚI-----------
# Dự đoán với dữ liệu mới
new_text = ["gói hàng cũng tạm, hàng thì ổn", " Tôi thích cái bình này, nó rất tốt"]
new_text_tfidf = tfidf_vectorizer.transform(new_text)

predictions = ensemble_model.predict(new_text_tfidf)

for text, prediction in zip(new_text, predictions):
    print(f"Câu: '{text}' => Sentiment: {prediction}")
