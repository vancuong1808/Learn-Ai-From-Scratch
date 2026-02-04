"""
TOKYO HOUSE PRICE PREDICTION - ĐƠN GIẢN CHO NGƯỜI MỚI HỌC
========================================================

Pipeline hoàn chỉnh để dự đoán giá nhà ở Tokyo với Linear Regression:
✅ Load và khám phá dữ liệu
✅ Phân tích và xử lý missing data (MCAR, MNAR, MAR)
✅ Feature engineering đơn giản
✅ Training Linear Regression
✅ Đánh giá và dự đoán

Tác giả: AI Assistant
Ngày: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Thiết lập style cho plots
plt.style.use('default')
sns.set_palette("husl")

print("🏠 TOKYO HOUSE PRICE PREDICTION - SIMPLE VERSION")
print("=" * 60)

# ========================================
# BƯỚC 1: LOAD VÀ KHÁM PHÁ DỮ LIỆU
# ========================================
print("\n📁 BƯỚC 1: Load và khám phá dữ liệu")
print("-" * 40)

try:
    # Load datasets
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    sample_submission = pd.read_csv("sample_submission.csv")
    
    print("✅ Load dữ liệu thành công!")
    print(f"   📊 Train data: {train_data.shape[0]:,} rows, {train_data.shape[1]} columns")
    print(f"   📊 Test data: {test_data.shape[0]:,} rows, {test_data.shape[1]} columns")
    
except FileNotFoundError as e:
    print(f"❌ Lỗi: Không tìm thấy file - {e}")
    print("💡 Hãy đảm bảo các file train.csv, test.csv, sample_submission.csv ở cùng thư mục")
    exit(1)

# Thông tin cơ bản về dataset
print(f"\n🔍 Thông tin dataset:")
numerical_cols = train_data.select_dtypes(include=[np.number]).columns
categorical_cols = train_data.select_dtypes(include=['object']).columns

print(f"   🔢 Numerical features: {len(numerical_cols)}")
print(f"   📝 Categorical features: {len(categorical_cols)}")

# Phân tích target variable
target = train_data['Price_JPY']
print(f"\n🎯 Phân tích biến mục tiêu (Price_JPY):")
print(f"   💰 Giá trung bình: {target.mean():,.0f} JPY")
print(f"   💰 Giá trung vị: {target.median():,.0f} JPY")
print(f"   💰 Giá thấp nhất: {target.min():,.0f} JPY")
print(f"   💰 Giá cao nhất: {target.max():,.0f} JPY")
print(f"   📈 Độ lệch chuẩn: {target.std():,.0f} JPY")

# Visualize target distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(target, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Phân phối giá nhà')
plt.xlabel('Giá (JPY)')
plt.ylabel('Tần suất')

plt.subplot(1, 3, 2)
plt.hist(np.log1p(target), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Phân phối Log(Giá)')
plt.xlabel('Log(Giá)')
plt.ylabel('Tần suất')

plt.subplot(1, 3, 3)
plt.boxplot(target)
plt.title('Box Plot - Phát hiện outliers')
plt.ylabel('Giá (JPY)')

plt.tight_layout()
plt.show()

# ========================================
# BƯỚC 2: PHÂN TÍCH MISSING DATA
# ========================================
print("\n" + "=" * 60)
print("🔍 BƯỚC 2: Phân tích Missing Data")
print("=" * 60)

def analyze_missing_data(df, name="Dataset"):
    """Phân tích chi tiết missing data"""
    print(f"\n📋 Phân tích Missing Data - {name}:")
    
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_count,
        'Missing_Percent': missing_percent,
        'Data_Type': df.dtypes
    })
    
    # Chỉ hiển thị columns có missing data
    missing_only = missing_summary[missing_summary['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    if len(missing_only) == 0:
        print("   ✅ Không có missing data!")
        return missing_only
    
    print(f"   📊 Tổng quan:")
    print(f"      - Columns có missing: {len(missing_only)}/{len(df.columns)}")
    print(f"      - Tổng missing values: {missing_count.sum():,}")
    
    print(f"\n   📈 Chi tiết:")
    for _, row in missing_only.head(10).iterrows():
        print(f"      - {row['Column']}: {row['Missing_Count']:,} ({row['Missing_Percent']:.1f}%) - {row['Data_Type']}")
    
    return missing_only

# Phân tích missing data cho cả train và test
missing_train = analyze_missing_data(train_data, "Train")
missing_test = analyze_missing_data(test_data, "Test")

# Giải thích các loại Missing Data Mechanisms
print(f"\n🧠 CÁC LOẠI MISSING DATA MECHANISMS:")
print(f"   💡 MCAR (Missing Completely At Random):")
print(f"      - Missing hoàn toàn ngẫu nhiên, không liên quan đến bất kỳ biến nào")
print(f"      - Ví dụ: Lỗi kỹ thuật khi thu thập dữ liệu")
print(f"   💡 MAR (Missing At Random):")
print(f"      - Missing phụ thuộc vào các biến khác có thể quan sát được")
print(f"      - Ví dụ: Người trẻ ít khai báo thu nhập hơn người già")
print(f"   💡 MNAR (Missing Not At Random):")
print(f"      - Missing phụ thuộc vào chính giá trị bị missing")
print(f"      - Ví dụ: Nhà chưa renovation thì YearRenovated = 0")

# Phân tích cụ thể YearRenovated
if 'YearRenovated' in train_data.columns:
    zero_renovation = (train_data['YearRenovated'] == 0).sum()
    missing_renovation = train_data['YearRenovated'].isnull().sum()
    
    print(f"\n🏠 Phân tích đặc biệt - YearRenovated:")
    print(f"   📊 Giá trị 0 (chưa renovation): {zero_renovation:,} ({zero_renovation/len(train_data)*100:.1f}%)")
    print(f"   📊 Giá trị missing (NaN): {missing_renovation:,}")
    print(f"   💭 Kết luận: Đây là trường hợp MNAR vì 0 có ý nghĩa 'chưa được renovation'")

# ========================================
# BƯỚC 3: XỬ LÝ MISSING DATA
# ========================================
print("\n" + "=" * 60)
print("🔧 BƯỚC 3: Xử lý Missing Data")
print("=" * 60)

# Tạo bản sao để xử lý
train_clean = train_data.copy()
test_clean = test_data.copy()

print(f"📝 CHIẾN LƯỢC XỬ LÝ:")

# 1. Xử lý YearRenovated (MNAR case)
if 'YearRenovated' in train_clean.columns:
    print(f"\n1️⃣ Xử lý YearRenovated (MNAR):")
    print(f"   - Chuyển đổi 0 → NaN (vì 0 có nghĩa là 'chưa renovation')")
    print(f"   - Sau đó impute NaN với median của các giá trị renovation thực tế")
    
    # Chuyển 0 thành NaN
    train_clean['YearRenovated'] = train_clean['YearRenovated'].replace(0, np.nan)
    test_clean['YearRenovated'] = test_clean['YearRenovated'].replace(0, np.nan)

# 2. Phân loại features
numerical_features = train_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train_clean.select_dtypes(include=['object']).columns.tolist()

# Loại bỏ target variable
if 'Price_JPY' in numerical_features:
    numerical_features.remove('Price_JPY')

print(f"\n2️⃣ Phân loại features:")
print(f"   🔢 Numerical: {len(numerical_features)} features")
print(f"   📝 Categorical: {len(categorical_features)} features")

# 3. Impute numerical features với median
print(f"\n3️⃣ Xử lý Numerical Features (Impute với Median):")
for col in numerical_features:
    if train_clean[col].isnull().sum() > 0:
        # Tính median từ train set
        median_val = train_clean[col].median()
        
        # Impute cả train và test
        train_clean[col].fillna(median_val, inplace=True)
        test_clean[col].fillna(median_val, inplace=True)
        
        print(f"   ✅ {col}: imputed with median = {median_val:.2f}")

# 4. Impute categorical features với mode
print(f"\n4️⃣ Xử lý Categorical Features (Impute với Mode):")
for col in categorical_features:
    if train_clean[col].isnull().sum() > 0:
        # Tính mode từ train set
        mode_values = train_clean[col].mode()
        mode_val = mode_values[0] if len(mode_values) > 0 else 'Unknown'
        
        # Impute cả train và test
        train_clean[col].fillna(mode_val, inplace=True)
        test_clean[col].fillna(mode_val, inplace=True)
        
        print(f"   ✅ {col}: imputed with mode = '{mode_val}'")

# Kiểm tra kết quả
remaining_train = train_clean.isnull().sum().sum()
remaining_test = test_clean.isnull().sum().sum()

print(f"\n🎯 KẾT QUẢ XỬ LÝ MISSING DATA:")
print(f"   ✅ Train: {remaining_train} missing values còn lại")
print(f"   ✅ Test: {remaining_test} missing values còn lại")

if remaining_train == 0 and remaining_test == 0:
    print(f"   🎉 Hoàn thành! Không còn missing data nào.")
else:
    print(f"   ⚠️ Vẫn còn missing data - cần kiểm tra lại!")

# ========================================
# BƯỚC 4: FEATURE ENGINEERING ĐƠN GIẢN
# ========================================
print("\n" + "=" * 60)
print("⚙️ BƯỚC 4: Feature Engineering Đơn Giản")
print("=" * 60)

def create_simple_features(df):
    """Tạo các features mới đơn giản và dễ hiểu"""
    df = df.copy()
    
    print(f"🔧 Đang tạo features mới...")
    
    # 1. Tuổi của tòa nhà
    current_year = 2024
    df['BuildingAge'] = current_year - df['YearBuilt']
    print(f"   ✅ BuildingAge = {current_year} - YearBuilt")
    
    # 2. Số năm kể từ lần renovation cuối
    df['YearsSinceRenovation'] = current_year - df['YearRenovated']
    # Nếu chưa renovation thì = tuổi tòa nhà
    df['YearsSinceRenovation'].fillna(df['BuildingAge'], inplace=True)
    print(f"   ✅ YearsSinceRenovation = {current_year} - YearRenovated")
    
    # 3. Tỷ lệ diện tích đất/sàn
    df['LandToFloorRatio'] = df['LandArea_sqm'] / (df['TotalFloorArea_sqm'] + 1e-8)
    print(f"   ✅ LandToFloorRatio = LandArea / TotalFloorArea")
    
    # 4. Diện tích trung bình mỗi phòng
    df['AreaPerRoom'] = df['TotalFloorArea_sqm'] / (df['RoomCount'] + 1e-8)
    print(f"   ✅ AreaPerRoom = TotalFloorArea / RoomCount")
    
    # 5. Tỷ lệ phòng ngủ
    df['BedroomRatio'] = df['BedroomCount'] / (df['RoomCount'] + 1e-8)
    print(f"   ✅ BedroomRatio = BedroomCount / RoomCount")
    
    # 6. Điểm tiện nghi (tổng số tiện ích cao cấp)
    luxury_features = ['HasGym', 'HasConcierge', 'HasLounge', 'HasGuestRoom']
    df['LuxuryScore'] = df[luxury_features].sum(axis=1)
    print(f"   ✅ LuxuryScore = sum of luxury amenities")
    
    # 7. Điểm tiện nghi cơ bản
    basic_amenities = ['SmartHome', 'CentralAC', 'FloorHeating', 'HasBalcony']
    df['BasicAmenityScore'] = df[basic_amenities].sum(axis=1)
    print(f"   ✅ BasicAmenityScore = sum of basic amenities")
    
    # 8. Có không gian ngoài trời
    outdoor_features = ['HasBalcony', 'HasRooftop', 'HasGarden']
    df['HasOutdoorSpace'] = (df[outdoor_features].sum(axis=1) > 0).astype(int)
    print(f"   ✅ HasOutdoorSpace = có ít nhất 1 outdoor feature")
    
    # 9. Nhà mới hay cũ
    df['IsNewBuilding'] = (df['BuildingAge'] <= 10).astype(int)
    print(f"   ✅ IsNewBuilding = 1 if BuildingAge <= 10")
    
    # 10. Điểm chất lượng tổng thể
    df['QualityScore'] = (df['ExteriorCondition'] + df['InteriorCondition']) / 2
    print(f"   ✅ QualityScore = trung bình ExteriorCondition và InteriorCondition")
    
    return df

# Áp dụng feature engineering
print(f"\n🚀 Áp dụng Feature Engineering:")
train_fe = create_simple_features(train_clean)
test_fe = create_simple_features(test_clean)

# Tính số features mới
original_features = train_clean.shape[1]
new_features_count = train_fe.shape[1] - original_features

print(f"\n📊 KẾT QUẢ FEATURE ENGINEERING:")
print(f"   📈 Features ban đầu: {original_features}")
print(f"   📈 Features sau khi tạo mới: {train_fe.shape[1]}")
print(f"   ✨ Số features mới: {new_features_count}")

# ========================================
# BƯỚC 5: CATEGORICAL ENCODING
# ========================================
print("\n" + "=" * 60)
print("🏷️ BƯỚC 5: Categorical Encoding")
print("=" * 60)

# Lấy danh sách categorical features
categorical_features_fe = train_fe.select_dtypes(include=['object']).columns.tolist()

print(f"📝 Cần encode {len(categorical_features_fe)} categorical features:")
for i, col in enumerate(categorical_features_fe, 1):
    unique_count = train_fe[col].nunique()
    print(f"   {i}. {col}: {unique_count} unique values")

# Label Encoding
print(f"\n🔄 Thực hiện Label Encoding:")
label_encoders = {}

for col in categorical_features_fe:
    print(f"   🏷️ Encoding {col}...")
    
    # Tạo label encoder
    le = LabelEncoder()
    
    # Kết hợp train và test để đảm bảo consistent encoding
    combined_values = pd.concat([train_fe[col], test_fe[col]]).astype(str)
    le.fit(combined_values)
    
    # Transform cả train và test
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col] = le.transform(test_fe[col].astype(str))
    
    # Lưu encoder để sau này có thể decode
    label_encoders[col] = le
    
    print(f"      ✅ {col}: {len(le.classes_)} categories encoded")

print(f"\n✅ Hoàn thành Categorical Encoding!")
print(f"   📊 Tất cả features giờ đã là numerical")

# ========================================
# BƯỚC 6: CHUẨN BỊ DỮ LIỆU CHO MODEL
# ========================================
print("\n" + "=" * 60)
print("📦 BƯỚC 6: Chuẩn bị dữ liệu cho Model")
print("=" * 60)

# Tách features và target
feature_columns = [col for col in train_fe.columns if col != 'Price_JPY']
X = train_fe[feature_columns]
y = train_fe['Price_JPY']
X_test = test_fe[feature_columns]

print(f"📊 KÍCH THƯỚC DỮ LIỆU:")
print(f"   🔢 Features (X): {X.shape}")
print(f"   🎯 Target (y): {y.shape}")
print(f"   🧪 Test features: {X_test.shape}")

# Kiểm tra không có missing data
assert X.isnull().sum().sum() == 0, "❌ Vẫn còn missing data trong X!"
assert X_test.isnull().sum().sum() == 0, "❌ Vẫn còn missing data trong X_test!"
print(f"   ✅ Không có missing data")

# Train-Validation Split
print(f"\n✂️ CHIA TRAIN-VALIDATION:")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"   📊 Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   📊 Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")

# Feature Scaling
print(f"\n⚖️ FEATURE SCALING:")
print(f"   🔧 Sử dụng StandardScaler (mean=0, std=1)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"   ✅ Đã scale train set (fit_transform)")
print(f"   ✅ Đã scale validation set (transform)")
print(f"   ✅ Đã scale test set (transform)")

# ========================================
# BƯỚC 7: TRAINING MODEL
# ========================================
print("\n" + "=" * 60)
print("🤖 BƯỚC 7: Training Linear Regression Model")
print("=" * 60)

print(f"🚀 TRAINING LINEAR REGRESSION:")
print(f"   📚 Algorithm: Ordinary Least Squares")
print(f"   📊 Features: {X_train_scaled.shape[1]}")
print(f"   📊 Training samples: {X_train_scaled.shape[0]:,}")

# Khởi tạo và train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f"   ✅ Training hoàn thành!")

# Thông tin về model
print(f"\n📋 THÔNG TIN MODEL:")
print(f"   🔢 Coefficients: {len(model.coef_):,}")
print(f"   🎯 Intercept: {model.intercept_:,.0f}")

# Top 5 features quan trọng nhất (theo absolute coefficients)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n⭐ TOP 5 FEATURES QUAN TRỌNG NHẤT:")
for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Coefficient']:,.2f}")

# ========================================
# BƯỚC 8: ĐÁNH GIÁ MODEL
# ========================================
print("\n" + "=" * 60)
print("📊 BƯỚC 8: Đánh giá Model")
print("=" * 60)

def calculate_metrics(y_true, y_pred, dataset_name):
    """Tính toán và hiển thị các metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n📈 {dataset_name.upper()} PERFORMANCE:")
    print(f"   🎯 RMSE: {rmse:,.0f} JPY")
    print(f"   🎯 MAE: {mae:,.0f} JPY")
    print(f"   🎯 R² Score: {r2:.4f}")
    
    # Giải thích R² score
    if r2 >= 0.8:
        performance = "Xuất sắc"
    elif r2 >= 0.6:
        performance = "Tốt"
    elif r2 >= 0.4:
        performance = "Trung bình"
    else:
        performance = "Cần cải thiện"
    
    print(f"   💬 Đánh giá: {performance}")
    
    return rmse, mae, r2

# Dự đoán và đánh giá
print(f"🔮 THỰC HIỆN DỰ ĐOÁN:")
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)

# Tính metrics
train_rmse, train_mae, train_r2 = calculate_metrics(y_train, y_train_pred, "training")
val_rmse, val_mae, val_r2 = calculate_metrics(y_val, y_val_pred, "validation")

# Kiểm tra overfitting/underfitting
print(f"\n🔍 PHÂN TÍCH OVERFITTING:")
r2_diff = train_r2 - val_r2
print(f"   📊 Train R²: {train_r2:.4f}")
print(f"   📊 Validation R²: {val_r2:.4f}")
print(f"   📊 Chênh lệch: {r2_diff:.4f}")

if r2_diff < 0.05:
    print(f"   ✅ Model ổn định (không overfitting)")
elif r2_diff < 0.1:
    print(f"   ⚠️ Có dấu hiệu overfitting nhẹ")
else:
    print(f"   ❌ Overfitting nghiêm trọng")

# Visualization của predictions
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue', label='Train')
plt.scatter(y_val, y_val_pred, alpha=0.5, color='red', label='Validation')
min_price = min(y_train.min(), y_val.min())
max_price = max(y_train.max(), y_val.max())
plt.plot([min_price, max_price], [min_price, max_price], 'k--', alpha=0.8)
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.title('Predicted vs Actual Prices')
plt.legend()

plt.subplot(1, 2, 2)
residuals_train = y_train - y_train_pred
residuals_val = y_val - y_val_pred
plt.scatter(y_train_pred, residuals_train, alpha=0.5, color='blue', label='Train')
plt.scatter(y_val_pred, residuals_val, alpha=0.5, color='red', label='Validation')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.8)
plt.xlabel('Giá dự đoán')
plt.ylabel('Residuals (Thực tế - Dự đoán)')
plt.title('Residual Plot')
plt.legend()

plt.tight_layout()
plt.show()

# ========================================
# BƯỚC 9: DỰ ĐOÁN TRÊN TEST SET
# ========================================
print("\n" + "=" * 60)
print("🔮 BƯỚC 9: Dự đoán trên Test Set")
print("=" * 60)

print(f"🚀 ĐANG THỰC HIỆN DỰ ĐOÁN...")
test_predictions = model.predict(X_test_scaled)

# Đảm bảo predictions không âm (giá nhà không thể âm)
negative_count = (test_predictions < 0).sum()
if negative_count > 0:
    print(f"   ⚠️ Phát hiện {negative_count} dự đoán âm - đã chuyển thành 0")
    test_predictions = np.maximum(test_predictions, 0)

print(f"   ✅ Hoàn thành dự đoán cho {len(test_predictions):,} mẫu")

# Thống kê dự đoán
print(f"\n📊 THỐNG KÊ DỰ ĐOÁN:")
print(f"   💰 Giá thấp nhất: {test_predictions.min():,.0f} JPY")
print(f"   💰 Giá cao nhất: {test_predictions.max():,.0f} JPY")
print(f"   💰 Giá trung bình: {test_predictions.mean():,.0f} JPY")
print(f"   💰 Giá trung vị: {np.median(test_predictions):,.0f} JPY")
print(f"   📈 Độ lệch chuẩn: {test_predictions.std():,.0f} JPY")

# So sánh với train data
print(f"\n🔍 SO SÁNH VỚI TRAIN DATA:")
print(f"   📊 Train - Giá trung bình: {y.mean():,.0f} JPY")
print(f"   📊 Test - Giá dự đoán TB: {test_predictions.mean():,.0f} JPY")
print(f"   📊 Chênh lệch: {abs(y.mean() - test_predictions.mean()):,.0f} JPY")

# ========================================
# BƯỚC 10: TẠO SUBMISSION FILE
# ========================================
print("\n" + "=" * 60)
print("💾 BƯỚC 10: Tạo Submission File")
print("=" * 60)

# Tạo submission dataframe
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'Price_JPY': test_predictions
})

# Kiểm tra format
print(f"📋 KIỂM TRA SUBMISSION:")
print(f"   📊 Số dòng: {len(submission):,}")
print(f"   📊 Columns: {list(submission.columns)}")
print(f"   📊 ID range: {submission['ID'].min()} - {submission['ID'].max()}")

# Hiển thị một vài dòng đầu
print(f"\n👀 XEM TRƯỚC SUBMISSION:")
print(submission.head())

# Lưu file
submission_filename = 'house_price_submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"\n💾 ĐÃ LUU SUBMISSION:")
print(f"   📁 File: {submission_filename}")
print(f"   ✅ Format: CSV không có index")
print(f"   📊 {len(submission):,} dự đoán đã được lưu")

# ========================================
# BƯỚC 11: TÓM TẮT VÀ KẾT LUẬN
# ========================================
print("\n" + "=" * 60)
print("🎯 TÓM TẮT VÀ KẾT LUẬN")
print("=" * 60)

print(f"\n📊 THÔNG TIN DATASET:")
print(f"   🏠 Train samples: {len(train_fe):,}")
print(f"   🧪 Test samples: {len(test_fe):,}")
print(f"   🔢 Features sử dụng: {len(feature_columns)}")
print(f"   ✨ Features tạo mới: {new_features_count}")

print(f"\n🔧 XỬ LÝ DỮ LIỆU:")
print(f"   ✅ Missing data đã được xử lý hoàn toàn")
print(f"   ✅ Categorical features đã được encode")
print(f"   ✅ Features đã được scaled")
print(f"   ✅ Outliers đã được xem xét")

print(f"\n🤖 MODEL PERFORMANCE:")
print(f"   📈 Validation RMSE: {val_rmse:,.0f} JPY")
print(f"   📈 Validation MAE: {val_mae:,.0f} JPY")
print(f"   📈 Validation R²: {val_r2:.4f}")
print(f"   🎯 Model: Linear Regression")

print(f"\n💡 CÁC BƯỚC TIẾP THEO ĐỂ CẢI THIỆN:")
print(f"   🔮 Thử các algorithm khác:")
print(f"      - Random Forest Regressor")
print(f"      - Gradient Boosting (XGBoost, LightGBM)")
print(f"      - Support Vector Regression")
print(f"   ⚙️ Feature Engineering nâng cao:")
print(f"      - Polynomial features")
print(f"      - Interaction terms")
print(f"      - Target encoding cho categorical")
print(f"   🎛️ Hyperparameter tuning")
print(f"   🔄 Cross-validation để đánh giá robust hơn")
print(f"   📊 Feature selection để loại bỏ features không quan trọng")

print(f"\n🎉 HOÀN THÀNH PIPELINE!")
print(f"📁 File submission: {submission_filename}")
print("=" * 60)

# Lưu feature importance để tham khảo
feature_importance.to_csv('feature_importance.csv', index=False)
print(f"💾 Đã lưu feature importance vào: feature_importance.csv")
