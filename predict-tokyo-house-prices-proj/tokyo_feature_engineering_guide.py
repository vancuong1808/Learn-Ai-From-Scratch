"""
Tokyo House Price Prediction - Feature Engineering Guide
========================================================

Hướng dẫn chi tiết về Feature Engineering cho dự án dự đoán giá nhà Tokyo
Dành cho người mới học Machine Learning

Author: AI Assistant
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class TokyoHouseFeatureEngineering:
    def __init__(self):
        """Khởi tạo class feature engineering"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """1. Load dữ liệu"""
        print("=" * 60)
        print("BƯỚC 1: LOAD DỮ LIỆU")
        print("=" * 60)
        
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        
        print(f"Train data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        print(f"Target variable: Price_JPY")
        
        # Hiển thị thông tin cơ bản
        print(f"\nGiá trị Price_JPY:")
        print(f"Min: {self.train_data['Price_JPY'].min():,.0f} JPY")
        print(f"Max: {self.train_data['Price_JPY'].max():,.0f} JPY")
        print(f"Mean: {self.train_data['Price_JPY'].mean():,.0f} JPY")
        
        return self.train_data, self.test_data
    
    def analyze_missing_data(self):
        """2. Phân tích missing data và phân loại MCAR/MAR/MNAR"""
        print("\n" + "=" * 60)
        print("BƯỚC 2: PHÂN TÍCH MISSING DATA")
        print("=" * 60)
        
        # Kiểm tra missing values
        missing_train = self.train_data.isnull().sum()
        missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
        
        if len(missing_train) > 0:
            print("Missing values trong train data:")
            for col, count in missing_train.items():
                pct = (count / len(self.train_data)) * 100
                print(f"{col}: {count} ({pct:.1f}%)")
        else:
            print("Không có missing values trong train data!")
            
        # Kiểm tra YearRenovated (0 = never renovated)
        zero_renovated = (self.train_data['YearRenovated'] == 0).sum()
        print(f"\nYearRenovated = 0 (never renovated): {zero_renovated} properties")
        
        return missing_train
    
    def create_basic_features(self):
        """3. Tạo các features cơ bản"""
        print("\n" + "=" * 60)
        print("BƯỚC 3: TẠO CÁC FEATURES CƠ BẢN")
        print("=" * 60)
        
        # Copy data để tránh thay đổi dữ liệu gốc
        train_fe = self.train_data.copy()
        test_fe = self.test_data.copy()
        
        print("3.1. Age-related features (Các đặc trưng về tuổi)")
        for df, name in [(train_fe, 'train'), (test_fe, 'test')]:
            # Tuổi của building
            current_year = 2024
            df['BuildingAge'] = current_year - df['YearBuilt']
            
            # Xử lý YearRenovated (0 = never renovated)
            df['YearRenovated_filled'] = df['YearRenovated'].replace(0, np.nan)
            df['YearsSinceRenovation'] = current_year - df['YearRenovated_filled']
            df['YearsSinceRenovation'].fillna(df['BuildingAge'], inplace=True)
            
            # Binary: đã renovation hay chưa
            df['HasBeenRenovated'] = (df['YearRenovated'] > 0).astype(int)
            
        print("✓ BuildingAge: Tuổi của building")
        print("✓ YearsSinceRenovation: Số năm kể từ lần renovation cuối")
        print("✓ HasBeenRenovated: Đã từng được renovation (1/0)")
        
        print("\n3.2. Area and Space features (Các đặc trưng về diện tích)")
        for df in [train_fe, test_fe]:
            # Tỷ lệ diện tích
            df['LandToFloorRatio'] = df['LandArea_sqm'] / (df['TotalFloorArea_sqm'] + 1e-8)
            df['FloorAreaPerRoom'] = df['TotalFloorArea_sqm'] / (df['RoomCount'] + 1e-8)
            df['BedroomRatio'] = df['BedroomCount'] / (df['RoomCount'] + 1e-8)
            
            # Efficiency score
            df['SpaceEfficiency'] = df['TotalFloorArea_sqm'] / (df['LandArea_sqm'] + 1e-8)
            
        print("✓ LandToFloorRatio: Tỷ lệ đất/sàn")
        print("✓ FloorAreaPerRoom: Diện tích sàn trung bình mỗi phòng")
        print("✓ BedroomRatio: Tỷ lệ phòng ngủ/tổng số phòng")
        print("✓ SpaceEfficiency: Hiệu quả sử dụng không gian")
        
        self.train_fe = train_fe
        self.test_fe = test_fe
        
        return train_fe, test_fe
    
    def create_location_features(self):
        """4. Tạo location features"""
        print("\n" + "=" * 60)
        print("BƯỚC 4: TẠO LOCATION FEATURES")
        print("=" * 60)
        
        print("4.1. Distance-based features")
        for df in [self.train_fe, self.test_fe]:
            # Composite location score
            df['LocationConvenience'] = (
                (1000 - np.minimum(df['DistanceToStation_m'], 1000)) / 1000 * 0.5 +
                (1500 - np.minimum(df['DistanceToSchool_m'], 1500)) / 1500 * 0.3 +
                (2000 - np.minimum(df['DistanceToHospital_m'], 2000)) / 2000 * 0.2
            )
            
            # Transportation score
            df['TransportScore'] = df['NumberOfTrainLines_at_NearestStation'] / (df['TravelTimeToMajorHub_min'] + 1e-8)
            
            # Log transform distances (giảm skewness)
            df['DistanceToStation_log'] = np.log1p(df['DistanceToStation_m'])
            df['DistanceToSchool_log'] = np.log1p(df['DistanceToSchool_m'])
            df['DistanceToHospital_log'] = np.log1p(df['DistanceToHospital_m'])
        
        print("✓ LocationConvenience: Điểm tổng hợp về vị trí thuận lợi")
        print("✓ TransportScore: Điểm về giao thông công cộng")
        print("✓ Distance_log: Log transform để giảm skewness")
        
        print("\n4.2. Ward-based features")
        # Premium wards (thường có giá cao)
        premium_wards = ['Minato', 'Shibuya', 'Chuo']
        for df in [self.train_fe, self.test_fe]:
            df['IsPremiumWard'] = df['Ward'].isin(premium_wards).astype(int)
        
        print("✓ IsPremiumWard: Có phải ward cao cấp không (Minato, Shibuya, Chuo)")
        
    def create_amenity_features(self):
        """5. Tạo amenity features"""
        print("\n" + "=" * 60)
        print("BƯỚC 5: TẠO AMENITY FEATURES")
        print("=" * 60)
        
        for df in [self.train_fe, self.test_fe]:
            # Luxury amenities
            luxury_features = ['HasGym', 'HasConcierge', 'HasLounge', 'HasGuestRoom']
            df['LuxuryScore'] = df[luxury_features].sum(axis=1)
            
            # Comfort features
            comfort_features = ['SmartHome', 'CentralAC', 'FloorHeating']
            df['ComfortScore'] = df[comfort_features].sum(axis=1)
            
            # Outdoor features
            outdoor_features = ['HasBalcony', 'HasRooftop', 'HasGarden']
            df['OutdoorScore'] = df[outdoor_features].sum(axis=1)
            df['HasOutdoorSpace'] = (df['OutdoorScore'] > 0).astype(int)
            
            # Parking features
            df['HasParkingAdvanced'] = (df['HasParking'] == 1).astype(int)
            
        print("✓ LuxuryScore: Tổng điểm tiện ích cao cấp")
        print("✓ ComfortScore: Tổng điểm tiện ích thoải mái")
        print("✓ OutdoorScore: Tổng điểm không gian ngoài trời")
        print("✓ HasOutdoorSpace: Có không gian ngoài trời hay không")
        
    def create_quality_features(self):
        """6. Tạo quality features"""
        print("\n" + "=" * 60)
        print("BƯỚC 6: TẠO QUALITY FEATURES")
        print("=" * 60)
        
        for df in [self.train_fe, self.test_fe]:
            # Overall quality score
            df['OverallQuality'] = (
                df['ExteriorCondition'] * 0.3 +
                df['InteriorCondition'] * 0.4 +
                df['MaintenanceScore'] / 10 * 0.3
            )
            
            # Safety score
            df['SafetyScore'] = (
                df['EarthquakeResistant'] * 0.4 +
                df['EarthquakeSafetyScore'] / 10 * 0.3 +
                df['LegalComplianceScore'] / 10 * 0.3
            )
            
            # Energy efficiency (convert A,B,C,D,E to numbers)
            energy_map = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
            df['EnergyEfficiencyNumeric'] = df['EnergyEfficiencyRating'].map(energy_map)
            
        print("✓ OverallQuality: Điểm chất lượng tổng thể")
        print("✓ SafetyScore: Điểm an toàn")
        print("✓ EnergyEfficiencyNumeric: Hiệu quả năng lượng (số)")
        
    def create_market_features(self):
        """7. Tạo market features"""
        print("\n" + "=" * 60)
        print("BƯỚC 7: TẠO MARKET FEATURES")
        print("=" * 60)
        
        for df in [self.train_fe, self.test_fe]:
            # Market activity
            df['MarketActivity'] = df['NumberOfListings_in_Ward_Last3Months'] / (df['AverageDaysOnMarket_in_Ward'] + 1e-8)
            
            # Income ratio
            median_income = df['NeighborhoodAvgIncome_JPY'].median()
            df['IncomeRatio'] = df['NeighborhoodAvgIncome_JPY'] / median_income
            
            # Population features
            df['PopulationGrowthPositive'] = (df['PopulationGrowthRate'] > 0).astype(int)
            df['HighForeignerRatio'] = (df['ForeignerPopulationRatio'] > 0.2).astype(int)
            
        print("✓ MarketActivity: Hoạt động thị trường")
        print("✓ IncomeRatio: Tỷ lệ thu nhập so với median")
        print("✓ PopulationGrowthPositive: Tăng trưởng dân số dương")
        print("✓ HighForeignerRatio: Tỷ lệ người nước ngoài cao")
        
    def create_interaction_features(self):
        """8. Tạo interaction features"""
        print("\n" + "=" * 60)
        print("BƯỚC 8: TẠO INTERACTION FEATURES")
        print("=" * 60)
        
        for df in [self.train_fe, self.test_fe]:
            # Luxury trong premium ward
            df['LuxuryInPremiumWard'] = df['IsPremiumWard'] * df['LuxuryScore']
            
            # Quality vs Age
            df['QualityAgeInteraction'] = df['OverallQuality'] * (1 / (df['BuildingAge'] + 1))
            
            # Location vs Luxury
            df['LocationLuxuryScore'] = df['LocationConvenience'] * df['LuxuryScore']
            
            # New building với high quality
            df['NewHighQuality'] = ((df['BuildingAge'] <= 10) & (df['OverallQuality'] >= 4)).astype(int)
            
        print("✓ LuxuryInPremiumWard: Luxury trong khu vực cao cấp")
        print("✓ QualityAgeInteraction: Tương tác giữa chất lượng và tuổi")
        print("✓ LocationLuxuryScore: Tương tác vị trí và luxury")
        print("✓ NewHighQuality: Building mới với chất lượng cao")
        
    def handle_categorical_features(self):
        """9. Xử lý categorical features"""
        print("\n" + "=" * 60)
        print("BƯỚC 9: XỬ LÝ CATEGORICAL FEATURES")
        print("=" * 60)
        
        # Lấy danh sách categorical columns
        categorical_cols = self.train_fe.select_dtypes(include=['object']).columns.tolist()
        print(f"Categorical columns: {categorical_cols}")
        
        # Label encoding
        for col in categorical_cols:
            le = LabelEncoder()
            
            # Combine train and test để đảm bảo consistent encoding
            combined_data = pd.concat([self.train_fe[col], self.test_fe[col]]).astype(str)
            le.fit(combined_data)
            
            # Transform
            self.train_fe[col] = le.transform(self.train_fe[col].astype(str))
            self.test_fe[col] = le.transform(self.test_fe[col].astype(str))
            
            # Store encoder
            self.label_encoders[col] = le
            
        print("✓ Đã encode tất cả categorical features bằng LabelEncoder")
        
    def feature_selection_and_preparation(self):
        """10. Feature selection và preparation"""
        print("\n" + "=" * 60)
        print("BƯỚC 10: FEATURE SELECTION & PREPARATION")
        print("=" * 60)
        
        # Loại bỏ các columns không cần thiết
        columns_to_drop = ['YearRenovated_filled']  # Helper column
        
        # Chuẩn bị features
        feature_cols = [col for col in self.train_fe.columns 
                       if col not in ['Price_JPY'] + columns_to_drop]
        
        X = self.train_fe[feature_cols]
        y = self.train_fe['Price_JPY']
        X_test = self.test_fe[feature_cols]
        
        print(f"Total features: {len(feature_cols)}")
        print(f"Train shape: {X.shape}")
        print(f"Test shape: {X_test.shape}")
        
        # Hiển thị một số features quan trọng
        important_features = [
            'TotalFloorArea_sqm', 'LandArea_sqm', 'BuildingAge', 
            'LocationConvenience', 'LuxuryScore', 'OverallQuality',
            'IsPremiumWard', 'SafetyScore'
        ]
        
        print(f"\nMột số features quan trọng:")
        for feat in important_features:
            if feat in feature_cols:
                print(f"✓ {feat}")
        
        return X, y, X_test, feature_cols
    
    def train_and_evaluate_model(self, X, y, X_test):
        """11. Train model và đánh giá"""
        print("\n" + "=" * 60)
        print("BƯỚC 11: TRAIN MODEL VÀ ĐÁNH GIÁ")
        print("=" * 60)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Linear Regression
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"Training RMSE: {train_rmse:,.0f}")
        print(f"Validation RMSE: {val_rmse:,.0f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_
        })
        feature_importance['abs_coefficient'] = np.abs(feature_importance['coefficient'])
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        print(f"\nTop 10 quan trọng nhất (theo coefficient):")
        for i, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['coefficient']:.2e}")
        
        # Make predictions on test set
        test_predictions = model.predict(X_test_scaled)
        
        return model, test_predictions, feature_importance
    
    def create_submission(self, test_predictions):
        """12. Tạo submission file"""
        print("\n" + "=" * 60)
        print("BƯỚC 12: TẠO SUBMISSION FILE")
        print("=" * 60)
        
        submission = pd.DataFrame({
            'ID': self.test_data['ID'],
            'Price_JPY': test_predictions
        })
        
        submission.to_csv('submission_with_feature_engineering.csv', index=False)
        
        print(f"✓ Đã tạo submission file: submission_with_feature_engineering.csv")
        print(f"Prediction statistics:")
        print(f"Min: {test_predictions.min():,.0f}")
        print(f"Max: {test_predictions.max():,.0f}")
        print(f"Mean: {test_predictions.mean():,.0f}")
        
        return submission
    
    def run_complete_pipeline(self):
        """Chạy toàn bộ pipeline"""
        print("TOKYO HOUSE PRICE PREDICTION - FEATURE ENGINEERING GUIDE")
        print("=" * 80)
        print("Hướng dẫn chi tiết về Feature Engineering cho người mới học")
        print("=" * 80)
        
        # Chạy từng bước
        self.load_data()
        self.analyze_missing_data()
        self.create_basic_features()
        self.create_location_features()
        self.create_amenity_features()
        self.create_quality_features()
        self.create_market_features()
        self.create_interaction_features()
        self.handle_categorical_features()
        
        X, y, X_test, feature_cols = self.feature_selection_and_preparation()
        model, test_predictions, feature_importance = self.train_and_evaluate_model(X, y, X_test)
        submission = self.create_submission(test_predictions)
        
        print("\n" + "=" * 80)
        print("HOÀN THÀNH PIPELINE FEATURE ENGINEERING!")
        print("=" * 80)
        print("\nTÓM TẮT CÁC FEATURES ĐÃ TẠO:")
        print("1. Age Features: BuildingAge, YearsSinceRenovation, HasBeenRenovated")
        print("2. Area Features: LandToFloorRatio, FloorAreaPerRoom, SpaceEfficiency")
        print("3. Location Features: LocationConvenience, TransportScore, IsPremiumWard")
        print("4. Amenity Features: LuxuryScore, ComfortScore, OutdoorScore")
        print("5. Quality Features: OverallQuality, SafetyScore, EnergyEfficiencyNumeric")
        print("6. Market Features: MarketActivity, IncomeRatio, PopulationGrowthPositive")
        print("7. Interaction Features: LuxuryInPremiumWard, QualityAgeInteraction")
        
        return model, submission, feature_importance

# Chạy pipeline
if __name__ == "__main__":
    fe = TokyoHouseFeatureEngineering()
    model, submission, feature_importance = fe.run_complete_pipeline()
    
    # Hiển thị feature importance
    print(f"\nFeature Importance (Top 15):")
    print(feature_importance.head(15)[['feature', 'coefficient']])
