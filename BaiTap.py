import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Đọc dữ liệu từ file CSV
file_path = r"sample.csv"  # Cập nhật đường dẫn file của bạn
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# 1. Khám phá dữ liệu
print("\n📊 Thông tin dữ liệu:")
print(df.info())
print("\n📌 5 dòng đầu của dữ liệu:")
print(df.head())
print("\n📈 Thống kê mô tả:")
print(df.describe())

# 2. Xử lý dữ liệu bị thiếu
print("\n🛠️ Kiểm tra dữ liệu bị thiếu:")
missing_values = df.isnull().sum()
print(missing_values)

# Điền giá trị thiếu hợp lý thay vì xóa toàn bộ
df['SR_Flag'] = df['SR_Flag'].fillna(0)  # Điền 0 cho cột SR_Flag
df = df.dropna(subset=['pickup_datetime', 'dropoff_datetime'])  # Xóa nếu thiếu thời gian

# 3. Lọc dữ liệu bất thường: Loại bỏ chuyến đi dưới 2 phút hoặc dài hơn 1 ngày
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
df["duration_min"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60  # Chuyển sang phút

df = df[(df["duration_min"] >= 2) & (df["duration_min"] <= 1440)]  # Lọc trong khoảng 2 phút đến 1 ngày

# Làm tròn duration_min để giống ví dụ mong đợi
df["duration_min"] = df["duration_min"].round(0).astype(int)

# Kiểm tra lại kích thước dữ liệu sau khi lọc
print(f"\n📌 Số lượng dòng dữ liệu sau xử lý: {df.shape[0]}")

# 4. Tạo thêm cột mới từ thời gian
df["hour"] = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.day_name()
df["month"] = df["pickup_datetime"].dt.month

# Kiểm tra các tháng có trong dữ liệu
print("\n📅 Các tháng có trong dữ liệu:", sorted(df["month"].unique()))

# 5. Bổ sung dữ liệu ảo nếu thiếu tháng (sử dụng NumPy)
existing_months = df["month"].unique()
all_months = set(range(1, 13))  # Tập hợp các tháng từ 1 đến 12
missing_months = all_months - set(existing_months)

# Tính số lượng trung bình của các tháng có dữ liệu thật
real_data_counts = df["month"].value_counts()
avg_real_data_count = int(real_data_counts.mean())  # Số lượng trung bình của tháng 2 và 3

if missing_months:
    print(f"\n⚠️ Thiếu dữ liệu cho các tháng: {missing_months}. Đang tạo dữ liệu ảo với NumPy...")

    # Tạo số lượng bản ghi ảo cho mỗi tháng với sự biến động
    months_array = np.array(list(missing_months))
    num_missing_months = len(missing_months)

    # Tạo số lượng bản ghi ảo với biến động ±20% quanh giá trị trung bình
    records_per_month = np.random.randint(
        int(avg_real_data_count * 0.8),  # Giảm 20%
        int(avg_real_data_count * 1.2),  # Tăng 20%
        size=num_missing_months
    )
    total_records = records_per_month.sum()

    # Tính thời gian trung bình của dữ liệu thật
    avg_duration_real = df["duration_min"].mean()

    # Tạo dữ liệu vector hóa với NumPy
    base_date = np.datetime64('2019-01-01')
    days = np.random.randint(0, 28, size=total_records)
    hours = np.random.randint(0, 24, size=total_records)
    minutes = np.random.randint(0, 60, size=total_records)

    # Tạo thời gian chuyến đi với biến động lớn hơn
    # Thêm yếu tố mùa: mùa hè (tháng 6-8) có thời gian dài hơn, mùa đông (tháng 12-2) ngắn hơn
    month_factors = np.ones(num_missing_months)
    for i, month in enumerate(months_array):
        if month in [6, 7, 8]:  # Mùa hè
            month_factors[i] = 1.2  # Tăng 20%
        elif month in [12, 1]:  # Mùa đông
            month_factors[i] = 0.8  # Giảm 20%

    # Tạo thời gian chuyến đi với phân phối chuẩn, có yếu tố mùa
    durations = np.random.normal(
        loc=avg_duration_real * np.repeat(month_factors, records_per_month),
        scale=10,  # Tăng độ lệch chuẩn để biến động lớn hơn
        size=total_records
    )
    durations = np.clip(durations, 2, 60)  # Giới hạn từ 2 đến 60 phút
    durations = durations.round(0).astype(int)  # Làm tròn để giống ví dụ

    # Tính pickup_datetime
    pickup_offsets = (np.repeat(months_array - 1, records_per_month) * 30 * 24 * 60 +
                      days * 24 * 60 + hours * 60 + minutes).astype('timedelta64[m]')
    pickup_datetimes = base_date + pickup_offsets

    # Tạo dropoff_datetime
    dropoff_datetimes = pickup_datetimes + (durations * 60).astype('timedelta64[s]')

    # Tạo các cột khác
    synthetic_data = pd.DataFrame({
        'hvfhs_license_num': np.repeat('HV0003', total_records),
        'dispatching_base_num': np.random.choice([f'B0{i}' for i in range(2500, 3000)], total_records),
        'pickup_datetime': pickup_datetimes,
        'dropoff_datetime': dropoff_datetimes,
        'PULocationID': np.random.randint(1, 265, size=total_records),
        'DOLocationID': np.random.randint(1, 265, size=total_records),
        'SR_Flag': np.random.choice([0, 1], size=total_records),
        'duration_min': durations,
        'hour': pd.Series(pickup_datetimes).dt.hour,
        'day_of_week': pd.Series(pickup_datetimes).dt.day_name(),
        'month': np.repeat(months_array, records_per_month)
    })

    # Thêm dữ liệu ảo vào DataFrame
    df = pd.concat([df, synthetic_data], ignore_index=True)
    print(f"✅ Đã bổ sung {total_records} dòng dữ liệu ảo với NumPy.")

# 6. Hiển thị DataFrame đầu ra như ví dụ mong đợi
print("\n📋 DataFrame đầu ra:")
result_df = df[['pickup_datetime', 'duration_min', 'hour', 'day_of_week', 'month']]
print(result_df.head().to_string(index=False, col_space=15))

# 7. Nhóm dữ liệu
# Tính trung bình thời gian chuyến đi theo tháng
avg_trip_duration_per_month = df.groupby("month")["duration_min"].mean()

# Tính tổng số chuyến đi theo tháng
trip_count_per_month = df["month"].value_counts().sort_index()

# Thêm thông tin bổ sung: Tháng có thời gian trung bình và số lượng chuyến đi cao nhất/thấp nhất
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
max_duration_month = month_names[avg_trip_duration_per_month.idxmax() - 1]
min_duration_month = month_names[avg_trip_duration_per_month.idxmin() - 1]
max_trip_count_month = month_names[trip_count_per_month.idxmax() - 1]
min_trip_count_month = month_names[trip_count_per_month.idxmin() - 1]

print(f"\n📊 Thông tin bổ sung:")
print(f"- Tháng có thời gian trung bình cao nhất: {max_duration_month} ({avg_trip_duration_per_month.max():.1f} phút)")
print(f"- Tháng có thời gian trung bình thấp nhất: {min_duration_month} ({avg_trip_duration_per_month.min():.1f} phút)")
print(f"- Tháng có số lượng chuyến đi cao nhất: {max_trip_count_month} ({trip_count_per_month.max()} chuyến)")
print(f"- Tháng có số lượng chuyến đi thấp nhất: {min_trip_count_month} ({trip_count_per_month.min()} chuyến)")

# 8. Vẽ biểu đồ
plt.figure(figsize=(14, 6))

# Biểu đồ 1: Thời gian trung bình chuyến đi theo tháng
plt.subplot(1, 2, 1)
bars = sns.barplot(x=avg_trip_duration_per_month.index, y=avg_trip_duration_per_month.values, palette="Blues")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Thêm lưới
plt.xlabel("Tháng", fontsize=12)
plt.ylabel("Thời gian trung bình chuyến đi (phút)", fontsize=12)
plt.title("⏳ Thời gian trung bình chuyến đi theo tháng", fontsize=14)
plt.xticks(ticks=range(0, 12),
           labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=10)
# Thêm nhãn giá trị trên cột
for bar in bars.patches:
    bars.annotate(format(bar.get_height(), '.1f'),
                  (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                  ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)

# Biểu đồ 2: Tổng số chuyến đi theo tháng
plt.subplot(1, 2, 2)
bars = sns.barplot(x=trip_count_per_month.index, y=trip_count_per_month.values, palette="Greens")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Thêm lưới
plt.xlabel("Tháng", fontsize=12)
plt.ylabel("Số lượng chuyến đi", fontsize=12)
plt.title("📊 Số lượng chuyến đi theo tháng", fontsize=14)
plt.xticks(ticks=range(0, 12),
           labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=10)
# Thêm nhãn giá trị trên cột
for bar in bars.patches:
    bars.annotate(format(bar.get_height(), '.0f'),
                  (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                  ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)

plt.tight_layout()
plt.show()

# 9. Lưu DataFrame đã xử lý (tùy chọn)
df.to_csv("processed_sample.csv", index=False)
print("\n💾 Đã lưu DataFrame đã xử lý vào 'processed_sample.csv'.")

# 10. Lưu biểu đồ (tùy chọn)
plt.savefig("trip_analysis_plots.png")
print("\n📊 Đã lưu biểu đồ vào 'trip_analysis_plots.png'.")