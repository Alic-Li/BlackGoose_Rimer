import pandas as pd
from sklearn.model_selection import train_test_split

def print_csv_information(file_path):
    df = pd.read_csv(file_path)
    
    column_names = df.columns.tolist()
    print("CSV文件的列名如下:")
    for column in column_names:
        print(column)
    print("数据的前5行:")
    print(df.head())
    print("\n")
    
    print("数据的基本信息:")
    print(df.info())
    print("\n")
    
    print("数据的描述性统计信息:")
    print(df.describe(include='all'))
    print("\n")

    print("缺失值情况:")
    print(df.isnull().sum())
    print("\n")
    
    print("数据的列名:")
    print(df.columns.tolist())
    print("\n")
    
    print("数据的形状（行数, 列数):")
    print(df.shape)
    print("\n")

def split_dataset(df, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42):
    # 先将数据集分为训练集和剩余部分（验证集+测试集）
    train_df, remaining_df = train_test_split(df, train_size=train_size, random_state=random_state)
    # 再将剩余部分分为验证集和测试集
    val_df, test_df = train_test_split(remaining_df, test_size=test_size/(val_size+test_size), random_state=random_state)
    return train_df, val_df, test_df

def save_datasets(train_df, val_df, test_df, train_file_path, val_file_path, test_file_path):
    train_df.to_csv(train_file_path, index=False)
    val_df.to_csv(val_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)
    print(f"训练集已保存到 {train_file_path}")
    print(f"验证集已保存到 {val_file_path}")
    print(f"测试集已保存到 {test_file_path}")

file_path = '/home/alic-li/PowerForecasting/dataset/electricity/electricity.csv'
df = pd.read_csv(file_path)

# 打印CSV信息
print_csv_information(file_path)

# 切分数据集
train_df, val_df, test_df = split_dataset(df)

# 输出切分后的数据集信息
print("训练集的形状（行数, 列数):")
print(train_df.shape)
print("\n")
print("验证集的形状（行数, 列数):")
print(val_df.shape)
print("\n")
print("测试集的形状（行数, 列数):")
print(test_df.shape)
print("\n")

# 保存数据集
train_file_path = '/home/alic-li/PowerForecasting/dataset/electricity/train_electricity.csv'
val_file_path = '/home/alic-li/PowerForecasting/dataset/electricity/val_electricity.csv'
test_file_path = '/home/alic-li/PowerForecasting/dataset/electricity/test_electricity.csv'
save_datasets(train_df, val_df, test_df, train_file_path, val_file_path, test_file_path)