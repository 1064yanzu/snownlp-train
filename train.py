# -*- coding: utf-8 -*-
import pandas as pd
import os
import time
import sys
import shutil
from snownlp.sentiment import Sentiment
from glob import glob
from tqdm import tqdm  # 进度条库

# 安装依赖（首次运行时自动安装）
try:
    from tqdm import tqdm
except ImportError:
    print("安装 tqdm 进度条库...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

try:
    from snownlp.sentiment import Sentiment
except ImportError:
    print("安装 snownlp 库...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "snownlp"])
    from snownlp.sentiment import Sentiment

# ================== 数据加载函数 ==================
def load_multiple_csvs(filepaths, text_col='content', label_col='sentiment'):
    """加载多个CSV文件并合并数据，过滤掉中性样本，带进度条"""
    label_mapping = {
        '负面': 0, '消极': 0, '负向': 0, 'negative': 0,
        '正面': 1, '积极': 1, '正向': 1, 'positive': 1,
        '中性': 1  # 中性样本分配为正面
    }

    all_texts, all_labels = [], []
    print(f"开始加载 {len(filepaths)} 个数据文件...")

    for path in tqdm(filepaths, desc="加载文件"):
        if not os.path.exists(path):
            print(f"文件不存在，跳过: {path}")
            continue
            
        try:
            # 尝试不同编码
            df = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"无法读取文件: {path}")
                continue
                
            if text_col not in df.columns or label_col not in df.columns:
                print(f"文件缺少必要列: {path}")
                continue
                
        except Exception as e:
            print(f"读取文件失败 {path}: {e}")
            continue
            
        texts = df[text_col].astype(str).tolist()
        labels = []
        valid_indices = []

        for i, label in enumerate(df[label_col]):
            label_str = str(label).strip().lower()
            mapped = label_mapping.get(label_str, None)

            if mapped is not None:
                labels.append(mapped)
                valid_indices.append(i)
            else:
                print(f"警告: 忽略未知标签值 '{label}' (文件: {path})")

        all_texts.extend([texts[i] for i in valid_indices])
        all_labels.extend(labels)

    print(f"共加载 {len(all_texts)} 个样本")
    pos_count = sum(1 for label in all_labels if label == 1)
    neg_count = sum(1 for label in all_labels if label == 0)
    print(f"正面样本: {pos_count}, 负面样本: {neg_count}")
    
    return all_texts, all_labels

# ================== 创建情感语料文件 ==================
def create_sentiment_files(texts, labels, pos_path, neg_path):
    """创建情感分析语料文件，带进度条"""
    os.makedirs(os.path.dirname(pos_path), exist_ok=True)

    with open(pos_path, 'w', encoding='utf-8') as f_pos, \
         open(neg_path, 'w', encoding='utf-8') as f_neg:

        print("创建情感语料文件...")
        pos_count, neg_count = 0, 0

        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="处理样本"):
            clean_text = text.replace('\n', '').replace('\r', '').strip()
            if len(clean_text) > 0:
                if label == 1:
                    f_pos.write(clean_text + '\n')
                    pos_count += 1
                elif label == 0:
                    f_neg.write(clean_text + '\n')
                    neg_count += 1

        print(f"创建完成: {pos_count} 个积极样本, {neg_count} 个消极样本")
        return pos_count, neg_count

# ================== 模型评估函数 ==================
def evaluate_model(model, test_texts, test_labels):
    """评估模型准确率，带进度条"""
    correct = 0
    total = len(test_texts)

    for text, label in tqdm(zip(test_texts, test_labels), total=total, desc="评估模型"):
        try:
            score = model.classify(text)
            pred_label = 1 if score > 0.5 else 0
            if pred_label == label:
                correct += 1
        except:
            continue

    return correct / total if total > 0 else 0

# ================== 创建示例数据 ==================
def create_sample_data():
    """如果没有数据文件，创建示例数据"""
    
    sample_train_data = [
        ("这个产品质量非常好，强烈推荐！", "正面"),
        ("服务态度很棒，物流也很快", "正面"),
        ("性价比很高，值得购买", "正面"),
        ("包装精美，质量上乘", "正面"),
        ("体验很好，功能强大", "正面"),
        ("质量太差了，不值这个价格", "负面"),
        ("服务态度恶劣，很不满意", "负面"),
        ("物流超级慢，包装粗糙", "负面"),
        ("用了几天就坏了", "负面"),
        ("功能有问题，操作不便", "负面"),
        ("还行吧，一般般", "中性"),
        ("价格合理，质量一般", "中性"),
        ("收到了，还没用", "中性"),
        ("和描述基本一致", "中性"),
    ]
    
    # 扩展数据
    extended_data = []
    for text, label in sample_train_data:
        extended_data.append((text, label))
        # 添加一些变体
        if "很好" in text:
            extended_data.append((text.replace("很好", "不错"), label))
        if "太差" in text:
            extended_data.append((text.replace("太差", "很差"), label))
    
    # 保存训练数据
    train_df = pd.DataFrame(extended_data, columns=['content', 'sentiment'])
    train_df.to_csv('train.csv', index=False, encoding='utf-8-sig')
    print("✅ 创建示例训练数据: train.csv")
    
    # 创建测试数据
    sample_test_data = [
        ("产品质量很棒，推荐购买", "正面"),
        ("服务很满意，会再来", "正面"),
        ("质量不行，不推荐", "负面"),
        ("客服态度差，很失望", "负面"),
        ("一般般，凑合用", "中性"),
    ]
    
    test_df = pd.DataFrame(sample_test_data, columns=['content', 'sentiment'])
    test_df.to_csv('test.csv', index=False, encoding='utf-8-sig')
    print("✅ 创建示例测试数据: test.csv")

# ================== 主程序 ==================
if __name__ == "__main__":
    start_time = time.time()

    print("=" * 60)
    print("🚀 SnowNLP情感分析模型训练脚本")
    print("=" * 60)

    # ========== 检查数据文件 ==========
    train_files = []
    for pattern in ['train.csv', '训练集.csv', '*train*.csv']:
        train_files.extend(glob(pattern))
    
    if not train_files:
        print("⚠️ 未找到训练数据，创建示例数据...")
        create_sample_data()
        train_files = ['train.csv']

    test_files = []
    for pattern in ['test.csv', '测试集.csv', '*test*.csv']:
        test_files.extend(glob(pattern))
    
    if not test_files:
        test_files = ['test.csv']  # 使用创建的示例数据

    print(f"训练文件: {train_files}")
    print(f"测试文件: {test_files}")

    # ========== 数据准备 ==========
    print("\n📂 加载训练数据...")
    train_texts, train_labels = load_multiple_csvs(train_files)

    if not train_texts:
        print("❌ 没有有效的训练数据")
        exit(1)

    # 创建临时情感语料文件
    pos_path = 'temp_data/pos.txt'
    neg_path = 'temp_data/neg.txt'
    pos_count, neg_count = create_sentiment_files(train_texts, train_labels, pos_path, neg_path)

    if pos_count == 0 or neg_count == 0:
        print("❌ 正面或负面样本数量为0，无法训练")
        exit(1)

    # 加载测试集
    print("\n📂 加载测试数据...")
    test_texts, test_labels = load_multiple_csvs(test_files)

    # ========== 训练前测试 ==========
    if test_texts:
        print("\n" + "=" * 50)
        print("📊 训练前测试...")
        base_model = Sentiment()
        base_acc = evaluate_model(base_model, test_texts, test_labels)
        print(f"【训练前】模型准确率：{base_acc:.2%}")

    # ========== 模型训练 ==========
    print("\n" + "=" * 50)
    print("🔧 开始训练模型...")

    # 创建新的情感分析器实例
    trainer = Sentiment()

    # 直接训练模型，传入正负样本文件路径
    print("正在训练...")
    trainer.train(neg_path, pos_path)

    # 保存模型到多个位置
    model_files = [
        'custom_sentiment.marshal.3',
        'trained_model_v1.marshal.3',
        'sentiment_model.marshal'
    ]
    
    for model_file in model_files:
        try:
            trainer.save(model_file)
            file_size = os.path.getsize(model_file)
            print(f"✅ 模型已保存: {model_file} ({file_size:,} 字节)")
        except Exception as e:
            print(f"❌ 保存失败 {model_file}: {e}")

    # ========== 训练后测试 ==========
    if test_texts:
        print("\n" + "=" * 50)
        print("📊 训练后测试...")
        
        # 加载自定义模型进行测试
        if os.path.exists('custom_sentiment.marshal.3'):
            trained_model = Sentiment()
            trained_model.load('custom_sentiment.marshal.3')
            trained_acc = evaluate_model(trained_model, test_texts, test_labels)
            print(f"【训练后】模型准确率：{trained_acc:.2%}")
            
            if 'base_acc' in locals():
                improvement = (trained_acc - base_acc) * 100
                print(f"准确率提升: {improvement:.2f}%")

    # ========== 清理临时文件 ==========
    try:
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
            print("\n🧹 清理临时文件完成")
    except:
        pass

    # ========== 总结 ==========
    total_time = time.time() - start_time
    print(f"\n⏱️ 总耗时: {total_time:.2f} 秒")
    print("\n🎉 训练完成！生成的模型文件:")
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            print(f"  📄 {model_file} ({size:,} 字节)")
    
    print("\n💡 使用提示:")
    print("1. 现在可以使用GUI工具的'选择模型测试'功能")
    print("2. 选择生成的.marshal文件进行对比测试")
    print("3. 使用'模型对比'功能比较不同模型效果")
    print("\n🚀 启动GUI: python 启动工具.py")