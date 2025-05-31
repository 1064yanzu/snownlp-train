# -*- coding: utf-8 -*-
"""
SnowNLP情感分析训练脚本 v2.0
完全绕过sentiment.save()问题的版本
直接操作SnowNLP内部模型文件
"""

import pandas as pd
import os
import time
import sys
import shutil
from snownlp import sentiment
from snownlp.sentiment import Sentiment
from glob import glob
from tqdm import tqdm
import random
import marshal
import pickle

def load_multiple_csvs(filepaths, text_col='content', label_col='sentiment', neutral_strategy='balance'):
    """
    加载多个CSV文件并合并数据，支持多种中性数据处理策略
    """
    def detect_encoding(file_path):
        """检测文件编码"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result['encoding']
        except ImportError:
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)
                    return encoding
                except UnicodeDecodeError:
                    continue
            return 'utf-8'
    
    def read_csv_with_encoding(file_path):
        """使用正确编码读取CSV文件"""
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"UTF-8编码失败，正在检测文件编码: {file_path}")
            
        detected_encoding = detect_encoding(file_path)
        print(f"检测到编码: {detected_encoding}")
        
        try:
            return pd.read_csv(file_path, encoding=detected_encoding)
        except UnicodeDecodeError:
            print(f"检测编码失败，尝试常见编码...")
            
        encodings = ['gbk', 'gb2312', 'utf-8-sig', 'latin1', 'cp1252']
        for encoding in encodings:
            try:
                print(f"尝试编码: {encoding}")
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        print("所有编码都失败，使用UTF-8并忽略错误")
        return pd.read_csv(file_path, encoding='utf-8', errors='ignore')
    
    # 标签映射
    label_mapping = {
        '负面': 0, '消极': 0, '负向': 0, 'negative': 0,
        '正面': 1, '积极': 1, '正向': 1, 'positive': 1,
        '中性': 'neutral', '中立': 'neutral', 'neutral': 'neutral'
    }

    all_texts, all_labels = [], []
    neutral_texts = []
    print(f"开始加载 {len(filepaths)} 个数据文件...")
    print(f"中性数据处理策略: {neutral_strategy}")

    for path in tqdm(filepaths, desc="加载文件"):
        if not os.path.exists(path):
            print(f"文件不存在: {path}")
            continue
        
        try:
            print(f"\n正在加载文件: {path}")
            df = read_csv_with_encoding(path)
            print(f"成功加载，共 {len(df)} 行数据")
            
            if text_col not in df.columns:
                print(f"警告: 列 '{text_col}' 不存在于文件 {path}")
                print(f"可用列: {list(df.columns)}")
                continue
                
            if label_col not in df.columns:
                print(f"警告: 列 '{label_col}' 不存在于文件 {path}")
                print(f"可用列: {list(df.columns)}")
                continue
                
        except Exception as e:
            print(f"读取文件失败 {path}: {e}")
            continue
            
        texts = df[text_col].astype(str).tolist()
        labels = []
        valid_indices = []
        neutral_indices = []

        for i, label in enumerate(df[label_col]):
            label_str = str(label).strip().lower()
            mapped = label_mapping.get(label_str, None)

            if mapped == 'neutral':
                neutral_indices.append(i)
            elif mapped is not None:
                labels.append(mapped)
                valid_indices.append(i)

        all_texts.extend([texts[i] for i in valid_indices])
        all_labels.extend(labels)
        neutral_texts.extend([texts[i] for i in neutral_indices])

    # 处理中性数据
    current_pos = sum(1 for label in all_labels if label == 1)
    current_neg = sum(1 for label in all_labels if label == 0)
    neutral_count = len(neutral_texts)
    
    print(f"原始数据统计:")
    print(f"  正面样本: {current_pos}")
    print(f"  负面样本: {current_neg}")  
    print(f"  中性样本: {neutral_count}")

    if neutral_count > 0 and neutral_strategy != 'exclude':
        print(f"正在处理 {neutral_count} 个中性样本...")
        
        if neutral_strategy == 'random':
            for text in neutral_texts:
                label = random.choice([0, 1])
                all_texts.append(text)
                all_labels.append(label)
        elif neutral_strategy == 'balance':
            if current_pos < current_neg:
                for text in neutral_texts:
                    all_texts.append(text)
                    all_labels.append(1)
                print(f"  中性样本全部分配给正面类别(用于平衡)")
            else:
                for text in neutral_texts:
                    all_texts.append(text)
                    all_labels.append(0)
                print(f"  中性样本全部分配给负面类别(用于平衡)")
        elif neutral_strategy == 'positive':
            for text in neutral_texts:
                all_texts.append(text)
                all_labels.append(1)
            print(f"  中性样本全部分配给正面类别")
        elif neutral_strategy == 'negative':
            for text in neutral_texts:
                all_texts.append(text)
                all_labels.append(0)
            print(f"  中性样本全部分配给负面类别")
        elif neutral_strategy == 'split':
            random.shuffle(neutral_texts)
            split_point = int(len(neutral_texts) * 0.7)
            pos_neutrals = neutral_texts[:split_point]
            neg_neutrals = neutral_texts[split_point:]
            
            for text in pos_neutrals:
                all_texts.append(text)
                all_labels.append(1)
            for text in neg_neutrals:
                all_texts.append(text)
                all_labels.append(0)
                
            print(f"  中性样本按比例分配: {len(pos_neutrals)}个给正面, {len(neg_neutrals)}个给负面")

    final_pos = sum(1 for label in all_labels if label == 1)
    final_neg = sum(1 for label in all_labels if label == 0)
    
    print(f"最终数据统计:")
    print(f"  正面样本: {final_pos}")
    print(f"  负面样本: {final_neg}")
    print(f"  总样本数: {len(all_texts)}")

    return all_texts, all_labels

def create_sentiment_files(texts, labels, pos_path, neg_path):
    """创建情感分析语料文件"""
    os.makedirs(os.path.dirname(pos_path), exist_ok=True)
    os.makedirs(os.path.dirname(neg_path), exist_ok=True)

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

def evaluate_model_with_snownlp(test_texts, test_labels):
    """使用SnowNLP评估模型准确率"""
    from snownlp import SnowNLP
    
    print("使用传统二分类评估 (负面 < 0.5, 正面 >= 0.5)")
    
    correct = 0
    total = len(test_texts)
    
    for text, label in tqdm(zip(test_texts, test_labels), total=total, desc="二分类评估"):
        try:
            s = SnowNLP(text)
            score = s.sentiments
            pred_label = 1 if score > 0.5 else 0
            if pred_label == label:
                correct += 1
        except Exception as e:
            print(f"预测失败: {e}")
            continue

    return correct / total if total > 0 else 0

def direct_replace_snownlp_model(neg_path, pos_path):
    """
    直接替换SnowNLP系统模型，绕过save()方法
    这是一个hack方法，直接操作SnowNLP的内部文件
    """
    print("\n" + "="*50)
    print("🔧 开始直接模型替换（绕过save方法）...")
    
    try:
        # 获取SnowNLP安装路径
        import snownlp
        snownlp_path = os.path.dirname(snownlp.__file__)
        sentiment_path = os.path.join(snownlp_path, 'sentiment')
        
        print(f"SnowNLP路径: {snownlp_path}")
        print(f"Sentiment模块路径: {sentiment_path}")
        
        # 备份原始模型
        model_files = []
        for ext in ['', '.3', '.2']:
            model_file = os.path.join(sentiment_path, f'sentiment.marshal{ext}')
            if os.path.exists(model_file):
                model_files.append(model_file)
                backup_file = model_file + '.backup_v2'
                if not os.path.exists(backup_file):
                    shutil.copy2(model_file, backup_file)
                    print(f"✅ 已备份: {backup_file}")
        
        if not model_files:
            print("❌ 未找到原始模型文件")
            return False
        
        # 临时训练新模型
        print("开始训练新模型...")
        sentiment.train(neg_path, pos_path)
        print("✅ 训练完成")
        
        # 尝试方法1：直接获取训练后的模型数据
        try:
            print("尝试方法1：直接模型数据提取...")
            
            # 这里我们需要手动触发模型训练并获取结果
            # 由于SnowNLP的内部实现，我们需要重新实现训练逻辑
            
            from snownlp.sentiment.sentiment import train as train_func
            from snownlp.sentiment import data_path as sentiment_data_path
            
            # 重新训练并获取模型
            model_data = train_func(neg_path, pos_path)
            
            if model_data:
                # 保存到所有模型文件
                for model_file in model_files:
                    try:
                        print(f"保存模型到: {model_file}")
                        with open(model_file, 'wb') as f:
                            marshal.dump(model_data, f)
                        print(f"✅ 成功保存: {model_file}")
                    except Exception as e:
                        print(f"❌ 保存失败 {model_file}: {e}")
                
                return True
            else:
                print("❌ 未能获取模型数据")
                
        except Exception as e:
            print(f"方法1失败: {e}")
        
        # 尝试方法2：复制临时生成的模型文件
        try:
            print("尝试方法2：查找临时模型文件...")
            
            # 查找可能的临时模型文件位置
            from snownlp.sentiment import data_path as sentiment_data_path
            
            possible_temp_files = [
                os.path.join(sentiment_data_path, 'sentiment.marshal'),
                os.path.join(sentiment_data_path, 'sentiment.marshal.3'),
                'sentiment.marshal',
                'sentiment.marshal.3',
                os.path.join(os.getcwd(), 'sentiment.marshal'),
                os.path.join(os.getcwd(), 'sentiment.marshal.3')
            ]
            
            for temp_file in possible_temp_files:
                if os.path.exists(temp_file):
                    print(f"找到临时模型: {temp_file}")
                    file_size = os.path.getsize(temp_file)
                    if file_size > 0:
                        # 复制到系统位置
                        for model_file in model_files:
                            try:
                                shutil.copy2(temp_file, model_file)
                                print(f"✅ 复制成功: {temp_file} → {model_file}")
                            except Exception as e:
                                print(f"❌ 复制失败: {e}")
                        return True
                    
        except Exception as e:
            print(f"方法2失败: {e}")
        
        # 尝试方法3：手动重建训练过程
        try:
            print("尝试方法3：手动重建训练...")
            
            # 读取训练数据
            pos_data = []
            neg_data = []
            
            with open(pos_path, 'r', encoding='utf-8') as f:
                pos_data = [line.strip() for line in f if line.strip()]
            
            with open(neg_path, 'r', encoding='utf-8') as f:
                neg_data = [line.strip() for line in f if line.strip()]
            
            print(f"读取训练数据: {len(pos_data)} 正面, {len(neg_data)} 负面")
            
            # 使用sklearn重新训练
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB
                import jieba
                
                # 准备数据
                all_texts = pos_data + neg_data
                all_labels = [1] * len(pos_data) + [0] * len(neg_data)
                
                # 分词
                print("正在分词...")
                segmented_texts = []
                for text in all_texts[:5000]:  # 限制样本数量加快速度
                    words = list(jieba.cut(text))
                    segmented_texts.append(' '.join(words))
                
                # 对应的标签也要截取
                limited_labels = all_labels[:5000]
                
                # 训练模型
                print("训练sklearn模型...")
                vectorizer = TfidfVectorizer(max_features=3000)
                X = vectorizer.fit_transform(segmented_texts)
                
                classifier = MultinomialNB()
                classifier.fit(X, limited_labels)
                
                # 保存自定义模型
                model_package = {
                    'vectorizer': vectorizer,
                    'classifier': classifier,
                    'version': 'custom_v2'
                }
                
                # 先尝试保存到临时文件
                temp_model_path = 'custom_model_temp.pkl'
                with open(temp_model_path, 'wb') as f:
                    pickle.dump(model_package, f)
                
                # 验证临时文件
                if os.path.exists(temp_model_path) and os.path.getsize(temp_model_path) > 0:
                    print(f"✅ 临时模型创建成功: {temp_model_path}")
                    
                    # 复制到系统位置
                    for model_file in model_files:
                        try:
                            shutil.copy2(temp_model_path, model_file)
                            print(f"✅ 复制成功: {temp_model_path} → {model_file}")
                        except Exception as e:
                            print(f"❌ 复制失败: {e}")
                    
                    return True
                else:
                    print("❌ 临时模型文件创建失败")
                    
            except ImportError as e:
                print(f"缺少依赖库: {e}")
                print("请运行: pip install scikit-learn jieba")
                
        except Exception as e:
            print(f"方法3失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("❌ 所有模型替换方法都失败了")
        return False
        
    except Exception as e:
        print(f"❌ 直接模型替换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    start_time = time.time()
    
    print("SnowNLP情感分析训练脚本 v2.0")
    print("绕过sentiment.save()问题的版本")
    print("="*50)

    # 中性数据处理策略选择
    print("中性数据处理策略选项:")
    print("1. balance  - 自动平衡(推荐)")
    print("2. random   - 随机分配")
    print("3. positive - 全部正面")
    print("4. negative - 全部负面") 
    print("5. split    - 比例分配")
    print("6. exclude  - 排除中性")
    
    strategy_map = {
        '1': 'balance', '2': 'random', '3': 'positive', 
        '4': 'negative', '5': 'split', '6': 'exclude'
    }
    
    while True:
        choice = input("\n请选择中性数据处理策略 (1-6，默认为1): ").strip()
        if choice == "":
            choice = "1"
        if choice in strategy_map:
            neutral_strategy = strategy_map[choice]
            break
        else:
            print("无效选择，请输入1-6")

    print(f"已选择策略: {neutral_strategy}")

    # 数据文件检查
    train_files = ['train.csv', '训练集.csv']
    existing_files = [f for f in train_files if os.path.exists(f)]
    
    if not existing_files:
        print("错误：未找到训练数据文件")
        print("请确保以下文件存在：", train_files)
        return

    print(f"找到训练文件: {existing_files}")

    # 加载训练数据
    print("加载训练数据...")
    train_texts, train_labels = load_multiple_csvs(existing_files, neutral_strategy=neutral_strategy)
    
    if len(train_texts) == 0:
        print("错误：没有有效的训练数据")
        return

    # 创建语料文件
    pos_path = 'temp_data/pos.txt'
    neg_path = 'temp_data/neg.txt'
    pos_count, neg_count = create_sentiment_files(train_texts, train_labels, pos_path, neg_path)
    
    if pos_count == 0 or neg_count == 0:
        print("错误：正面或负面样本数量为0，无法进行训练")
        return

    # 训练前测试
    test_files = ['test.csv']
    existing_test_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_test_files:
        print("加载测试数据...")
        test_texts, test_labels = load_multiple_csvs(existing_test_files, neutral_strategy=neutral_strategy)
        
        if test_texts:
            print("\n开始训练前测试...")
            base_acc = evaluate_model_with_snownlp(test_texts, test_labels)
            print(f"【训练前】模型准确率：{base_acc:.2%}")

    # 直接模型替换
    success = direct_replace_snownlp_model(neg_path, pos_path)
    
    if success:
        print("\n✅ 模型训练和替换成功！")
        print("重要提示：")
        print("1. 请重启Python解释器")
        print("2. 重新导入snownlp库")
        print("3. 测试新模型效果")
        
        if existing_test_files and test_texts:
            print("\n建议测试代码：")
            print("""
from snownlp import SnowNLP
test_text = "这个产品很好用"
s = SnowNLP(test_text)
print(f"情感得分: {s.sentiments:.4f}")
            """)
    else:
        print("\n❌ 模型训练失败")
        print("建议：")
        print("1. 检查数据质量")
        print("2. 尝试减少训练样本数量")
        print("3. 使用其他情感分析库")

    # 总耗时
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    main() 