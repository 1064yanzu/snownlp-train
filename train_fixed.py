# -*- coding: utf-8 -*-
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

def load_multiple_csvs(filepaths, text_col='content', label_col='sentiment', neutral_strategy='balance'):
    """
    加载多个CSV文件并合并数据，支持多种中性数据处理策略
    
    Args:
        filepaths: 文件路径列表
        text_col: 文本列名
        label_col: 标签列名  
        neutral_strategy: 中性数据处理策略
            - 'exclude': 排除中性数据(原来的方式)
            - 'random': 随机分配中性数据到正面/负面
            - 'balance': 将中性数据分配给样本数较少的类别来平衡数据
            - 'positive': 全部分配给正面类别
            - 'negative': 全部分配给负面类别
            - 'split': 按比例分配(70%正面, 30%负面)
            - 'keep_original': 保留原始三分类标签(0=负面, 1=中性, 2=正面)
    """
    def detect_encoding(file_path):
        """检测文件编码"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # 读取前10KB用于检测
                result = chardet.detect(raw_data)
                return result['encoding']
        except ImportError:
            # 如果没有chardet，尝试常见编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)  # 尝试读取一部分
                    return encoding
                except UnicodeDecodeError:
                    continue
            return 'utf-8'  # 默认返回utf-8
    
    def read_csv_with_encoding(file_path):
        """使用正确编码读取CSV文件"""
        # 首先尝试UTF-8
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"UTF-8编码失败，正在检测文件编码: {file_path}")
            
        # 检测编码
        detected_encoding = detect_encoding(file_path)
        print(f"检测到编码: {detected_encoding}")
        
        try:
            return pd.read_csv(file_path, encoding=detected_encoding)
        except UnicodeDecodeError:
            print(f"检测编码失败，尝试常见编码...")
            
        # 尝试常见编码
        encodings = ['gbk', 'gb2312', 'utf-8-sig', 'latin1', 'cp1252']
        for encoding in encodings:
            try:
                print(f"尝试编码: {encoding}")
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        # 最后尝试忽略错误
        print("所有编码都失败，使用UTF-8并忽略错误")
        return pd.read_csv(file_path, encoding='utf-8', errors='ignore')
    
    if neutral_strategy == 'keep_original':
        # 保留原始三分类标签的映射
        label_mapping = {
            '负面': 0, '消极': 0, '负向': 0, 'negative': 0,
            '中性': 1, '中立': 1, 'neutral': 1,
            '正面': 2, '积极': 2, '正向': 2, 'positive': 2
        }
    else:
        # 原有的映射方式
        label_mapping = {
            '负面': 0, '消极': 0, '负向': 0, 'negative': 0,
            '正面': 1, '积极': 1, '正向': 1, 'positive': 1,
            '中性': 'neutral', '中立': 'neutral', 'neutral': 'neutral'
        }

    all_texts, all_labels = [], []
    neutral_texts = []  # 暂存中性数据
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
            
            # 检查列是否存在
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

            if neutral_strategy == 'keep_original':
                # 保留原始三分类标签
                if mapped is not None:
                    labels.append(mapped)
                    valid_indices.append(i)
                else:
                    print(f"警告: 忽略未知标签值 '{label}' (文件: {path})")
            else:
                # 原有的处理方式
                if mapped == 'neutral':
                    neutral_indices.append(i)
                elif mapped is not None:  # 明确的正面或负面
                    labels.append(mapped)
                    valid_indices.append(i)
                else:
                    # 跳过未知标签
                    continue

        # 添加明确的正负样本（或三分类样本）
        all_texts.extend([texts[i] for i in valid_indices])
        all_labels.extend(labels)
        
        # 暂存中性样本（只在非keep_original模式下）
        if neutral_strategy != 'keep_original':
            neutral_texts.extend([texts[i] for i in neutral_indices])

    if neutral_strategy == 'keep_original':
        # 统计三分类数据分布
        neg_count = sum(1 for label in all_labels if label == 0)
        neu_count = sum(1 for label in all_labels if label == 1)
        pos_count = sum(1 for label in all_labels if label == 2)
        
        print(f"真实三分类数据统计:")
        print(f"  负面样本: {neg_count}")
        print(f"  中性样本: {neu_count}")
        print(f"  正面样本: {pos_count}")
        print(f"  总样本数: {len(all_texts)}")
        
        return all_texts, all_labels
    
    # 原有的中性数据处理逻辑
    # 统计当前正负样本数量
    current_pos = sum(1 for label in all_labels if label == 1)
    current_neg = sum(1 for label in all_labels if label == 0)
    neutral_count = len(neutral_texts)
    
    print(f"原始数据统计:")
    print(f"  正面样本: {current_pos}")
    print(f"  负面样本: {current_neg}")  
    print(f"  中性样本: {neutral_count}")

    # 处理中性数据
    if neutral_count > 0 and neutral_strategy != 'exclude':
        print(f"正在处理 {neutral_count} 个中性样本...")
        
        if neutral_strategy == 'random':
            # 随机分配
            for text in neutral_texts:
                label = random.choice([0, 1])
                all_texts.append(text)
                all_labels.append(label)
                
        elif neutral_strategy == 'balance':
            # 平衡分配 - 分配给样本数较少的类别
            if current_pos < current_neg:
                # 正面样本较少，将中性数据分配给正面
                for text in neutral_texts:
                    all_texts.append(text)
                    all_labels.append(1)
                print(f"  中性样本全部分配给正面类别(用于平衡)")
            else:
                # 负面样本较少或相等，将中性数据分配给负面  
                for text in neutral_texts:
                    all_texts.append(text)
                    all_labels.append(0)
                print(f"  中性样本全部分配给负面类别(用于平衡)")
                
        elif neutral_strategy == 'positive':
            # 全部分配给正面
            for text in neutral_texts:
                all_texts.append(text)
                all_labels.append(1)
            print(f"  中性样本全部分配给正面类别")
            
        elif neutral_strategy == 'negative':
            # 全部分配给负面
            for text in neutral_texts:
                all_texts.append(text)
                all_labels.append(0)
            print(f"  中性样本全部分配给负面类别")
            
        elif neutral_strategy == 'split':
            # 按比例分配 (70%正面, 30%负面)
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

    # 最终统计
    final_pos = sum(1 for label in all_labels if label == 1)
    final_neg = sum(1 for label in all_labels if label == 0)
    
    print(f"最终数据统计:")
    print(f"  正面样本: {final_pos}")
    print(f"  负面样本: {final_neg}")
    print(f"  总样本数: {len(all_texts)}")

    return all_texts, all_labels

def create_sentiment_files(texts, labels, pos_path, neg_path):
    """创建情感分析语料文件，带进度条"""
    os.makedirs(os.path.dirname(pos_path), exist_ok=True)
    os.makedirs(os.path.dirname(neg_path), exist_ok=True)

    with open(pos_path, 'w', encoding='utf-8') as f_pos, \
         open(neg_path, 'w', encoding='utf-8') as f_neg:

        print("创建情感语料文件...")
        pos_count, neg_count = 0, 0

        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="处理样本"):
            # 清理文本，移除换行符等
            clean_text = text.replace('\n', '').replace('\r', '').strip()
            if len(clean_text) > 0:  # 确保文本不为空
                if label == 1:
                    f_pos.write(clean_text + '\n')
                    pos_count += 1
                elif label == 0:
                    f_neg.write(clean_text + '\n')
                    neg_count += 1

        print(f"创建完成: {pos_count} 个积极样本, {neg_count} 个消极样本")
        return pos_count, neg_count

def evaluate_model_with_snownlp(test_texts, test_labels, use_three_class=True, 
                                negative_threshold=0.4, positive_threshold=0.6):
    """使用 SnowNLP 评估模型准确率，支持二分类和三分类"""
    from snownlp import SnowNLP
    
    if use_three_class:
        print(f"使用三分类评估 (负面 < {negative_threshold}, 中性 {negative_threshold}-{positive_threshold}, 正面 > {positive_threshold})")
    else:
        print("使用传统二分类评估 (负面 < 0.5, 正面 >= 0.5)")
    
    correct = 0
    total = len(test_texts)
    
    # 三分类统计
    if use_three_class:
        # 原始标签转换为三分类格式 (0=负面, 1=中性, 2=正面)
        # 如果原始标签是二分类，需要重新定义
        class_correct = [0, 0, 0]  # 负面、中性、正面
        class_total = [0, 0, 0]
        confusion_matrix = [[0 for _ in range(3)] for _ in range(3)]
        
        for text, expected_binary in tqdm(zip(test_texts, test_labels), total=total, desc="三分类评估"):
            try:
                s = SnowNLP(text)
                score = s.sentiments
                
                # 三分类预测
                if score < negative_threshold:
                    predicted = 0  # 负面
                elif score > positive_threshold:
                    predicted = 2  # 正面
                else:
                    predicted = 1  # 中性
                
                # 将原始二分类标签转换为三分类期望
                # 这里我们需要根据得分重新定义期望标签
                if expected_binary == 0:  # 原本是负面
                    expected = 0
                else:  # 原本是正面
                    expected = 2
                
                # 更新混淆矩阵
                confusion_matrix[expected][predicted] += 1
                
                # 统计准确率
                if predicted == expected:
                    correct += 1
                    class_correct[expected] += 1
                class_total[expected] += 1
                    
            except Exception as e:
                print(f"预测失败: {e}")
                continue
        
        # 输出三分类详细结果
        overall_accuracy = correct / total
        class_names = ["负面", "中性", "正面"]
        
        print(f"\n三分类评估结果:")
        print(f"总体准确率: {overall_accuracy:.2%} ({correct}/{total})")
        
        print("\n各类别准确率:")
        for i, class_name in enumerate(class_names):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                print(f"  {class_name}: {class_acc:.2%} ({class_correct[i]}/{class_total[i]})")
        
        print("\n混淆矩阵:")
        print("实际\\预测", end="")
        for class_name in class_names:
            print(f"{class_name:>8}", end="")
        print()
        
        for i, class_name in enumerate(class_names):
            print(f"{class_name:>6}", end="")
            for j in range(3):
                print(f"{confusion_matrix[i][j]:>8}", end="")
            print()
        
        return overall_accuracy
        
    else:
        # 传统二分类评估
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

def evaluate_model_with_true_three_class(test_texts, test_labels, 
                                        negative_threshold=0.4, positive_threshold=0.6):
    """使用真正的三分类数据进行评估（当测试数据包含真实的中性标签时）"""
    from snownlp import SnowNLP
    
    print(f"使用真实三分类数据评估 (负面 < {negative_threshold}, 中性 {negative_threshold}-{positive_threshold}, 正面 > {positive_threshold})")
    
    correct = 0
    total = len(test_texts)
    
    class_correct = [0, 0, 0]  # 负面、中性、正面
    class_total = [0, 0, 0]
    confusion_matrix = [[0 for _ in range(3)] for _ in range(3)]
    predicted_counts = [0, 0, 0]
    
    for text, expected in tqdm(zip(test_texts, test_labels), total=total, desc="真实三分类评估"):
        try:
            s = SnowNLP(text)
            score = s.sentiments
            
            # 三分类预测
            if score < negative_threshold:
                predicted = 0  # 负面
            elif score > positive_threshold:
                predicted = 2  # 正面
            else:
                predicted = 1  # 中性
            
            predicted_counts[predicted] += 1
            
            # 更新混淆矩阵
            confusion_matrix[expected][predicted] += 1
            
            # 统计准确率
            if predicted == expected:
                correct += 1
                class_correct[expected] += 1
            class_total[expected] += 1
                
        except Exception as e:
            print(f"预测失败: {e}")
            continue
    
    # 输出详细结果
    overall_accuracy = correct / total
    class_names = ["负面", "中性", "正面"]
    
    print(f"\n真实三分类评估结果:")
    print(f"总体准确率: {overall_accuracy:.2%} ({correct}/{total})")
    
    print("\n原始标签分布:")
    for i, class_name in enumerate(class_names):
        percentage = class_total[i] / total * 100
        print(f"  {class_name}: {class_total[i]} ({percentage:.1f}%)")
    
    print("\n预测标签分布:")
    for i, class_name in enumerate(class_names):
        percentage = predicted_counts[i] / total * 100
        print(f"  {class_name}: {predicted_counts[i]} ({percentage:.1f}%)")
    
    print("\n各类别准确率:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"  {class_name}: {class_acc:.2%} ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {class_name}: N/A (无样本)")
    
    print("\n混淆矩阵:")
    print("实际\\预测", end="")
    for class_name in class_names:
        print(f"{class_name:>8}", end="")
    print()
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>6}", end="")
        for j in range(3):
            print(f"{confusion_matrix[i][j]:>8}", end="")
        print()
    
    return overall_accuracy

def backup_and_replace_model(new_model_path):
    """备份原模型并替换为新训练的模型"""
    try:
        import snownlp
        snownlp_path = os.path.dirname(snownlp.__file__)
        sentiment_path = os.path.join(snownlp_path, 'sentiment')
        
        print(f"SnowNLP 安装路径: {snownlp_path}")
        print(f"Sentiment 模型路径: {sentiment_path}")
        
        # 查找模型文件
        model_files = []
        for ext in ['', '.3', '.2']:
            model_file = os.path.join(sentiment_path, f'sentiment.marshal{ext}')
            if os.path.exists(model_file):
                model_files.append(model_file)
                print(f"找到模型文件: {model_file}")
        
        if not model_files:
            print("警告：未找到原始模型文件")
            # 尝试创建基础的模型文件名
            base_model = os.path.join(sentiment_path, 'sentiment.marshal')
            if sys.version_info[0] >= 3:
                base_model += '.3'
            model_files = [base_model]
        
        # 备份原始模型
        for model_file in model_files:
            if os.path.exists(model_file):
                backup_file = model_file + '.backup'
                if not os.path.exists(backup_file):
                    shutil.copy2(model_file, backup_file)
                    print(f"已备份原始模型: {backup_file}")
        
        # 复制新模型
        if os.path.exists(new_model_path):
            for model_file in model_files:
                try:
                    shutil.copy2(new_model_path, model_file)
                    print(f"已替换模型: {model_file}")
                except PermissionError:
                    print(f"权限不足，无法替换 {model_file}")
                    print("请以管理员权限运行或手动复制模型文件")
                    return False
        else:
            print(f"新模型文件不存在: {new_model_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"模型替换失败: {e}")
        return False

if __name__ == "__main__":
    start_time = time.time()

    # ========== 中性数据处理策略选择 ==========
    print("中性数据处理策略选项:")
    print("1. balance  - 自动平衡(推荐): 将中性数据分配给样本数较少的类别")
    print("2. random   - 随机分配: 随机分配中性数据到正面/负面")
    print("3. positive - 全部正面: 将所有中性数据标记为正面")
    print("4. negative - 全部负面: 将所有中性数据标记为负面") 
    print("5. split    - 比例分配: 70%分配给正面，30%分配给负面")
    print("6. exclude  - 排除中性: 完全排除中性数据(原来的方式)")
    
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

    # ========== 评估模式选择 ==========
    print("\n评估模式选择:")
    print("1. 二分类评估 - 传统的正面/负面评估 (阈值0.5)")
    print("2. 三分类评估 - 支持中性类别的评估 (推荐)")
    print("3. 真实三分类评估 - 当测试数据包含真实中性标签时使用")
    
    eval_map = {
        '1': 'binary', '2': 'three_class', '3': 'true_three_class'
    }
    
    while True:
        eval_choice = input("\n请选择评估模式 (1-3，默认为2): ").strip()
        if eval_choice == "":
            eval_choice = "2"
        if eval_choice in eval_map:
            eval_mode = eval_map[eval_choice]
            break
        else:
            print("无效选择，请输入1-3")

    # 三分类阈值设置
    if eval_mode in ['three_class', 'true_three_class']:
        print(f"已选择: {eval_mode} 评估模式")
        
        custom_threshold = input("是否自定义三分类阈值? (y/n，默认n): ").strip().lower()
        if custom_threshold == "y":
            while True:
                try:
                    neg_threshold = float(input("请输入负面阈值 (默认0.4): ") or "0.4")
                    pos_threshold = float(input("请输入正面阈值 (默认0.6): ") or "0.6")
                    if 0 <= neg_threshold < pos_threshold <= 1:
                        break
                    else:
                        print("阈值必须满足: 0 <= 负面阈值 < 正面阈值 <= 1")
                except ValueError:
                    print("请输入有效的数字")
        else:
            neg_threshold, pos_threshold = 0.4, 0.6
            
        print(f"阈值设置: 负面 < {neg_threshold}, 中性 {neg_threshold}-{pos_threshold}, 正面 > {pos_threshold}")
    else:
        print("已选择: 二分类评估模式")
        neg_threshold, pos_threshold = 0.4, 0.6  # 默认值，虽然不会用到

    # ========== 数据准备 ==========
    train_files = [
        'train.csv',
        '训练集.csv'
    ]

    # 检查文件是否存在
    existing_files = [f for f in train_files if os.path.exists(f)]
    if not existing_files:
        print("错误：未找到训练数据文件")
        print("请确保以下文件存在：", train_files)
        sys.exit(1)

    print(f"找到训练文件: {existing_files}")

    # 加载所有训练集数据
    print("加载训练数据...")
    train_texts, train_labels = load_multiple_csvs(existing_files, neutral_strategy=neutral_strategy)
    
    if len(train_texts) == 0:
        print("错误：没有有效的训练数据")
        print("请检查数据文件格式和标签是否正确")
        sys.exit(1)

    # 统计样本分布
    pos_samples = sum(1 for label in train_labels if label == 1)
    neg_samples = sum(1 for label in train_labels if label == 0)
    print(f"训练数据统计: 正面样本 {pos_samples}, 负面样本 {neg_samples}")

    # 创建临时情感语料文件
    pos_path = 'temp_data/pos.txt'
    neg_path = 'temp_data/neg.txt'
    pos_count, neg_count = create_sentiment_files(train_texts, train_labels, pos_path, neg_path)
    
    if pos_count == 0 or neg_count == 0:
        print("错误：正面或负面样本数量为0，无法进行训练")
        print(f"正面样本: {pos_count}, 负面样本: {neg_count}")
        sys.exit(1)

    # 检查样本平衡性
    ratio = min(pos_count, neg_count) / max(pos_count, neg_count)
    if ratio < 0.1:
        print(f"警告：样本不平衡，比例为 {ratio:.2f}")
        print("建议调整训练数据以获得更好的效果")

    # 加载测试集
    test_files = ['test.csv']
    existing_test_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_test_files:
        print("加载测试数据...")
        
        if eval_mode == 'true_three_class':
            # 对于真实三分类，需要保留原始的三分类标签
            print("加载真实三分类测试数据...")
            test_texts, test_labels = load_multiple_csvs(
                existing_test_files, 
                neutral_strategy='keep_original'  # 特殊标记，保留原始三分类标签
            )
        else:
            # 对于其他模式，使用指定的中性数据处理策略
            test_texts, test_labels = load_multiple_csvs(
                existing_test_files, 
                neutral_strategy=neutral_strategy
            )
        
        print(f"测试数据: {len(test_texts)} 个样本")
    else:
        print("警告：未找到测试数据文件，将跳过评估")
        test_texts, test_labels = [], []

    # ========== 训练前测试 ==========
    if test_texts:
        print("\n" + "=" * 50)
        print("开始训练前测试...")
        
        if eval_mode == 'binary':
            base_acc = evaluate_model_with_snownlp(
                test_texts, test_labels, 
                use_three_class=False
            )
        elif eval_mode == 'three_class':
            base_acc = evaluate_model_with_snownlp(
                test_texts, test_labels, 
                use_three_class=True,
                negative_threshold=neg_threshold,
                positive_threshold=pos_threshold
            )
        else:  # true_three_class
            base_acc = evaluate_model_with_true_three_class(
                test_texts, test_labels,
                negative_threshold=neg_threshold,
                positive_threshold=pos_threshold
            )
        
        print(f"【训练前】模型准确率：{base_acc:.2%}")

    # ========== 模型训练 ==========
    print("\n" + "=" * 50)
    print("开始训练模型...")

    try:
        # 使用正确的 API 进行训练
        print("开始训练...")
        print(f"负面样本文件: {neg_path}")
        print(f"正面样本文件: {pos_path}")
        
        # 检查语料文件是否存在且有内容
        if not os.path.exists(neg_path) or not os.path.exists(pos_path):
            print("错误：语料文件不存在")
            sys.exit(1)
            
        neg_size = os.path.getsize(neg_path)
        pos_size = os.path.getsize(pos_path)
        print(f"负面语料文件大小: {neg_size} 字节")
        print(f"正面语料文件大小: {pos_size} 字节")
        
        if neg_size == 0 or pos_size == 0:
            print("错误：语料文件为空")
            sys.exit(1)
        
        # 注意：SnowNLP 的训练参数顺序是 (neg_file, pos_file)
        print("正在训练模型...")
        sentiment.train(neg_path, pos_path)
        print("模型训练完成")

        # 尝试多种方式保存模型
        model_path = 'custom_sentiment.marshal'
        backup_paths = [
            'custom_sentiment.marshal',
            'sentiment_model.marshal',
            'trained_model.marshal'
        ]
        
        saved_successfully = False
        
        for i, save_path in enumerate(backup_paths):
            try:
                print(f"尝试保存模型到: {save_path}")
                sentiment.save(save_path)
                
                # 检查文件是否真的被创建
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    if file_size > 0:
                        print(f"模型保存成功: {save_path} (大小: {file_size} 字节)")
                        model_path = save_path
                        saved_successfully = True
                        break
                    else:
                        print(f"模型文件为空: {save_path}")
                else:
                    print(f"模型文件未创建: {save_path}")
                    
            except Exception as e:
                print(f"保存失败 {save_path}: {e}")
                continue
        
        # 如果标准方法失败，尝试手动保存
        if not saved_successfully:
            print("标准保存方法失败，尝试手动保存...")
            
            try:
                # 尝试获取训练好的模型
                import pickle
                import marshal
                
                # 这是一个hack方法，尝试访问SnowNLP内部状态
                print("尝试手动提取模型...")
                
                # 创建一个临时的sentiment实例来保存
                temp_sentiment = sentiment
                
                # 检查当前工作目录
                print(f"当前工作目录: {os.getcwd()}")
                
                # 尝试在不同路径保存
                alternative_paths = [
                    os.path.join(os.getcwd(), 'manual_sentiment.marshal'),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manual_sentiment.marshal'),
                    'manual_sentiment.marshal'
                ]
                
                for alt_path in alternative_paths:
                    try:
                        print(f"尝试手动保存到: {alt_path}")
                        sentiment.save(alt_path)
                        if os.path.exists(alt_path) and os.path.getsize(alt_path) > 0:
                            model_path = alt_path
                            saved_successfully = True
                            print(f"手动保存成功: {alt_path}")
                            break
                    except Exception as e:
                        print(f"手动保存失败 {alt_path}: {e}")
                        
            except Exception as e:
                print(f"手动保存过程出错: {e}")

        if not saved_successfully:
            print("❌ 模型保存失败！")
            print("可能的原因：")
            print("1. 权限不足")
            print("2. 磁盘空间不足") 
            print("3. SnowNLP版本问题")
            print("4. 训练数据问题")
            
            print("\n建议解决方案：")
            print("1. 以管理员权限运行")
            print("2. 检查磁盘空间")
            print("3. 更新SnowNLP版本")
            print("4. 检查训练数据质量")
            
            # 尝试直接替换系统模型（如果训练成功了）
            print("\n尝试直接替换系统模型...")
            try:
                # 这是一个临时解决方案
                import snownlp
                snownlp_path = os.path.dirname(snownlp.__file__)
                sentiment_path = os.path.join(snownlp_path, 'sentiment')
                
                print("警告：由于模型保存失败，无法进行模型替换")
                print("建议检查训练配置后重新运行")
                
            except Exception as e:
                print(f"系统模型替换也失败: {e}")
        else:
            # 替换系统模型
            print("\n正在替换系统模型...")
            if backup_and_replace_model(model_path):
                print("✅ 模型替换成功！")
                print("重要：请重启 Python 解释器以使新模型生效")
            else:
                print("❌ 模型替换失败，请手动替换模型文件")
                print(f"请将 {model_path} 复制到 SnowNLP 的 sentiment 目录下")

    except Exception as e:
        print(f"训练失败: {e}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()
        
        print("\n请检查：")
        print("1. 语料文件格式是否正确")
        print("2. 是否有足够的内存")
        print("3. 文件权限是否正确")
        print("4. SnowNLP版本是否兼容")
        sys.exit(1)

    # ========== 训练后测试提示 ==========
    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"训练样本数: 正面 {pos_count}, 负面 {neg_count}")
    print(f"样本比例: {pos_count/(pos_count+neg_count):.1%} 正面, {neg_count/(pos_count+neg_count):.1%} 负面")
    
    if test_texts:
        print(f"\n使用的评估模式: {eval_mode}")
        if eval_mode in ['three_class', 'true_three_class']:
            print(f"三分类阈值: 负面 < {neg_threshold}, 中性 {neg_threshold}-{pos_threshold}, 正面 > {pos_threshold}")
        
        print("\n注意：要测试新模型效果，请按照以下步骤操作：")
        print("1. 重启 Python 解释器")
        print("2. 重新导入 snownlp 库")
        print("3. 运行以下测试代码验证新模型效果：")
        
        # 生成测试代码示例
        if eval_mode == 'true_three_class':
            print("""
# 测试新模型 - 真实三分类
from test_new_model import evaluate_model_with_true_three_class
test_texts = [...]  # 你的测试文本
test_labels = [...]  # 你的测试标签 (0=负面, 1=中性, 2=正面)
acc = evaluate_model_with_true_three_class(test_texts, test_labels, {}, {})
print(f"新模型准确率: {{acc:.2%}}")
            """.format(neg_threshold, pos_threshold))
        elif eval_mode == 'three_class':
            print("""
# 测试新模型 - 三分类
from test_new_model import ThreeClassSentiment
classifier = ThreeClassSentiment({}, {})
text = "测试文本"
score, label, label_name = classifier.classify(text)
print(f"{{text}} → {{label_name}} ({{score:.4f}})")
            """.format(neg_threshold, pos_threshold))
        else:
            print("""
# 测试新模型 - 二分类
from snownlp import SnowNLP
s = SnowNLP("测试文本")
print(f"情感得分: {s.sentiments:.4f}")
            """)

    # ========== 总耗时 ==========
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f} 秒")
    
    # 清理临时文件
    cleanup = input("\n是否删除临时语料文件? (y/n): ").lower().strip()
    if cleanup == 'y':
        try:
            if os.path.exists(pos_path):
                os.remove(pos_path)
            if os.path.exists(neg_path):
                os.remove(neg_path)
            print("临时文件已删除")
        except Exception as e:
            print(f"删除临时文件失败: {e}")
    
    print("\n训练流程完成！")
    print("下一步：重启 Python 解释器后测试新模型") 