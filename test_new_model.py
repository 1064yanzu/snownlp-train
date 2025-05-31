# -*- coding: utf-8 -*-
"""
新训练模型测试脚本 - 支持三分类预测
运行此脚本前请确保：
1. 已经运行了 train_fixed.py 训练脚本
2. 已经重启了 Python 解释器
3. 新模型已经替换到 SnowNLP 安装目录
"""

from snownlp import SnowNLP
import pandas as pd
import os
from tqdm import tqdm

class ThreeClassSentiment:
    """三分类情感分析器"""
    
    def __init__(self, negative_threshold=0.4, positive_threshold=0.6):
        """
        初始化三分类情感分析器
        
        Args:
            negative_threshold: 负面阈值，得分低于此值为负面
            positive_threshold: 正面阈值，得分高于此值为正面
            中间区间为中性
        """
        self.negative_threshold = negative_threshold
        self.positive_threshold = positive_threshold
        
    def classify(self, text):
        """
        三分类预测
        
        Returns:
            (score, label, label_name)
            label: 0=负面, 1=中性, 2=正面
        """
        s = SnowNLP(text)
        score = s.sentiments
        
        if score < self.negative_threshold:
            return score, 0, "负面"
        elif score > self.positive_threshold:
            return score, 2, "正面"
        else:
            return score, 1, "中性"
    
    def batch_classify(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results

def test_three_class_samples():
    """测试三分类示例案例"""
    print("=" * 60)
    print("三分类示例测试")
    print("=" * 60)
    
    classifier = ThreeClassSentiment()
    
    test_cases = [
        ("这个电影真的很好看，我很喜欢", 2, "正面"),  
        ("太糟糕了，完全不推荐", 0, "负面"),  
        ("还可以吧，没什么特别的", 1, "中性"),
        ("演员表演很棒，剧情也很有趣", 2, "正面"),  
        ("浪费时间，很无聊", 0, "负面"),  
        ("一般般，看不看都行", 1, "中性"),
        ("这部电影超级棒，强烈推荐大家去看", 2, "正面"),  
        ("垃圾电影，看了后悔", 0, "负面"),  
        ("普通的电影，中规中矩", 1, "中性"),
        ("剧情很精彩，值得一看", 2, "正面"),  
        ("制作粗糙，剧情无聊", 0, "负面"),
        ("还行吧，不好不坏", 1, "中性"),
    ]
    
    correct = 0
    total = len(test_cases)
    class_correct = [0, 0, 0]  # 负面、中性、正面的正确数
    class_total = [0, 0, 0]
    
    print(f"分类阈值设置: 负面 < {classifier.negative_threshold}, 中性 {classifier.negative_threshold}-{classifier.positive_threshold}, 正面 > {classifier.positive_threshold}")
    print()
    
    for i, (text, expected, expected_name) in enumerate(test_cases, 1):
        score, predicted, predicted_name = classifier.classify(text)
        is_correct = predicted == expected
        
        print(f"{i:2d}. 文本: {text}")
        print(f"    情感得分: {score:.4f}")
        print(f"    预测: {predicted_name} | 期望: {expected_name} | 结果: {'✓' if is_correct else '✗'}")
        print()
        
        if is_correct:
            correct += 1
            class_correct[expected] += 1
        class_total[expected] += 1
    
    # 计算各类准确率
    overall_accuracy = correct / total
    class_names = ["负面", "中性", "正面"]
    
    print("=" * 40)
    print("三分类测试结果:")
    print("=" * 40)
    print(f"总体准确率: {overall_accuracy:.2%} ({correct}/{total})")
    
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"{class_name}准确率: {class_acc:.2%} ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"{class_name}准确率: N/A (无样本)")
    
    return overall_accuracy

def test_csv_data_three_class(csv_file, text_col='content', label_col='sentiment'):
    """测试CSV文件中的数据，使用三分类预测"""
    if not os.path.exists(csv_file):
        print(f"测试文件不存在: {csv_file}")
        return None
    
    print("=" * 60)
    print(f"CSV数据三分类测试: {csv_file}")
    print("=" * 60)
    
    classifier = ThreeClassSentiment()
    
    # 读取测试数据
    df = pd.read_csv(csv_file)
    print(f"加载测试数据: {len(df)} 条记录")
    
    # 标签映射 - 现在支持真正的三分类
    label_mapping = {
        '负面': 0, '消极': 0, '负向': 0, 'negative': 0,
        '中性': 1, '中立': 1, 'neutral': 1,
        '正面': 2, '积极': 2, '正向': 2, 'positive': 2,
    }
    
    # 过滤和转换数据
    valid_data = []
    
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        label_str = str(row[label_col]).strip().lower()
        
        if label_str in label_mapping and len(text) > 0:
            valid_data.append((text, label_mapping[label_str]))
    
    if not valid_data:
        print("没有找到有效的测试数据")
        return None
    
    print(f"有效测试数据: {len(valid_data)} 条")
    
    # 统计原始标签分布
    label_counts = [0, 0, 0]  # 负面、中性、正面
    for _, label in valid_data:
        label_counts[label] += 1
    
    class_names = ["负面", "中性", "正面"]
    print("原始标签分布:")
    for i, (class_name, count) in enumerate(zip(class_names, label_counts)):
        percentage = count / len(valid_data) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # 测试模型
    correct = 0
    total = len(valid_data)
    
    # 混淆矩阵 - 3x3
    confusion_matrix = [[0 for _ in range(3)] for _ in range(3)]
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    
    # 得分分布统计
    score_distribution = {
        '0.0-0.1': 0, '0.1-0.2': 0, '0.2-0.3': 0, '0.3-0.4': 0, 
        '0.4-0.5': 0, '0.5-0.6': 0, '0.6-0.7': 0, '0.7-0.8': 0, 
        '0.8-0.9': 0, '0.9-1.0': 0
    }
    
    # 三分类预测分布
    predicted_counts = [0, 0, 0]
    
    print("开始三分类测试...")
    for text, expected in tqdm(valid_data, desc="测试进度"):
        try:
            score, predicted, predicted_name = classifier.classify(text)
            
            # 统计得分分布
            for range_key in score_distribution:
                range_parts = range_key.split('-')
                if float(range_parts[0]) <= score < float(range_parts[1]):
                    score_distribution[range_key] += 1
                    break
            else:
                if score == 1.0:
                    score_distribution['0.9-1.0'] += 1
            
            # 统计预测分布
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
    
    # 计算结果
    overall_accuracy = correct / total
    
    print("\n" + "=" * 50)
    print("三分类测试结果:")
    print("=" * 50)
    print(f"总体准确率: {overall_accuracy:.2%} ({correct}/{total})")
    
    print("\n各类别准确率:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"  {class_name}: {class_acc:.2%} ({class_correct[i]}/{class_total[i]})")
    
    print("\n预测分布:")
    for i, (class_name, count) in enumerate(zip(class_names, predicted_counts)):
        percentage = count / total * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
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
    
    print("\n得分分布:")
    for range_key, count in score_distribution.items():
        percentage = count / total * 100
        range_label = "负面区间" if range_key.split('-')[1] <= "0.4" else \
                     "中性区间" if range_key.split('-')[0] >= "0.4" and range_key.split('-')[1] <= "0.6" else \
                     "正面区间"
        print(f"  {range_key} ({range_label}): {count:4d} ({percentage:5.1f}%)")
    
    return overall_accuracy

def test_sample_cases():
    """测试一些示例案例"""
    print("=" * 60)
    print("示例测试")
    print("=" * 60)
    
    test_cases = [
        ("这个电影真的很好看，我很喜欢", 1),  # 正面
        ("太糟糕了，完全不推荐", 0),  # 负面
        ("演员表演很棒，剧情也很有趣", 1),  # 正面
        ("浪费时间，很无聊", 0),  # 负面
        ("这部电影超级棒，强烈推荐大家去看", 1),  # 正面
        ("垃圾电影，看了后悔", 0),  # 负面
        ("还可以吧，没什么特别的", 1),  # 中性偏正面
        ("剧情很精彩，值得一看", 1),  # 正面
        ("制作粗糙，剧情无聊", 0),  # 负面
        ("非常感动，看完想哭", 1),  # 正面
    ]
    
    correct = 0
    total = len(test_cases)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        s = SnowNLP(text)
        score = s.sentiments
        predicted = 1 if score > 0.5 else 0
        is_correct = predicted == expected
        
        print(f"{i:2d}. 文本: {text}")
        print(f"    情感得分: {score:.4f}")
        print(f"    预测: {'正面' if predicted == 1 else '负面'} | "
              f"期望: {'正面' if expected == 1 else '负面'} | "
              f"结果: {'✓' if is_correct else '✗'}")
        print()
        
        if is_correct:
            correct += 1
    
    accuracy = correct / total
    print(f"示例测试准确率: {accuracy:.2%} ({correct}/{total})")
    return accuracy

def test_csv_data(csv_file, text_col='content', label_col='sentiment', neutral_strategy='balance'):
    """测试CSV文件中的数据，支持中性数据处理"""
    if not os.path.exists(csv_file):
        print(f"测试文件不存在: {csv_file}")
        return None
    
    print("=" * 60)
    print(f"CSV数据测试: {csv_file}")
    print(f"中性数据处理策略: {neutral_strategy}")
    print("=" * 60)
    
    # 读取测试数据
    df = pd.read_csv(csv_file)
    print(f"加载测试数据: {len(df)} 条记录")
    
    # 标签映射
    label_mapping = {
        '负面': 0, '消极': 0, '负向': 0, 'negative': 0,
        '正面': 1, '积极': 1, '正向': 1, 'positive': 1,
        '中性': 'neutral', '中立': 'neutral', 'neutral': 'neutral'
    }
    
    # 过滤和转换数据
    valid_data = []
    neutral_data = []
    
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        label_str = str(row[label_col]).strip().lower()
        
        if label_str in label_mapping and len(text) > 0:
            if label_mapping[label_str] == 'neutral':
                neutral_data.append(text)
            else:
                valid_data.append((text, label_mapping[label_str]))
    
    # 统计原始数据
    original_pos = sum(1 for _, label in valid_data if label == 1)
    original_neg = len(valid_data) - original_pos
    neutral_count = len(neutral_data)
    
    print(f"原始数据分布:")
    print(f"  正面: {original_pos}, 负面: {original_neg}, 中性: {neutral_count}")
    
    # 处理中性数据
    if neutral_count > 0 and neutral_strategy != 'exclude':
        import random
        print(f"正在处理 {neutral_count} 个中性样本...")
        
        if neutral_strategy == 'random':
            for text in neutral_data:
                label = random.choice([0, 1])
                valid_data.append((text, label))
                
        elif neutral_strategy == 'balance':
            target_label = 1 if original_pos < original_neg else 0
            for text in neutral_data:
                valid_data.append((text, target_label))
            print(f"  中性样本分配给{'正面' if target_label == 1 else '负面'}类别(平衡策略)")
                
        elif neutral_strategy == 'positive':
            for text in neutral_data:
                valid_data.append((text, 1))
            print(f"  中性样本全部分配给正面类别")
            
        elif neutral_strategy == 'negative':
            for text in neutral_data:
                valid_data.append((text, 0))
            print(f"  中性样本全部分配给负面类别")
            
        elif neutral_strategy == 'split':
            random.shuffle(neutral_data)
            split_point = int(len(neutral_data) * 0.7)
            
            for text in neutral_data[:split_point]:
                valid_data.append((text, 1))
            for text in neutral_data[split_point:]:
                valid_data.append((text, 0))
            print(f"  中性样本按比例分配: {split_point}个给正面, {len(neutral_data)-split_point}个给负面")
    
    if not valid_data:
        print("没有找到有效的测试数据")
        return None
    
    print(f"最终测试数据: {len(valid_data)} 条")
    
    # 统计最终标签分布
    final_pos = sum(1 for _, label in valid_data if label == 1)
    final_neg = len(valid_data) - final_pos
    print(f"最终标签分布: 正面 {final_pos}, 负面 {final_neg}")
    
    # 测试模型
    correct = 0
    total = len(valid_data)
    
    pos_correct = 0
    neg_correct = 0
    pos_total = 0
    neg_total = 0
    
    score_distribution = {'0.0-0.1': 0, '0.1-0.2': 0, '0.2-0.3': 0, '0.3-0.4': 0, 
                         '0.4-0.5': 0, '0.5-0.6': 0, '0.6-0.7': 0, '0.7-0.8': 0, 
                         '0.8-0.9': 0, '0.9-1.0': 0}
    
    print("开始测试...")
    for text, expected in tqdm(valid_data, desc="测试进度"):
        try:
            s = SnowNLP(text)
            score = s.sentiments
            predicted = 1 if score > 0.5 else 0
            
            # 统计得分分布
            for range_key in score_distribution:
                range_parts = range_key.split('-')
                if float(range_parts[0]) <= score < float(range_parts[1]):
                    score_distribution[range_key] += 1
                    break
            else:
                # 处理 1.0 的情况
                if score == 1.0:
                    score_distribution['0.9-1.0'] += 1
            
            # 统计准确率
            if predicted == expected:
                correct += 1
                if expected == 1:
                    pos_correct += 1
                else:
                    neg_correct += 1
            
            if expected == 1:
                pos_total += 1
            else:
                neg_total += 1
                
        except Exception as e:
            print(f"预测失败: {e}")
            continue
    
    # 计算结果
    overall_accuracy = correct / total
    pos_accuracy = pos_correct / pos_total if pos_total > 0 else 0
    neg_accuracy = neg_correct / neg_total if neg_total > 0 else 0
    
    print("\n" + "=" * 40)
    print("测试结果:")
    print("=" * 40)
    print(f"总体准确率: {overall_accuracy:.2%} ({correct}/{total})")
    print(f"正面样本准确率: {pos_accuracy:.2%} ({pos_correct}/{pos_total})")
    print(f"负面样本准确率: {neg_accuracy:.2%} ({neg_correct}/{neg_total})")
    
    print("\n得分分布:")
    for range_key, count in score_distribution.items():
        percentage = count / total * 100
        print(f"  {range_key}: {count:4d} ({percentage:5.1f}%)")
    
    return overall_accuracy

def compare_with_original():
    """比较新模型和原始模型的差异"""
    print("=" * 60)
    print("新旧模型对比测试")
    print("=" * 60)
    
    test_texts = [
        "这个产品质量很好，我很满意",
        "服务态度差，不推荐",
        "价格合理，性价比高",
        "完全不值这个价钱",
        "物流很快，包装也很好",
        "产品有问题，退货很麻烦"
    ]
    
    print("测试文本和新模型预测结果:")
    for i, text in enumerate(test_texts, 1):
        s = SnowNLP(text)
        score = s.sentiments
        sentiment = "正面" if score > 0.5 else "负面"
        print(f"{i}. {text}")
        print(f"   得分: {score:.4f} ({sentiment})")
        print()

def main():
    """主测试函数"""
    print("SnowNLP 新训练模型测试")
    print("时间:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # 选择测试模式
    print("测试模式选择:")
    print("1. 二分类模式 - 传统的正面/负面分类")
    print("2. 三分类模式 - 支持中性类别 (0.4-0.6为中性)")
    
    while True:
        mode_choice = input("\n请选择测试模式 (1-2，默认为2): ").strip()
        if mode_choice == "":
            mode_choice = "2"
        if mode_choice in ["1", "2"]:
            break
        else:
            print("无效选择，请输入1或2")
    
    use_three_class = (mode_choice == "2")
    
    if use_three_class:
        print("已选择: 三分类模式 (负面 < 0.4, 中性 0.4-0.6, 正面 > 0.6)")
        
        # 询问是否自定义阈值
        custom_threshold = input("是否自定义阈值? (y/n，默认n): ").strip().lower()
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
        print("已选择: 二分类模式 (负面 < 0.5, 正面 >= 0.5)")
        
        # 对于二分类，询问中性数据处理策略
        print("\n中性数据处理策略选项:")
        print("1. balance  - 自动平衡: 分配给样本数较少的类别")
        print("2. random   - 随机分配: 随机分配到正面/负面")
        print("3. positive - 全部正面: 将所有中性数据标记为正面")
        print("4. negative - 全部负面: 将所有中性数据标记为负面") 
        print("5. split    - 比例分配: 70%正面，30%负面")
        print("6. exclude  - 排除中性: 完全排除中性数据")
        
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
    
    print()
    
    # 1. 示例测试
    if use_three_class:
        # 修改ThreeClassSentiment的阈值
        global ThreeClassSentiment
        temp_classifier = ThreeClassSentiment(neg_threshold, pos_threshold)
        ThreeClassSentiment.__init__ = lambda self, nt=neg_threshold, pt=pos_threshold: (
            setattr(self, 'negative_threshold', nt),
            setattr(self, 'positive_threshold', pt)
        )[-1]
        
        sample_acc = test_three_class_samples()
    else:
        sample_acc = test_sample_cases()
    
    # 2. 测试CSV数据
    test_files = ['test.csv', 'train.csv', '训练集.csv']
    csv_acc = None
    
    for test_file in test_files:
        if os.path.exists(test_file):
            if use_three_class:
                csv_acc = test_csv_data_three_class(test_file)
            else:
                csv_acc = test_csv_data(test_file, neutral_strategy=neutral_strategy)
            break
    
    if csv_acc is None:
        print("没有找到可用的测试数据文件")
    
    # 3. 新旧模型对比
    if not use_three_class:
        compare_with_original()
    else:
        print("=" * 60)
        print("三分类模式示例对比")
        print("=" * 60)
        
        classifier = ThreeClassSentiment(neg_threshold, pos_threshold)
        test_texts = [
            "这个产品质量很好，我很满意",
            "服务态度差，不推荐", 
            "价格合理，性价比高",
            "完全不值这个价钱",
            "还可以吧，不好不坏",
            "一般般的产品"
        ]
        
        print("三分类预测示例:")
        for i, text in enumerate(test_texts, 1):
            score, label, label_name = classifier.classify(text)
            print(f"{i}. {text}")
            print(f"   得分: {score:.4f} → {label_name}")
            print()
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"示例测试准确率: {sample_acc:.2%}")
    if csv_acc:
        print(f"CSV数据测试准确率: {csv_acc:.2%}")
    
    if sample_acc > 0.7:
        print("✅ 模型表现良好")
    elif sample_acc > 0.5:
        print("⚠️  模型表现一般，可能需要更多训练数据或调整")
    else:
        print("❌ 模型表现较差，建议检查训练数据质量")
    
    print("\n建议:")
    print("1. 如果准确率较低，检查训练数据的质量和数量")
    print("2. 确保正负样本分布均衡")
    print("3. 可以尝试增加更多领域相关的训练数据")
    print("4. 检查文本预处理是否合适")

if __name__ == "__main__":
    main() 