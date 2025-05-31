# -*- coding: utf-8 -*-
"""
SnowNLP新训练模型测试脚本
全面测试新模型的情感分析效果
"""

import pandas as pd
import os
import time
from snownlp import SnowNLP
from tqdm import tqdm
import random

def basic_sentiment_test():
    """基础情感分析测试"""
    print("="*60)
    print("🧪 基础情感分析测试")
    print("="*60)
    
    # 精心设计的测试用例
    test_cases = [
        # 明显正面
        ("这个产品质量非常好，强烈推荐大家购买！", "正面"),
        ("服务态度超棒，物流也很快，非常满意", "正面"),
        ("性价比很高，用了一段时间效果很不错", "正面"),
        ("包装精美，质量上乘，值得信赖的品牌", "正面"),
        ("体验很棒，功能强大，使用简单方便", "正面"),
        
        # 明显负面  
        ("质量太差了，完全不值这个价格", "负面"),
        ("服务态度恶劣，客服回复很慢很敷衍", "负面"),
        ("物流超级慢，包装也很粗糙", "负面"),
        ("用了几天就坏了，太失望了", "负面"),
        ("功能有很多问题，操作也不方便", "负面"),
        
        # 中性/模糊
        ("还可以吧，凑合能用", "中性"),
        ("价格合理，质量一般般", "中性"),
        ("没什么特别的，普通产品", "中性"),
        ("收到了，暂时还没用", "中性"),
        ("和描述基本一致", "中性"),
        
        # 复杂情感
        ("价格有点贵，但是质量确实不错", "正面"),
        ("功能很好，就是界面有点丑", "正面"),
        ("质量还行，但是客服态度不太好", "中性"),
        ("物流很快，但是包装有点简陋", "正面"),
        ("总体满意，就是有点小瑕疵", "正面")
    ]
    
    print(f"测试 {len(test_cases)} 个样本:")
    print("-" * 60)
    
    correct = 0
    total = len(test_cases)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        try:
            s = SnowNLP(text)
            score = s.sentiments
            
            if score > 0.6:
                predicted = "正面"
            elif score < 0.4:
                predicted = "负面"
            else:
                predicted = "中性"
            
            # 判断预测是否正确
            if expected == "中性":
                is_correct = True  # 中性样本不参与准确率计算
                status = "😐"
            elif predicted == expected:
                is_correct = True
                correct += 1
                status = "✅"
            else:
                is_correct = False
                status = "❌"
            
            print(f"{status} [{i:2d}] {score:.4f} ({predicted:^4}) | {text}")
            if not is_correct and expected != "中性":
                print(f"     预期: {expected}")
            
        except Exception as e:
            print(f"❌ [{i:2d}] 测试失败: {e}")
    
    # 计算准确率（排除中性样本）
    non_neutral = sum(1 for _, expected in test_cases if expected != "中性")
    accuracy = correct / non_neutral if non_neutral > 0 else 0
    
    print("-" * 60)
    print(f"📊 测试结果: {correct}/{non_neutral} 正确")
    print(f"🎯 准确率: {accuracy:.2%}")
    
    if accuracy >= 0.8:
        print("🎉 优秀！模型表现很好")
    elif accuracy >= 0.6:
        print("👍 良好！模型表现不错")
    elif accuracy >= 0.4:
        print("😐 一般！模型需要改进")
    else:
        print("😞 较差！建议重新训练")
    
    return accuracy

def dataset_evaluation():
    """使用测试数据集进行评估"""
    print("\n" + "="*60)
    print("📊 测试数据集评估")
    print("="*60)
    
    # 检查测试文件是否存在
    test_file = 'test.csv'
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return None
    
    try:
        # 读取测试数据
        print("加载测试数据...")
        df = pd.read_csv(test_file, encoding='utf-8')
        
        if 'content' not in df.columns or 'sentiment' not in df.columns:
            print("❌ 测试文件缺少必要的列 (content, sentiment)")
            return None
        
        # 样本数量控制（避免测试时间过长）
        max_samples = 1000
        if len(df) > max_samples:
            print(f"数据量较大，随机采样 {max_samples} 个样本进行测试")
            df = df.sample(n=max_samples, random_state=42)
        
        print(f"测试样本数: {len(df)}")
        
        # 标签映射
        label_mapping = {
            '负面': 0, '消极': 0, '负向': 0, 'negative': 0,
            '正面': 1, '积极': 1, '正向': 1, 'positive': 1,
            '中性': 2, '中立': 2, 'neutral': 2
        }
        
        # 处理测试数据
        test_texts = []
        test_labels = []
        
        for _, row in df.iterrows():
            text = str(row['content']).strip()
            label_str = str(row['sentiment']).strip().lower()
            
            if label_str in label_mapping and len(text) > 0:
                test_texts.append(text)
                test_labels.append(label_mapping[label_str])
        
        if len(test_texts) == 0:
            print("❌ 没有有效的测试样本")
            return None
        
        print(f"有效测试样本: {len(test_texts)}")
        
        # 统计标签分布
        label_counts = {}
        for label in test_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        label_names = {0: "负面", 1: "正面", 2: "中性"}
        print("数据分布:")
        for label, count in label_counts.items():
            percentage = count / len(test_labels) * 100
            print(f"  {label_names[label]}: {count} ({percentage:.1f}%)")
        
        # 开始评估
        print("\n开始评估...")
        
        correct = 0
        predictions = []
        
        for text, true_label in tqdm(zip(test_texts, test_labels), total=len(test_texts), desc="评估进度"):
            try:
                s = SnowNLP(text)
                score = s.sentiments
                
                # 三分类预测
                if score > 0.6:
                    pred_label = 1  # 正面
                elif score < 0.4:
                    pred_label = 0  # 负面
                else:
                    pred_label = 2  # 中性
                
                predictions.append(pred_label)
                
                if pred_label == true_label:
                    correct += 1
                    
            except Exception as e:
                # 如果预测失败，随机分配一个标签
                predictions.append(random.choice([0, 1, 2]))
        
        # 计算总体准确率
        accuracy = correct / len(test_texts)
        
        # 计算各类别准确率
        class_accuracy = {}
        for class_label in [0, 1, 2]:
            class_correct = 0
            class_total = 0
            for true, pred in zip(test_labels, predictions):
                if true == class_label:
                    class_total += 1
                    if pred == class_label:
                        class_correct += 1
            
            if class_total > 0:
                class_accuracy[class_label] = class_correct / class_total
            else:
                class_accuracy[class_label] = 0
        
        # 输出结果
        print(f"\n📊 评估结果:")
        print(f"总体准确率: {accuracy:.2%} ({correct}/{len(test_texts)})")
        print(f"各类别准确率:")
        for label, acc in class_accuracy.items():
            total_for_class = sum(1 for l in test_labels if l == label)
            print(f"  {label_names[label]}: {acc:.2%} (样本数: {total_for_class})")
        
        return accuracy
        
    except Exception as e:
        print(f"❌ 数据集评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def sentiment_distribution_test():
    """情感得分分布测试"""
    print("\n" + "="*60)
    print("📈 情感得分分布测试")
    print("="*60)
    
    # 测试不同类型的文本，观察得分分布
    test_groups = {
        "强烈正面": [
            "太棒了！完美的产品！",
            "非常满意，强烈推荐！",
            "质量超赞，爱死了！",
            "绝对的好评，完美体验！"
        ],
        "一般正面": [
            "还不错，比较满意",
            "质量可以，值得购买",
            "总体来说还行",
            "基本满足需求"
        ],
        "中性": [
            "一般般，没什么特别的",
            "收到了，还没用",
            "和描述差不多",
            "普通的产品"
        ],
        "一般负面": [
            "有点失望，质量一般",
            "不太满意，有待改进",
            "感觉不值这个价",
            "用起来不太方便"
        ],
        "强烈负面": [
            "太差了！完全不推荐！",
            "质量糟糕，浪费钱！",
            "服务态度恶劣！",
            "用了就后悔，垃圾产品！"
        ]
    }
    
    for group_name, texts in test_groups.items():
        scores = []
        print(f"\n{group_name}组:")
        
        for text in texts:
            try:
                s = SnowNLP(text)
                score = s.sentiments
                scores.append(score)
                print(f"  {score:.4f} | {text}")
            except Exception as e:
                print(f"  ERROR  | {text} ({e})")
        
        if scores:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            print(f"  → 平均: {avg_score:.4f}, 范围: {min_score:.4f} - {max_score:.4f}")

def interactive_test():
    """交互式测试"""
    print("\n" + "="*60)
    print("🎮 交互式测试")
    print("="*60)
    print("输入文本进行情感分析测试 (输入 'quit' 退出)")
    print("-" * 60)
    
    while True:
        try:
            text = input("\n请输入测试文本: ").strip()
            
            if text.lower() in ['quit', 'exit', '退出', 'q']:
                print("退出交互式测试")
                break
            
            if not text:
                print("请输入有效文本")
                continue
            
            s = SnowNLP(text)
            score = s.sentiments
            
            if score > 0.6:
                sentiment = "正面 😊"
            elif score < 0.4:
                sentiment = "负面 😞"
            else:
                sentiment = "中性 😐"
            
            print(f"得分: {score:.4f} | 情感: {sentiment}")
            
            # 提供一些额外信息
            if score > 0.8:
                print("💡 强烈正面情感")
            elif score < 0.2:
                print("💡 强烈负面情感")
            elif 0.45 <= score <= 0.55:
                print("💡 情感模糊，接近中性")
            
        except KeyboardInterrupt:
            print("\n退出交互式测试")
            break
        except Exception as e:
            print(f"测试失败: {e}")

def model_info():
    """显示模型信息"""
    print("="*60)
    print("ℹ️  SnowNLP模型信息")
    print("="*60)
    
    try:
        import snownlp
        snownlp_dir = os.path.dirname(snownlp.__file__)
        sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
        
        print(f"SnowNLP安装路径: {snownlp_dir}")
        print(f"Sentiment模块路径: {sentiment_dir}")
        
        # 检查模型文件
        model_files = ['sentiment.marshal', 'sentiment.marshal.3']
        for fname in model_files:
            fpath = os.path.join(sentiment_dir, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                mtime = os.path.getmtime(fpath)
                mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                print(f"模型文件: {fname}")
                print(f"  大小: {size:,} 字节")
                print(f"  修改时间: {mtime_str}")
                
                # 检查是否是备份文件
                backup_files = [f for f in os.listdir(sentiment_dir) if fname in f and 'backup' in f]
                if backup_files:
                    print(f"  备份文件: {len(backup_files)} 个")
        
        # 简单测试
        test_text = "测试文本"
        s = SnowNLP(test_text)
        score = s.sentiments
        print(f"\n快速测试: '{test_text}' → {score:.4f}")
        
    except Exception as e:
        print(f"获取模型信息失败: {e}")

def main():
    """主函数"""
    print("🚀 SnowNLP新训练模型测试工具")
    print("="*60)
    print("请选择测试模式:")
    print("1. 基础情感分析测试")
    print("2. 测试数据集评估")
    print("3. 情感得分分布测试")
    print("4. 交互式测试")
    print("5. 全部测试")
    print("6. 模型信息")
    print("="*60)
    
    while True:
        choice = input("请选择 (1-6): ").strip()
        
        if choice == '1':
            basic_sentiment_test()
            break
        elif choice == '2':
            dataset_evaluation()
            break
        elif choice == '3':
            sentiment_distribution_test()
            break
        elif choice == '4':
            interactive_test()
            break
        elif choice == '5':
            # 运行所有测试
            model_info()
            accuracy1 = basic_sentiment_test()
            accuracy2 = dataset_evaluation()
            sentiment_distribution_test()
            
            print("\n" + "="*60)
            print("📋 测试总结")
            print("="*60)
            if accuracy1 is not None:
                print(f"基础测试准确率: {accuracy1:.2%}")
            if accuracy2 is not None:
                print(f"数据集测试准确率: {accuracy2:.2%}")
            
            if accuracy1 and accuracy2:
                avg_accuracy = (accuracy1 + accuracy2) / 2
                print(f"平均准确率: {avg_accuracy:.2%}")
                
                if avg_accuracy >= 0.75:
                    print("🎉 模型表现优秀！")
                elif avg_accuracy >= 0.6:
                    print("👍 模型表现良好！")
                else:
                    print("😐 模型需要进一步优化")
            
            break
        elif choice == '6':
            model_info()
            break
        else:
            print("无效选择，请输入1-6")

if __name__ == "__main__":
    main() 