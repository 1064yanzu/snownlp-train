# -*- coding: utf-8 -*-
"""
SnowNLP情感分析训练工具 - 命令行版本
适用于Linux云环境、无头服务器等无图形界面环境
"""

import os
import sys
import time
import shutil
import argparse
import pandas as pd
from glob import glob
from datetime import datetime
import json
import random

def print_banner():
    """打印程序横幅"""
    print("=" * 60)
    print("🚀 SnowNLP情感分析训练工具 - 命令行版本")
    print("=" * 60)
    print("🌟 专为Linux云环境和无头服务器设计")
    print("⚡ 支持完整的模型训练、测试和管理功能")
    print("=" * 60)

def check_dependencies():
    """检查依赖"""
    print("\n🔍 检查依赖库...")
    dependencies = {
        'pandas': 'pandas',
        'snownlp': 'snownlp', 
        'tqdm': 'tqdm',
        'numpy': 'numpy',
        'jieba': 'jieba'
    }
    
    missing = []
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name} 已安装")
        except ImportError:
            print(f"❌ {name} 未安装")
            missing.append(module)
    
    if missing:
        print(f"\n📦 安装缺失依赖...")
        import subprocess
        for module in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                print(f"✅ {module} 安装成功")
            except Exception as e:
                print(f"❌ {module} 安装失败: {e}")
                return False
    
    print("🎉 所有依赖检查完成!")
    return True

def find_data_files():
    """查找数据文件"""
    print("\n📁 查找数据文件...")
    
    # 训练文件模式
    train_patterns = ['train.csv', '训练*.csv', '*train*.csv']
    train_files = []
    for pattern in train_patterns:
        train_files.extend(glob(pattern))
    
    # 测试文件模式
    test_patterns = ['test.csv', '测试*.csv', '*test*.csv']
    test_files = []
    for pattern in test_patterns:
        test_files.extend(glob(pattern))
    
    print(f"📊 找到 {len(train_files)} 个训练文件")
    for f in train_files:
        size = os.path.getsize(f)
        print(f"  - {f} ({size:,} 字节)")
    
    print(f"📊 找到 {len(test_files)} 个测试文件")
    for f in test_files:
        size = os.path.getsize(f)
        print(f"  - {f} ({size:,} 字节)")
    
    return train_files, test_files

def load_data_with_progress(filepaths, data_type="数据", neutral_strategy="balance"):
    """加载数据并显示进度"""
    from tqdm import tqdm
    
    print(f"\n📂 加载{data_type}文件...")
    print(f"🔧 中性数据处理策略: {neutral_strategy}")
    
    # 标签映射
    label_mapping = {
        # 中文标签
        '负面': 0, '消极': 0, '负向': 0, '差': 0, '不好': 0, '坏': 0,
        '正面': 1, '积极': 1, '正向': 1, '好': 1, '很好': 1, '棒': 1,
        '中性': 'neutral', '中立': 'neutral', '一般': 'neutral',
        
        # 英文标签
        'negative': 0, 'bad': 0, 'poor': 0,
        'positive': 1, 'good': 1, 'excellent': 1,
        'neutral': 'neutral',
        
        # 数字标签
        '0': 0, '1': 1, '2': 'neutral',
        0: 0, 1: 1, 2: 'neutral',
    }
    
    all_texts, all_labels = [], []
    neutral_texts = []
    total_rows = 0
    
    for path in tqdm(filepaths, desc="处理文件", unit="文件"):
        if not os.path.exists(path):
            continue
            
        # 尝试不同编码
        df = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
            try:
                df = pd.read_csv(path, encoding=encoding)
                print(f"  ✅ {os.path.basename(path)}: 使用编码 {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"  ❌ 无法读取: {path}")
            continue
        
        if 'content' not in df.columns or 'sentiment' not in df.columns:
            print(f"  ❌ 缺少必要列: {path}")
            continue
        
        total_rows += len(df)
        texts = df['content'].astype(str).tolist()
        
        # 处理标签
        valid_indices = []
        neutral_indices = []
        
        for i, label in enumerate(tqdm(df['sentiment'], desc=f"处理标签", leave=False)):
            if pd.isna(label):
                continue
            
            label_key = int(label) if isinstance(label, (int, float)) else str(label).strip().lower()
            mapped = label_mapping.get(label_key, None)
            
            if mapped == 'neutral':
                neutral_indices.append(i)
            elif mapped is not None:
                all_labels.append(mapped)
                valid_indices.append(i)
        
        all_texts.extend([texts[i] for i in valid_indices])
        neutral_texts.extend([texts[i] for i in neutral_indices])
    
    # 处理中性数据
    current_pos = sum(1 for label in all_labels if label == 1)
    current_neg = sum(1 for label in all_labels if label == 0)
    
    print(f"\n📊 原始数据统计:")
    print(f"  正面样本: {current_pos:,}")
    print(f"  负面样本: {current_neg:,}")  
    print(f"  中性样本: {len(neutral_texts):,}")
    print(f"  总行数: {total_rows:,}")
    
    if neutral_texts and neutral_strategy != 'exclude':
        print(f"\n🔄 处理中性样本...")
        
        if neutral_strategy == 'balance':
            # 平衡策略：分配给较少的类别
            if current_pos < current_neg:
                all_texts.extend(neutral_texts)
                all_labels.extend([1] * len(neutral_texts))
                print(f"  ✅ {len(neutral_texts):,}个中性样本分配给正面类别")
            else:
                all_texts.extend(neutral_texts)
                all_labels.extend([0] * len(neutral_texts))
                print(f"  ✅ {len(neutral_texts):,}个中性样本分配给负面类别")
        elif neutral_strategy == 'split':
            # 按比例分配
            random.shuffle(neutral_texts)
            split_point = int(len(neutral_texts) * 0.7)
            pos_neutrals = neutral_texts[:split_point]
            neg_neutrals = neutral_texts[split_point:]
            
            all_texts.extend(pos_neutrals + neg_neutrals)
            all_labels.extend([1] * len(pos_neutrals) + [0] * len(neg_neutrals))
            print(f"  ✅ 中性样本分配: {len(pos_neutrals):,}个给正面, {len(neg_neutrals):,}个给负面")
    
    final_pos = sum(1 for label in all_labels if label == 1)
    final_neg = sum(1 for label in all_labels if label == 0)
    utilization = (len(all_texts) / total_rows * 100) if total_rows > 0 else 0
    
    print(f"\n📈 最终数据统计:")
    print(f"  正面样本: {final_pos:,}")
    print(f"  负面样本: {final_neg:,}")
    print(f"  总样本数: {len(all_texts):,}")
    print(f"  数据利用率: {utilization:.1f}%")
    
    return all_texts, all_labels

def create_sentiment_files(texts, labels, pos_path, neg_path):
    """创建语料文件"""
    from tqdm import tqdm
    
    print(f"\n📝 创建语料文件...")
    os.makedirs(os.path.dirname(pos_path), exist_ok=True)
    os.makedirs(os.path.dirname(neg_path), exist_ok=True)
    
    with open(pos_path, 'w', encoding='utf-8') as f_pos, \
         open(neg_path, 'w', encoding='utf-8') as f_neg:
        
        pos_count, neg_count = 0, 0
        
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="创建语料"):
            clean_text = text.replace('\n', '').replace('\r', '').strip()
            if len(clean_text) > 0:
                if label == 1:
                    f_pos.write(clean_text + '\n')
                    pos_count += 1
                elif label == 0:
                    f_neg.write(clean_text + '\n')
                    neg_count += 1
    
    print(f"  ✅ 正面语料: {pos_count:,} 个样本")
    print(f"  ✅ 负面语料: {neg_count:,} 个样本")
    
    return pos_count, neg_count

def train_model(neg_path, pos_path):
    """训练模型"""
    from snownlp import sentiment
    from tqdm import tqdm
    
    print(f"\n🧠 开始模型训练...")
    print("⚠️  这可能需要几分钟时间，请耐心等待...")
    
    start_time = time.time()
    
    try:
        # 显示进度提示
        print("🔄 SnowNLP核心算法训练中...")
        sentiment.train(neg_path, pos_path)
        
        elapsed = time.time() - start_time
        print(f"✅ 模型训练完成! 耗时: {elapsed:.1f}秒")
        
        # 查找生成的模型文件
        model_files = []
        for pattern in ['*.marshal*', 'custom_sentiment.*']:
            model_files.extend(glob(pattern))
        
        if model_files:
            largest_file = max(model_files, key=os.path.getsize)
            size = os.path.getsize(largest_file)
            print(f"📦 找到模型文件: {largest_file} ({size:,} 字节)")
            return largest_file
        else:
            print("❌ 未找到生成的模型文件")
            return None
            
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return None

def replace_model(model_file):
    """替换系统模型"""
    print(f"\n🔄 部署新模型...")
    
    try:
        import snownlp
        snownlp_dir = os.path.dirname(snownlp.__file__)
        sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
        
        print(f"📁 SnowNLP目录: {sentiment_dir}")
        
        # 查找目标文件
        target_files = []
        for fname in ['sentiment.marshal', 'sentiment.marshal.3']:
            fpath = os.path.join(sentiment_dir, fname)
            if os.path.exists(fpath):
                target_files.append(fpath)
        
        if not target_files:
            print("❌ 未找到目标模型文件")
            return False
        
        # 备份原文件
        backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        for target_file in target_files:
            backup_file = f"{target_file}.backup_{backup_time}"
            shutil.copy2(target_file, backup_file)
            print(f"📋 备份: {os.path.basename(backup_file)}")
        
        # 替换模型
        success_count = 0
        for target_file in target_files:
            try:
                shutil.copy2(model_file, target_file)
                size = os.path.getsize(target_file)
                print(f"✅ 替换: {os.path.basename(target_file)} ({size:,} 字节)")
                success_count += 1
            except Exception as e:
                print(f"❌ 替换失败 {os.path.basename(target_file)}: {e}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 模型部署失败: {e}")
        return False

def evaluate_model(texts, labels, sample_size=1000):
    """评估模型"""
    from snownlp import SnowNLP
    from tqdm import tqdm
    
    print(f"\n📊 模型性能评估...")
    
    # 如果数据量很大，随机采样
    if len(texts) > sample_size:
        print(f"📝 数据量较大，随机采样 {sample_size:,} 个样本进行评估")
        indices = random.sample(range(len(texts)), sample_size)
        eval_texts = [texts[i] for i in indices]
        eval_labels = [labels[i] for i in indices]
    else:
        eval_texts = texts
        eval_labels = labels
    
    correct = 0
    total = len(eval_texts)
    
    for text, true_label in tqdm(zip(eval_texts, eval_labels), 
                                total=total, desc="评估模型"):
        try:
            s = SnowNLP(text)
            score = s.sentiments
            pred_label = 1 if score > 0.5 else 0
            
            if pred_label == true_label:
                correct += 1
        except:
            continue
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n📈 评估结果:")
    print(f"  测试样本: {total:,}")
    print(f"  正确预测: {correct:,}")
    print(f"  准确率: {accuracy:.2%}")
    
    if accuracy >= 0.8:
        print("🎉 模型表现优秀!")
    elif accuracy >= 0.6:
        print("👍 模型表现良好!")
    else:
        print("😐 模型需要改进")
    
    return accuracy

def interactive_test():
    """交互式测试"""
    from snownlp import SnowNLP
    
    print(f"\n🎮 交互式测试模式")
    print("输入文本进行情感分析，输入 'quit' 退出")
    print("-" * 50)
    
    while True:
        try:
            text = input("\n请输入测试文本: ").strip()
            
            if text.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 退出测试模式")
                break
            
            if not text:
                continue
            
            s = SnowNLP(text)
            score = s.sentiments
            
            if score > 0.6:
                sentiment = "正面 😊"
                color = "绿色"
            elif score < 0.4:
                sentiment = "负面 😞"
                color = "红色"
            else:
                sentiment = "中性 😐"
                color = "黄色"
            
            print(f"📊 分析结果:")
            print(f"  得分: {score:.4f}")
            print(f"  情感: {sentiment}")
            
            if score > 0.8:
                print("  强度: 强烈正面")
            elif score < 0.2:
                print("  强度: 强烈负面")
            
        except KeyboardInterrupt:
            print("\n👋 退出测试模式")
            break
        except Exception as e:
            print(f"❌ 分析失败: {e}")

def quick_test():
    """快速验证测试"""
    from snownlp import SnowNLP
    
    print(f"\n⚡ 快速验证测试")
    
    test_cases = [
        ("这个产品质量非常好，强烈推荐！", "正面"),
        ("服务态度太差了，很不满意", "负面"),
        ("还可以吧，一般般", "中性"),
        ("物流速度很快，包装也不错", "正面"),
        ("价格有点贵，但质量确实好", "正面"),
        ("用了几天就坏了，太失望", "负面"),
        ("性价比很高，值得购买", "正面"),
        ("客服态度恶劣，很生气", "负面")
    ]
    
    correct = 0
    total = 0
    
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
            
            is_correct = predicted == expected or expected == "中性"
            if expected != "中性":
                total += 1
                if is_correct:
                    correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} [{i}] {score:.4f} ({predicted}) | {text}")
            
        except Exception as e:
            print(f"❌ [{i}] 测试失败: {e}")
    
    if total > 0:
        accuracy = correct / total
        print(f"\n📊 快速测试结果: {correct}/{total} 正确，准确率: {accuracy:.2%}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SnowNLP情感分析训练工具 - 命令行版本")
    parser.add_argument('--train', action='store_true', help='执行模型训练')
    parser.add_argument('--test', action='store_true', help='快速验证测试')
    parser.add_argument('--interactive', action='store_true', help='交互式测试')
    parser.add_argument('--eval', action='store_true', help='模型评估')
    parser.add_argument('--neutral-strategy', choices=['balance', 'split', 'exclude'], 
                       default='balance', help='中性数据处理策略')
    
    args = parser.parse_args()
    
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 如果没有指定参数，显示菜单
    if not any([args.train, args.test, args.interactive, args.eval]):
        while True:
            print(f"\n🎯 请选择操作:")
            print("1. 🚀 训练新模型")
            print("2. ⚡ 快速验证测试")
            print("3. 📊 模型评估") 
            print("4. 🎮 交互式测试")
            print("5. 🔍 查看数据文件信息")
            print("0. 🚪 退出程序")
            
            try:
                choice = input("\n请输入选择 (0-5): ").strip()
                
                if choice == '0':
                    print("👋 再见!")
                    break
                elif choice == '1':
                    args.train = True
                    break
                elif choice == '2':
                    args.test = True
                    break
                elif choice == '3':
                    args.eval = True
                    break
                elif choice == '4':
                    args.interactive = True
                    break
                elif choice == '5':
                    find_data_files()
                    continue
                else:
                    print("❌ 无效选择，请重新输入")
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
    
    # 执行选择的操作
    if args.train:
        # 训练模型
        train_files, test_files = find_data_files()
        
        if not train_files:
            print("❌ 未找到训练数据文件")
            return 1
        
        # 加载训练数据
        train_texts, train_labels = load_data_with_progress(
            train_files, "训练", args.neutral_strategy)
        
        if not train_texts:
            print("❌ 训练数据加载失败")
            return 1
        
        # 创建语料文件
        pos_path = 'temp_data/pos.txt'
        neg_path = 'temp_data/neg.txt'
        pos_count, neg_count = create_sentiment_files(train_texts, train_labels, pos_path, neg_path)
        
        if pos_count == 0 or neg_count == 0:
            print("❌ 正面或负面样本数量为0，无法训练")
            return 1
        
        # 训练模型
        model_file = train_model(neg_path, pos_path)
        if not model_file:
            return 1
        
        # 替换模型
        if replace_model(model_file):
            print(f"\n🎉 模型训练和部署完成!")
            
            # 如果有测试数据，进行评估
            if test_files:
                test_texts, test_labels = load_data_with_progress(test_files, "测试")
                if test_texts:
                    evaluate_model(test_texts, test_labels)
        
        # 清理临时文件
        try:
            if os.path.exists('temp_data'):
                shutil.rmtree('temp_data')
        except:
            pass
    
    elif args.test:
        quick_test()
    
    elif args.eval:
        train_files, test_files = find_data_files()
        if test_files:
            test_texts, test_labels = load_data_with_progress(test_files, "测试")
            if test_texts:
                evaluate_model(test_texts, test_labels)
        else:
            print("❌ 未找到测试数据文件")
    
    elif args.interactive:
        interactive_test()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 