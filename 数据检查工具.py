# -*- coding: utf-8 -*-
"""
数据检查工具 - 分析CSV文件中的标签分布
帮助诊断为什么训练数据被大量过滤
"""

import pandas as pd
import os
from collections import Counter

def analyze_csv_file(file_path, text_col='content', label_col='sentiment'):
    """分析单个CSV文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"\n📁 分析文件: {os.path.basename(file_path)}")
    print("=" * 60)
    
    try:
        # 尝试不同编码
        df = None
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ 成功使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("❌ 所有编码都失败")
            return
        
        print(f"📊 总行数: {len(df)}")
        print(f"📝 列名: {list(df.columns)}")
        
        # 检查必要列
        if text_col not in df.columns:
            print(f"❌ 缺少文本列: {text_col}")
            return
        
        if label_col not in df.columns:
            print(f"❌ 缺少标签列: {label_col}")
            return
        
        # 分析标签分布
        print(f"\n🏷️ 标签分布分析:")
        label_counts = Counter()
        
        for i, label in enumerate(df[label_col]):
            if pd.isna(label):
                label_counts['<空值>'] += 1
            elif isinstance(label, (int, float)):
                label_counts[f"数字_{int(label)}"] += 1
            else:
                label_str = str(label).strip()
                label_counts[label_str] += 1
        
        # 显示标签统计
        total_valid = len(df) - label_counts.get('<空值>', 0)
        
        print(f"有效标签数: {total_valid}")
        print(f"空值数量: {label_counts.get('<空值>', 0)}")
        print("\n标签详细分布:")
        
        for label, count in label_counts.most_common():
            percentage = count / len(df) * 100
            print(f"  '{label}': {count} 个 ({percentage:.1f}%)")
        
        # 检查当前映射规则能识别多少
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
            
            # 情感标签
            'angry': 0, 'sad': 0, 'fear': 0,
            'happy': 1, 'surprise': 1,
        }
        
        recognized = 0
        unrecognized = 0
        unrecognized_labels = set()
        
        for i, label in enumerate(df[label_col]):
            if pd.isna(label):
                continue
                
            if isinstance(label, (int, float)):
                label_key = int(label)
            else:
                label_key = str(label).strip().lower()
            
            if label_key in label_mapping:
                recognized += 1
            else:
                unrecognized += 1
                unrecognized_labels.add(str(label))
        
        print(f"\n🔍 映射规则分析:")
        print(f"可识别样本: {recognized} 个 ({recognized/total_valid*100:.1f}%)")
        print(f"不可识别样本: {unrecognized} 个 ({unrecognized/total_valid*100:.1f}%)")
        
        if unrecognized_labels:
            print(f"\n⚠️ 不可识别的标签:")
            for label in sorted(unrecognized_labels):
                count = label_counts.get(label, 0)
                print(f"  '{label}': {count} 个")
        
        # 生成建议
        print(f"\n💡 处理建议:")
        if unrecognized > 0:
            print(f"• 有 {unrecognized} 个样本因标签格式问题被跳过")
            print(f"• 数据利用率: {recognized/total_valid*100:.1f}%")
            
            # 推荐标签映射
            major_unrecognized = [label for label, count in label_counts.most_common() 
                                if label not in ['<空值>'] and 
                                str(label).strip().lower() not in label_mapping and count > 10]
            
            if major_unrecognized:
                print(f"• 建议添加以下标签映射:")
                for label in major_unrecognized[:5]:  # 只显示前5个
                    print(f"  '{label}' -> ? (需要人工判断是正面/负面/中性)")
        else:
            print(f"✅ 所有标签都能被正确识别!")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def analyze_multiple_files():
    """分析多个文件"""
    print("🔍 数据文件标签分析工具")
    print("=" * 60)
    
    # 查找可能的数据文件
    possible_files = []
    
    # 当前目录下的CSV文件
    import glob
    csv_files = glob.glob("*.csv")
    possible_files.extend(csv_files)
    
    if not possible_files:
        print("❌ 未找到CSV文件")
        return
    
    print(f"发现 {len(possible_files)} 个CSV文件:")
    for i, file in enumerate(possible_files, 1):
        print(f"  {i}. {file}")
    
    # 分析每个文件
    for file_path in possible_files:
        analyze_csv_file(file_path)
    
    print(f"\n" + "=" * 60)
    print("📋 总结:")
    print("• 如果发现大量'不可识别标签'，这就是数据被过滤的原因")
    print("• 请将不可识别的标签信息提供给开发者，以完善标签映射")
    print("• 或者可以手动修改数据文件，将标签统一为标准格式")

if __name__ == "__main__":
    analyze_multiple_files() 