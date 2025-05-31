# -*- coding: utf-8 -*-
"""
手动替换SnowNLP模型文件
使用已找到的custom_sentiment.marshal.3文件
"""

import os
import shutil
import sys

def manual_replace_model():
    """手动替换模型文件"""
    print("="*50)
    print("🔧 手动模型文件替换")
    print("="*50)
    
    # 1. 检查源文件
    source_file = 'custom_sentiment.marshal.3'
    if not os.path.exists(source_file):
        print(f"❌ 源模型文件不存在: {source_file}")
        return False
    
    file_size = os.path.getsize(source_file)
    print(f"✅ 找到源模型文件: {source_file} ({file_size} 字节)")
    
    if file_size < 100000:  # 小于100KB可能不是有效模型
        print("⚠️ 警告：文件大小较小，可能不是有效的模型文件")
        proceed = input("是否继续? (y/n): ").strip().lower()
        if proceed != 'y':
            return False
    
    # 2. 获取目标路径
    try:
        import snownlp
        snownlp_dir = os.path.dirname(snownlp.__file__)
        sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
        
        print(f"SnowNLP目录: {snownlp_dir}")
        print(f"Sentiment目录: {sentiment_dir}")
        
        # 3. 查找目标文件
        target_files = []
        for fname in ['sentiment.marshal', 'sentiment.marshal.3']:
            fpath = os.path.join(sentiment_dir, fname)
            if os.path.exists(fpath):
                target_files.append(fpath)
                print(f"找到目标文件: {fpath}")
        
        if not target_files:
            print("❌ 未找到目标模型文件")
            return False
        
        # 4. 备份原文件
        for target_file in target_files:
            backup_file = target_file + '.backup_manual'
            if not os.path.exists(backup_file):
                shutil.copy2(target_file, backup_file)
                print(f"✅ 备份完成: {backup_file}")
            else:
                print(f"备份已存在: {backup_file}")
        
        # 5. 复制新模型
        success_count = 0
        for target_file in target_files:
            try:
                shutil.copy2(source_file, target_file)
                new_size = os.path.getsize(target_file)
                print(f"✅ 复制成功: {source_file} → {target_file} ({new_size} 字节)")
                success_count += 1
            except Exception as e:
                print(f"❌ 复制失败 {target_file}: {e}")
        
        if success_count > 0:
            print(f"\n🎉 成功替换 {success_count} 个模型文件！")
            return True
        else:
            print("\n❌ 所有文件复制都失败了")
            return False
            
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_replaced_model():
    """测试替换后的模型"""
    print("\n" + "="*30)
    print("🧪 测试新模型")
    print("="*30)
    
    test_cases = [
        ("这个产品质量很好，很满意", "正面"),
        ("服务态度很差，非常不满意", "负面"),
        ("价格合理，性价比不错", "正面"),
        ("物流很慢，包装也不好", "负面"),
        ("还可以吧", "中性")
    ]
    
    try:
        from snownlp import SnowNLP
        
        print("测试结果:")
        for text, expected in test_cases:
            s = SnowNLP(text)
            score = s.sentiments
            predicted = "正面" if score > 0.5 else "负面"
            status = "✅" if predicted == expected or expected == "中性" else "❌"
            print(f"{status} '{text}' → {score:.4f} ({predicted})")
        
        print("\n✅ 模型测试完成")
        print("💡 提示：重启Python解释器可以确保完全使用新模型")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def main():
    """主函数"""
    print("SnowNLP模型手动替换工具")
    print("使用已找到的custom_sentiment.marshal.3文件")
    print("="*50)
    
    # 执行替换
    if manual_replace_model():
        print("\n重要提示：")
        print("1. 模型文件已成功替换")
        print("2. 建议重启Python解释器以确保使用新模型")
        print("3. 新模型基于您的训练数据，应该有更好的效果")
        
        # 询问是否测试
        test_now = input("\n是否现在测试新模型? (y/n): ").strip().lower()
        if test_now == 'y':
            print("\n注意：这个测试可能仍使用旧模型缓存")
            print("要确保使用新模型，请重启Python解释器后测试")
            test_replaced_model()
        
        print("\n📋 替换完成总结:")
        print("✅ 找到了有效的训练模型文件")
        print("✅ 成功替换了系统模型文件")
        print("✅ 创建了原文件备份")
        print("\n🎯 下一步：重启Python解释器，测试新模型效果")
        
    else:
        print("\n❌ 模型替换失败")
        print("\n可能的解决方案:")
        print("1. 以管理员权限运行此脚本")
        print("2. 检查是否有其他Python进程在使用SnowNLP")
        print("3. 手动复制文件（详见说明）")
        
        print("\n手动复制说明:")
        print("1. 复制 custom_sentiment.marshal.3")
        print("2. 导航到 SnowNLP 安装目录的 sentiment 文件夹")
        print("3. 备份原 sentiment.marshal 和 sentiment.marshal.3")
        print("4. 将复制的文件重命名并替换原文件")

if __name__ == "__main__":
    main() 