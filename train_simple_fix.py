# -*- coding: utf-8 -*-
"""
SnowNLP训练问题极简修复版本
专注解决模型保存问题
"""

import os
import sys
import shutil
import time
from snownlp import sentiment
from snownlp import SnowNLP

def simple_model_replacement():
    """
    极简模型替换方案
    直接训练后查找并复制模型文件
    """
    print("="*50)
    print("🔧 极简模型替换方案")
    print("="*50)
    
    try:
        # 1. 获取SnowNLP路径
        import snownlp
        snownlp_dir = os.path.dirname(snownlp.__file__)
        sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
        
        print(f"SnowNLP目录: {snownlp_dir}")
        print(f"Sentiment目录: {sentiment_dir}")
        
        # 2. 找到现有模型文件
        existing_models = []
        for fname in ['sentiment.marshal', 'sentiment.marshal.3']:
            fpath = os.path.join(sentiment_dir, fname)
            if os.path.exists(fpath):
                existing_models.append(fpath)
                # 备份
                backup = fpath + '.backup_simple'
                if not os.path.exists(backup):
                    shutil.copy2(fpath, backup)
                    print(f"✅ 备份完成: {backup}")
        
        if not existing_models:
            print("❌ 未找到现有模型文件")
            return False
        
        # 3. 重新训练（这次我们知道训练是成功的）
        print("开始重新训练...")
        
        # 检查语料文件
        pos_file = 'temp_data/pos.txt'
        neg_file = 'temp_data/neg.txt'
        
        if not os.path.exists(pos_file) or not os.path.exists(neg_file):
            print("❌ 语料文件不存在，请先运行完整训练脚本")
            return False
        
        # 训练
        sentiment.train(neg_file, pos_file)
        print("✅ 训练完成")
        
        # 4. 查找新生成的模型文件
        print("查找新生成的模型文件...")
        
        # 可能的位置
        search_paths = [
            os.getcwd(),  # 当前目录
            sentiment_dir,  # sentiment目录
            snownlp_dir,   # snownlp根目录
            os.path.expanduser('~'),  # 用户目录
            os.path.join(os.getcwd(), 'temp_data')  # temp_data目录
        ]
        
        found_models = []
        for search_dir in search_paths:
            if not os.path.exists(search_dir):
                continue
                
            for fname in ['sentiment.marshal', 'sentiment.marshal.3']:
                fpath = os.path.join(search_dir, fname)
                if os.path.exists(fpath):
                    # 检查文件是否是最近修改的（5分钟内）
                    mtime = os.path.getmtime(fpath)
                    if time.time() - mtime < 300:  # 5分钟
                        size = os.path.getsize(fpath)
                        if size > 1000:  # 至少1KB
                            found_models.append((fpath, size, mtime))
                            print(f"找到新模型: {fpath} ({size} 字节)")
        
        if not found_models:
            print("❌ 未找到新生成的模型文件")
            print("尝试手动查找...")
            
            # 列出所有可能的文件
            for search_dir in search_paths:
                if os.path.exists(search_dir):
                    try:
                        files = os.listdir(search_dir)
                        marshal_files = [f for f in files if 'marshal' in f.lower()]
                        if marshal_files:
                            print(f"目录 {search_dir} 中的marshal文件: {marshal_files}")
                    except:
                        pass
            return False
        
        # 5. 选择最新的模型文件
        found_models.sort(key=lambda x: x[2], reverse=True)  # 按时间排序
        best_model = found_models[0]
        source_file = best_model[0]
        
        print(f"选择模型: {source_file}")
        
        # 6. 复制到系统位置
        success_count = 0
        for target_model in existing_models:
            try:
                shutil.copy2(source_file, target_model)
                print(f"✅ 复制成功: {source_file} → {target_model}")
                success_count += 1
            except Exception as e:
                print(f"❌ 复制失败: {e}")
        
        if success_count > 0:
            print(f"✅ 成功替换 {success_count} 个模型文件")
            return True
        else:
            print("❌ 所有复制都失败了")
            return False
            
    except Exception as e:
        print(f"❌ 极简替换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_model():
    """测试新模型是否工作"""
    print("\n" + "="*30)
    print("🧪 测试新模型")
    print("="*30)
    
    test_texts = [
        "这个产品很好用，质量不错",
        "服务态度很差，很不满意",
        "价格合理，性价比高",
        "快递很慢，包装破损"
    ]
    
    try:
        for text in test_texts:
            s = SnowNLP(text)
            score = s.sentiments
            sentiment_label = "正面" if score > 0.5 else "负面"
            print(f"'{text}' → {score:.4f} ({sentiment_label})")
        
        print("✅ 新模型测试正常")
        return True
        
    except Exception as e:
        print(f"❌ 新模型测试失败: {e}")
        return False

def main():
    """主函数"""
    print("SnowNLP训练问题极简修复工具")
    print("专注解决模型保存问题")
    print("="*50)
    
    # 检查是否已经有训练数据
    if not os.path.exists('temp_data/pos.txt'):
        print("❌ 未找到训练数据文件")
        print("请先运行 train_fixed.py 或 train_fixed_v2.py 生成语料文件")
        return
    
    print("✅ 找到训练数据文件")
    
    # 执行极简替换
    if simple_model_replacement():
        print("\n🎉 模型替换成功！")
        
        # 测试新模型
        print("\n重要提示：请重启Python解释器后测试")
        restart = input("是否现在测试新模型？(y/n): ").strip().lower()
        
        if restart == 'y':
            print("\n注意：这个测试可能使用的还是旧模型")
            print("要确保使用新模型，请重启Python解释器")
            test_new_model()
        
        print("\n📋 成功总结:")
        print("✅ 模型训练完成")
        print("✅ 模型文件替换成功")
        print("📝 下一步：重启Python解释器，然后测试新模型")
        
    else:
        print("\n❌ 模型替换失败")
        print("可能的原因：")
        print("1. 权限不足")
        print("2. SnowNLP训练没有生成模型文件")
        print("3. 文件被其他程序占用")
        
        print("\n建议解决方案：")
        print("1. 以管理员权限运行")
        print("2. 关闭所有Python进程后重试")
        print("3. 考虑使用其他情感分析库")

if __name__ == "__main__":
    main() 