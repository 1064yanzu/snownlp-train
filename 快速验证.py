# -*- coding: utf-8 -*-
"""
快速验证SnowNLP模型是否成功替换
"""

from snownlp import SnowNLP
import time
import os

def quick_test():
    """快速测试新模型"""
    print("🚀 SnowNLP模型快速验证")
    print("="*50)
    
    # 简单测试用例
    test_cases = [
        "这个产品质量很好，非常满意！",
        "服务态度太差了，很不满意",
        "还可以吧，一般般",
        "物流速度很快，包装也不错",
        "价格有点贵，但质量确实好"
    ]
    
    print("测试结果:")
    print("-" * 50)
    
    for i, text in enumerate(test_cases, 1):
        try:
            s = SnowNLP(text)
            score = s.sentiments
            
            if score > 0.6:
                sentiment = "正面 😊"
            elif score < 0.4:
                sentiment = "负面 😞"
            else:
                sentiment = "中性 😐"
            
            print(f"{i}. {score:.4f} ({sentiment:^8}) | {text}")
            
        except Exception as e:
            print(f"{i}. ERROR: {e}")
    
    print("-" * 50)
    print("✅ 快速验证完成")

def check_model_info():
    """检查模型文件信息"""
    print("\n📁 模型文件信息:")
    print("-" * 30)
    
    try:
        import snownlp
        snownlp_dir = os.path.dirname(snownlp.__file__)
        sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
        
        model_files = ['sentiment.marshal', 'sentiment.marshal.3']
        for fname in model_files:
            fpath = os.path.join(sentiment_dir, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                mtime = os.path.getmtime(fpath)
                mtime_str = time.strftime('%m-%d %H:%M', time.localtime(mtime))
                print(f"{fname}: {size:,}字节 ({mtime_str})")
        
    except Exception as e:
        print(f"检查失败: {e}")

if __name__ == "__main__":
    quick_test()
    check_model_info()
    
    print(f"\n💡 提示：")
    print(f"- 如果得分都相似，可能还在使用旧模型")
    print(f"- 建议重启Python解释器后再测试")
    print(f"- 运行 '测试新模型.py' 进行完整测试") 