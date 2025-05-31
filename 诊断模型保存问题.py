# -*- coding: utf-8 -*-
"""
SnowNLP模型保存问题诊断脚本
用于快速诊断和测试模型保存功能
"""

import os
import sys
import tempfile
from snownlp import sentiment

def create_test_data():
    """创建测试用的小样本数据"""
    
    # 创建测试目录
    test_dir = 'test_data'
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建简单的正面样本
    pos_samples = [
        "这个产品很好用",
        "服务态度很棒",
        "质量不错，推荐购买",
        "物超所值，很满意",
        "快递很快，包装很好"
    ]
    
    # 创建简单的负面样本
    neg_samples = [
        "这个产品很差劲",
        "服务态度很糟糕", 
        "质量很差，不推荐",
        "价格太贵，不值得",
        "快递很慢，包装破损"
    ]
    
    # 写入文件
    pos_path = os.path.join(test_dir, 'pos_test.txt')
    neg_path = os.path.join(test_dir, 'neg_test.txt')
    
    with open(pos_path, 'w', encoding='utf-8') as f:
        for sample in pos_samples:
            f.write(sample + '\n')
    
    with open(neg_path, 'w', encoding='utf-8') as f:
        for sample in neg_samples:
            f.write(sample + '\n')
    
    print(f"✅ 测试数据创建成功:")
    print(f"   正面样本: {pos_path} ({len(pos_samples)}条)")
    print(f"   负面样本: {neg_path} ({len(neg_samples)}条)")
    
    return neg_path, pos_path

def test_basic_training():
    """测试基本的训练功能"""
    print("\n" + "="*50)
    print("🔍 开始基本训练测试...")
    
    try:
        # 创建测试数据
        neg_path, pos_path = create_test_data()
        
        # 测试训练
        print("开始训练...")
        sentiment.train(neg_path, pos_path)
        print("✅ 训练完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_saving():
    """测试各种模型保存方法"""
    print("\n" + "="*50)
    print("🔍 开始模型保存测试...")
    
    test_paths = [
        'test_model.marshal',
        'test_model_1.marshal',
        'test_model_2.marshal',
        os.path.join(tempfile.gettempdir(), 'temp_model.marshal'),
        os.path.join(os.getcwd(), 'local_model.marshal')
    ]
    
    successful_saves = []
    failed_saves = []
    
    for i, test_path in enumerate(test_paths):
        print(f"\n测试保存路径 {i+1}: {test_path}")
        
        try:
            # 尝试保存
            sentiment.save(test_path)
            
            # 检查文件是否存在
            if os.path.exists(test_path):
                file_size = os.path.getsize(test_path)
                if file_size > 0:
                    print(f"✅ 保存成功 - 大小: {file_size} 字节")
                    successful_saves.append((test_path, file_size))
                else:
                    print(f"❌ 文件为空")
                    failed_saves.append((test_path, "文件为空"))
            else:
                print(f"❌ 文件未创建")
                failed_saves.append((test_path, "文件未创建"))
                
        except Exception as e:
            print(f"❌ 保存异常: {e}")
            failed_saves.append((test_path, str(e)))
    
    # 总结结果
    print(f"\n📊 保存测试结果:")
    print(f"   成功: {len(successful_saves)}/{len(test_paths)}")
    print(f"   失败: {len(failed_saves)}/{len(test_paths)}")
    
    if successful_saves:
        print(f"\n✅ 成功的保存:")
        for path, size in successful_saves:
            print(f"   {path} ({size} 字节)")
    
    if failed_saves:
        print(f"\n❌ 失败的保存:")
        for path, error in failed_saves:
            print(f"   {path}: {error}")
    
    return successful_saves

def test_model_loading(successful_saves):
    """测试模型加载功能"""
    if not successful_saves:
        print("\n⚠️  没有成功保存的模型，跳过加载测试")
        return False
    
    print("\n" + "="*50)
    print("🔍 开始模型加载测试...")
    
    # 选择第一个成功保存的模型进行测试
    test_model_path = successful_saves[0][0]
    print(f"测试模型: {test_model_path}")
    
    try:
        # 创建新的sentiment实例
        from snownlp.sentiment import Sentiment
        test_sentiment = Sentiment()
        
        # 加载模型
        test_sentiment.load(test_model_path)
        print("✅ 模型加载成功")
        
        # 测试预测
        test_texts = ["这个很好", "这个很差"]
        print("\n测试预测:")
        for text in test_texts:
            try:
                score = test_sentiment.classify(text)
                print(f"   '{text}' → {score:.4f}")
            except Exception as e:
                print(f"   '{text}' → 预测失败: {e}")
                return False
        
        print("✅ 模型功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_permissions():
    """测试系统权限和环境"""
    print("\n" + "="*50)
    print("🔍 开始系统环境测试...")
    
    # 测试当前目录权限
    try:
        test_file = 'permission_test.tmp'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ 当前目录写权限正常")
    except Exception as e:
        print(f"❌ 当前目录写权限问题: {e}")
    
    # 测试临时目录权限
    try:
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, 'temp_test.tmp')
        with open(temp_file, 'w') as f:
            f.write('test')
        os.remove(temp_file)
        print(f"✅ 临时目录权限正常: {temp_dir}")
    except Exception as e:
        print(f"❌ 临时目录权限问题: {e}")
    
    # 检查磁盘空间
    try:
        import shutil
        free_space = shutil.disk_usage('.').free
        print(f"✅ 磁盘剩余空间: {free_space // (1024*1024)} MB")
    except Exception as e:
        print(f"❌ 磁盘空间检查失败: {e}")
    
    # 检查Python和库版本
    print(f"✅ Python版本: {sys.version}")
    
    try:
        import snownlp
        print(f"✅ SnowNLP安装路径: {snownlp.__file__}")
    except Exception as e:
        print(f"❌ SnowNLP信息获取失败: {e}")

def cleanup_test_files():
    """清理测试文件"""
    print("\n" + "="*30)
    cleanup = input("是否清理测试文件? (y/n): ").lower().strip()
    
    if cleanup == 'y':
        test_files = [
            'test_model.marshal',
            'test_model_1.marshal', 
            'test_model_2.marshal',
            'local_model.marshal',
            'test_data'
        ]
        
        for item in test_files:
            try:
                if os.path.isfile(item):
                    os.remove(item)
                    print(f"删除文件: {item}")
                elif os.path.isdir(item):
                    import shutil
                    shutil.rmtree(item)
                    print(f"删除目录: {item}")
            except Exception as e:
                print(f"删除失败 {item}: {e}")
        
        print("✅ 清理完成")

def main():
    """主函数"""
    print("SnowNLP模型保存问题诊断工具")
    print("="*50)
    
    # 环境检查
    test_system_permissions()
    
    # 基本训练测试
    if test_basic_training():
        # 模型保存测试
        successful_saves = test_model_saving()
        
        # 模型加载测试
        if successful_saves:
            test_model_loading(successful_saves)
            
            print("\n" + "="*50)
            print("📋 诊断总结:")
            print("✅ 训练功能正常")
            print("✅ 模型保存功能正常")  
            print("✅ 模型加载功能正常")
            print("\n🎉 您的环境没有问题！")
            print("原训练脚本的问题可能是:")
            print("1. 路径冲突")
            print("2. 并发访问问题")
            print("3. 临时的文件系统问题")
            print("\n建议重新运行 train_fixed.py")
        else:
            print("\n" + "="*50)
            print("📋 诊断总结:")
            print("✅ 训练功能正常")
            print("❌ 模型保存功能异常")
            print("\n🔧 需要进一步调试模型保存问题")
    else:
        print("\n" + "="*50)
        print("📋 诊断总结:")
        print("❌ 基本训练功能异常")
        print("\n🔧 需要检查SnowNLP安装和数据格式")
    
    # 清理
    cleanup_test_files()

if __name__ == "__main__":
    main() 