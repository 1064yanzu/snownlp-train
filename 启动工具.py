# -*- coding: utf-8 -*-
"""
SnowNLP情感分析训练测试工具 - 智能启动器
自动检测运行环境，选择最佳的界面模式
"""

import sys
import os
import subprocess
import platform
import time

def print_banner():
    """打印程序信息"""
    print("=" * 60)
    print("🚀 SnowNLP情感分析训练测试工具 v3.0")
    print("=" * 60)
    
    # 显示系统信息
    print(f"✅ Python版本: {sys.version}")
    print(f"💻 操作系统: {platform.system()} {platform.release()}")
    print(f"🏠 当前目录: {os.getcwd()}")

def check_dependencies():
    """检查基础依赖"""
    print("🔍 检查依赖库...")
    
    required_packages = {
        'pandas': 'pandas',
        'snownlp': 'snownlp',
        'tqdm': 'tqdm',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy'
    }
    
    optional_packages = {
        'scikit-learn': 'sklearn',
        'jieba': 'jieba'
    }
    
    missing_required = []
    missing_optional = []
    
    # 检查必需包
    for name, module in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {name} 已安装")
        except ImportError:
            print(f"❌ {name} 未安装")
            missing_required.append(name)
    
    # 检查可选包
    for name, module in optional_packages.items():
        try:
            __import__(module)
            print(f"✅ {name} 已安装")
        except ImportError:
            print(f"❌ {name} 未安装")
            missing_optional.append(name)
    
    # 安装缺失的包
    all_missing = missing_required + missing_optional
    if all_missing:
        print(f"\n📦 正在安装缺失的依赖: {', '.join(all_missing)}")
        
        for package in all_missing:
            try:
                print(f"正在安装 {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package,
                    "--index-url", "https://mirrors.tencent.com/pypi/simple/"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ {package} 安装成功")
                else:
                    print(f"❌ {package} 安装失败")
                    if package in missing_required:
                        print(f"🚨 {package} 是必需依赖，程序可能无法正常运行")
            except Exception as e:
                print(f"❌ {package} 安装异常: {e}")
    
    print("🎉 所有依赖检查完成!")

def check_data_files():
    """检查数据文件"""
    print("\n📁 检查数据文件...")
    
    from glob import glob
    
    # 训练文件
    train_patterns = ['train.csv', '训练*.csv', '*train*.csv']
    train_files = []
    for pattern in train_patterns:
        train_files.extend(glob(pattern))
    
    # 测试文件
    test_patterns = ['test.csv', '测试*.csv', '*test*.csv']
    test_files = []
    for pattern in test_patterns:
        test_files.extend(glob(pattern))
    
    if train_files:
        print("✅ 找到训练数据文件")
        for f in train_files[:3]:  # 只显示前3个
            size = os.path.getsize(f)
            print(f"  - {f} ({size:,} 字节)")
        if len(train_files) > 3:
            print(f"  ... 还有 {len(train_files) - 3} 个文件")
    else:
        print("⚠️ 未找到训练数据文件")
    
    if test_files:
        print("✅ 找到测试数据文件")
        for f in test_files[:3]:  # 只显示前3个
            size = os.path.getsize(f)
            print(f"  - {f} ({size:,} 字节)")
        if len(test_files) > 3:
            print(f"  ... 还有 {len(test_files) - 3} 个文件")
    else:
        print("⚠️ 未找到测试数据文件")
    
    return len(train_files) > 0, len(test_files) > 0

def check_gui_support():
    """检测图形界面支持"""
    print("\n🖥️ 检测图形界面支持...")
    
    # 检查操作系统
    system = platform.system().lower()

    if system not in {"windows", "darwin", "linux"}:
        print(f"❓ 未知操作系统: {system}")
        return False
    
    if system == "linux":
        # Linux环境需要进一步检查
        print("🐧 Linux环境，检查X11支持...")
        
        # 检查DISPLAY环境变量
        display = os.environ.get('DISPLAY')
        if not display:
            print("❌ 未设置DISPLAY环境变量")
            return False
        
        print(f"📺 DISPLAY环境变量: {display}")

    # 统一进行 tkinter 可用性测试（macOS / Windows / Linux 都可能缺少 _tkinter）
    try:
        import tkinter as tk
    except Exception as e:
        print(f"❌ 无法导入tkinter模块: {e}")
        return False

    try:
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口
        root.update_idletasks()
        root.destroy()   # 销毁窗口
        print("✅ Tkinter测试成功，支持图形界面")
        return True
    except Exception as e:
        print(f"❌ Tkinter测试失败: {e}")
        return False


def launch_gui():
    """启动图形界面"""
    print("\n🎮 启动图形界面...")
    try:
        # 检查GUI工具文件是否存在
        gui_file = "SnowNLP训练测试工具.py"
        if not os.path.exists(gui_file):
            print(f"❌ 找不到GUI文件: {gui_file}")
            return False
        
        # 使用子进程启动GUI，避免import阻塞
        proc = subprocess.Popen([sys.executable, gui_file])

        # 处理“立即崩溃”场景（例如缺少 _tkinter）
        time.sleep(0.3)
        rc = proc.poll()
        if rc is not None and rc != 0:
            print(f"❌ 图形界面启动失败 (exit code: {rc})")
            return False

        print("✅ 图形界面已启动")
        return True
        
    except Exception as e:
        print(f"❌ 图形界面启动失败: {e}")
        return False

def launch_cli():
    """启动命令行界面"""
    print("\n💻 启动命令行界面...")
    try:
        # 检查命令行工具文件是否存在
        cli_file = "命令行训练工具.py"
        if not os.path.exists(cli_file):
            print(f"❌ 找不到命令行文件: {cli_file}")
            return False
        
        # 使用子进程启动CLI
        proc = subprocess.Popen([sys.executable, cli_file])

        time.sleep(0.3)
        rc = proc.poll()
        if rc is not None and rc != 0:
            print(f"❌ 命令行界面启动失败 (exit code: {rc})")
            return False

        print("✅ 命令行界面已启动")
        return True
        
    except Exception as e:
        print(f"❌ 命令行界面启动失败: {e}")
        return False

def show_usage_tips():
    """显示使用提示"""
    print("\n💡 使用提示:")
    print("📊 如果您有训练数据文件，可以直接开始训练")
    print("🧪 没有数据文件？工具会自动创建示例数据")
    print("⚡ 支持多种数据格式和中性数据处理策略")
    print("🔄 训练完成后会自动替换系统模型")
    
    print("\n📚 功能说明:")
    print("• 训练模型: 使用您的数据训练专属情感分析模型")
    print("• 测试验证: 多种测试方式验证模型效果") 
    print("• 模型管理: 管理多个训练模型，支持对比和导出")
    print("• 交互测试: 实时输入文本进行情感分析")

def show_manual_options():
    """显示手动选择选项"""
    print("\n🎯 请选择运行模式:")
    print("1. 💻 命令行模式 (适合无图形界面环境)")
    print("2. 🖥️ 图形界面模式 (需要图形界面支持)")
    print("3. 📊 直接运行快速测试")
    print("4. 🔧 环境诊断")
    print("0. 🚪 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (0-4): ").strip()
            
            if choice == '0':
                print("👋 再见!")
                return False
            elif choice == '1':
                return launch_cli()
            elif choice == '2':
                return launch_gui()
            elif choice == '3':
                run_quick_test()
                return True
            elif choice == '4':
                run_environment_diagnosis()
                continue
            else:
                print("❌ 无效选择，请重新输入")
        except KeyboardInterrupt:
            print("\n👋 再见!")
            return False

def run_quick_test():
    """运行快速测试"""
    print("\n⚡ 快速测试模式")
    try:
        from snownlp import SnowNLP
        
        test_cases = [
            "这个产品质量非常好，强烈推荐！",
            "服务态度太差了，很不满意",
            "还可以吧，一般般",
            "物流速度很快，包装也不错",
            "价格有点贵，但质量确实好"
        ]
        
        print("🧪 测试用例:")
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
                
                print(f"[{i}] {score:.4f} ({sentiment}) | {text}")
            except Exception as e:
                print(f"[{i}] 测试失败: {e}")
        
        print("\n✅ 快速测试完成")
        
    except ImportError:
        print("❌ SnowNLP模块未安装，无法进行测试")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def run_environment_diagnosis():
    """运行环境诊断"""
    print("\n🔧 环境诊断")
    print("-" * 50)
    
    # Python信息
    print(f"🐍 Python版本: {sys.version}")
    print(f"📂 Python可执行文件: {sys.executable}")
    print(f"📋 Python路径: {sys.path[:3]}...")
    
    # 系统信息
    print(f"💻 操作系统: {platform.system()} {platform.release()}")
    print(f"🏗️ 系统架构: {platform.machine()}")
    print(f"🏠 当前目录: {os.getcwd()}")
    
    # 环境变量
    print(f"📺 DISPLAY: {os.environ.get('DISPLAY', '未设置')}")
    print(f"🏠 HOME: {os.environ.get('HOME', '未设置')}")
    
    # 图形界面测试
    gui_support = check_gui_support()
    
    # 依赖检查
    check_dependencies()
    
    # 文件检查
    has_train, has_test = check_data_files()
    
    print("\n📋 诊断总结:")
    print(f"• 图形界面支持: {'✅ 是' if gui_support else '❌ 否'}")
    print(f"• 训练数据文件: {'✅ 有' if has_train else '❌ 无'}")
    print(f"• 测试数据文件: {'✅ 有' if has_test else '❌ 无'}")
    
    if gui_support:
        print("💡 建议: 可以使用图形界面模式")
    else:
        print("💡 建议: 使用命令行模式")

def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    check_dependencies()
    
    # 检查数据文件
    has_train, has_test = check_data_files()
    
    # 检测图形界面支持
    gui_support = check_gui_support()
    
    show_usage_tips()
    
    # 自动选择运行模式
    if gui_support:
        print("\n🎮 检测到图形界面支持，尝试启动图形界面...")
        if launch_gui():
            return 0
        else:
            print("⚠️ 图形界面启动失败，切换到命令行模式")
            if launch_cli():
                return 0
    else:
        print("\n💻 未检测到图形界面支持，启动命令行模式...")
        if launch_cli():
            return 0
    
    # 如果自动启动失败，显示手动选择
    print("\n⚠️ 自动启动失败，请手动选择运行模式")
    if show_manual_options():
        return 0
    
    return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 