# -*- coding: utf-8 -*-
"""
SnowNLP情感分析训练测试工具 - 可视化界面版
集成训练、测试、评估的完整GUI工具
"""

# 在任何导入之前设置 matplotlib 后端，避免 macOS 版本兼容性问题
import os
os.environ['MPLBACKEND'] = 'Agg'  # 使用非 GUI 后端

import sys

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
except Exception as e:
    print("❌ 无法启动GUI：当前 Python 环境缺少 Tk 支持（_tkinter）")
    print(f"详细错误: {e}")
    print("\n✅ 解决方案（macOS + Homebrew Python 3.12 常见）:")
    print("1) 安装 Tk 支持: brew install python-tk@3.12")
    print("2) 或者改用命令行/网页界面: python 启动工具.py")
    sys.exit(1)
import pandas as pd
import time
import shutil
import threading
from snownlp import SnowNLP, sentiment
from snownlp.sentiment import Sentiment
from glob import glob
from tqdm import tqdm
import random
import marshal
import pickle
import numpy as np
import json
from datetime import datetime

# matplotlib 延迟导入
plt = None
FigureCanvasTkAgg = None
def _import_matplotlib():
    """延迟导入 matplotlib，在需要时才加载"""
    global plt, FigureCanvasTkAgg
    if plt is None:
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # 切换到 TkAgg 后端用于嵌入 Tkinter
            import matplotlib.pyplot as _plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FigureCanvasTkAgg
            plt = _plt
            FigureCanvasTkAgg = _FigureCanvasTkAgg
        except Exception as e:
            print(f"警告: matplotlib 加载失败 ({e})，图表功能将不可用")
    return plt, FigureCanvasTkAgg

class ModelManager:
    """模型管理器"""
    def __init__(self, config_file="model_history.json"):
        self.config_file = config_file
        self.models = self.load_models()
    
    def load_models(self):
        """加载模型历史"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_models(self):
        """保存模型历史"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.models, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存模型历史失败: {e}")
    
    def add_model(self, model_path, info):
        """添加模型记录"""
        model_id = info.get('id', str(int(time.time())))
        self.models[model_id] = {
            'path': model_path,
            'name': info.get('name', f"模型_{datetime.now().strftime('%m%d_%H%M')}"),
            'created_time': info.get('created_time', datetime.now().isoformat()),
            'train_files': info.get('train_files', []),
            'test_files': info.get('test_files', []),
            'train_samples': info.get('train_samples', 0),
            'test_accuracy': info.get('test_accuracy', 0),
            'neutral_strategy': info.get('neutral_strategy', ''),
            'notes': info.get('notes', ''),
            'file_size': os.path.getsize(model_path) if os.path.exists(model_path) else 0
        }
        self.save_models()
        return model_id
    
    def get_model_list(self):
        """获取模型列表"""
        valid_models = {}
        for model_id, info in self.models.items():
            if os.path.exists(info['path']):
                valid_models[model_id] = info
        
        # 如果有无效模型，更新配置
        if len(valid_models) != len(self.models):
            self.models = valid_models
            self.save_models()
        
        return valid_models
    
    def update_model(self, model_id, updates):
        """更新模型信息"""
        if model_id in self.models:
            self.models[model_id].update(updates)
            self.save_models()
    
    def delete_model(self, model_id):
        """删除模型记录"""
        if model_id in self.models:
            model_path = self.models[model_id]['path']
            del self.models[model_id]
            self.save_models()
            return model_path
        return None

class SnowNLPTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SnowNLP情感分析训练测试工具 v3.0")
        self.root.geometry("1400x900")  # 增大窗口以容纳更多功能
        
        # 变量
        self.train_files = []
        self.test_files = []
        self.neutral_strategy = tk.StringVar(value="balance")
        self.training_running = False
        
        # 模型管理器
        self.model_manager = ModelManager()
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # 0. 状态栏和操作指导
        self.create_status_guide(main_frame)
        
        # 1. 文件选择区域
        self.create_file_selection(main_frame)
        
        # 2. 训练配置区域
        self.create_training_config(main_frame)
        
        # 3. 操作按钮区域
        self.create_action_buttons(main_frame)
        
        # 4. 日志和结果显示区域
        self.create_log_and_results(main_frame)
        
        # 5. 测试区域
        self.create_test_section(main_frame)
        
        # 初始化状态
        self.update_status_guide("ready")
    
    def create_status_guide(self, parent):
        """创建状态栏和操作指导"""
        guide_frame = ttk.LabelFrame(parent, text="💡 操作指导", padding="10")
        guide_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        guide_frame.columnconfigure(0, weight=1)
        
        # 当前状态显示
        self.status_var = tk.StringVar(value="准备就绪 - 请选择数据文件开始")
        self.status_label = ttk.Label(guide_frame, textvariable=self.status_var, 
                                     font=("", 10, "bold"), foreground="blue")
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # 操作提示
        self.guide_var = tk.StringVar(value="步骤1: 选择训练数据和测试数据 → 步骤2: 配置训练参数 → 步骤3: 开始训练")
        self.guide_label = ttk.Label(guide_frame, textvariable=self.guide_var, 
                                    font=("", 9), foreground="gray")
        self.guide_label.grid(row=1, column=0, sticky=tk.W)
    
    def update_status_guide(self, status, message=None):
        """更新状态指导"""
        status_messages = {
            "ready": "准备就绪 - 请选择数据文件开始",
            "files_selected": "训练和测试文件已就绪 - 可以开始训练",
            "training": "正在训练模型 - 请耐心等待",
            "training_complete": "训练完成 - 可以进行测试验证",
            "testing": "正在测试模型 - 分析性能表现",
            "model_testing": "正在测试选择的模型文件",
            "data_testing": "正在使用选择的数据集测试",
            "comparing": "正在对比多个模型性能"
        }
        
        guide_messages = {
            "ready": "步骤1: 选择训练数据和测试数据 → 步骤2: 选择中性数据处理策略 → 步骤3: 点击'开始训练'",
            "files_selected": "步骤2: 选择中性数据处理策略 → 步骤3: 点击'开始训练'",
            "training": "训练进行中: 数据加载 → 模型训练 → 性能测试 → 模型替换",
            "training_complete": "可选操作: 快速验证 | 完整测试 | 数据集评估 | 交互测试",
            "testing": "测试进行中: 加载数据 → 模型预测 → 计算准确率 → 生成报告",
            "model_testing": "正在使用选中的模型进行标准测试",
            "data_testing": "正在使用选中的数据集评估当前模型",
            "comparing": "正在对比多个模型，将显示性能排名"
        }
        
        if message:
            self.status_var.set(message)
        else:
            self.status_var.set(status_messages.get(status, "操作进行中..."))
        
        self.guide_var.set(guide_messages.get(status, "请查看日志了解详细进度"))
        self.root.update()
    
    def create_file_selection(self, parent):
        """创建文件选择区域"""
        file_frame = ttk.LabelFrame(parent, text="📁 数据文件选择", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # 训练文件选择
        ttk.Label(file_frame, text="训练数据:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.train_files_var = tk.StringVar(value="未选择文件")
        ttk.Label(file_frame, textvariable=self.train_files_var, background="white", relief="sunken").grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(file_frame, text="选择训练文件", command=self.select_train_files).grid(
            row=0, column=2, padx=(0, 10))
        ttk.Button(file_frame, text="自动查找", command=self.auto_find_train_files).grid(
            row=0, column=3)
        
        # 测试文件选择
        ttk.Label(file_frame, text="测试数据:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.test_files_var = tk.StringVar(value="未选择文件")
        ttk.Label(file_frame, textvariable=self.test_files_var, background="white", relief="sunken").grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        ttk.Button(file_frame, text="选择测试文件", command=self.select_test_files).grid(
            row=1, column=2, padx=(0, 10), pady=(10, 0))
        ttk.Button(file_frame, text="自动查找", command=self.auto_find_test_files).grid(
            row=1, column=3, pady=(10, 0))
    
    def create_training_config(self, parent):
        """创建训练配置区域"""
        config_frame = ttk.LabelFrame(parent, text="⚙️ 训练配置", padding="10")
        config_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 中性数据处理策略
        ttk.Label(config_frame, text="中性数据处理:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        strategies = [
            ("balance", "自动平衡(推荐)"),
            ("random", "随机分配"),
            ("positive", "全部正面"),
            ("negative", "全部负面"),
            ("split", "比例分配"),
            ("exclude", "排除中性")
        ]
        
        strategy_frame = ttk.Frame(config_frame)
        strategy_frame.grid(row=0, column=1, sticky=tk.W)
        
        for i, (value, text) in enumerate(strategies):
            ttk.Radiobutton(strategy_frame, text=text, variable=self.neutral_strategy, 
                           value=value).grid(row=0, column=i, padx=(0, 15))
    
    def create_action_buttons(self, parent):
        """创建操作按钮区域"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10))
        
        # 训练相关按钮
        train_frame = ttk.LabelFrame(button_frame, text="🚀 模型训练 (基础功能)")
        train_frame.grid(row=0, column=0, padx=(0, 10))
        
        self.train_btn = ttk.Button(train_frame, text="🔥 开始训练\n(训练新模型)", 
                                   command=self.start_training_with_confirm, 
                                   style="Accent.TButton")
        self.train_btn.grid(row=0, column=0, padx=10, pady=10)
        
        self.stop_btn = ttk.Button(train_frame, text="⏹️ 停止训练\n(中断当前训练)", 
                                  command=self.stop_training, 
                                  state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # 测试相关按钮
        test_frame = ttk.LabelFrame(button_frame, text="🧪 模型测试 (验证效果)")
        test_frame.grid(row=0, column=1, padx=(0, 10))
        
        # 第一行：基础测试
        ttk.Button(test_frame, text="⚡ 快速验证\n(内置测试用例)", command=self.quick_test_with_info).grid(
            row=0, column=0, padx=5, pady=5)
        ttk.Button(test_frame, text="🔬 完整测试\n(详细性能分析)", command=self.full_test_with_info).grid(
            row=0, column=1, padx=5, pady=5)
        ttk.Button(test_frame, text="📈 数据集评估\n(使用测试数据)", command=self.dataset_evaluation_with_info).grid(
            row=0, column=2, padx=5, pady=5)
        
        # 第二行：高级测试
        ttk.Button(test_frame, text="📁 选择模型测试\n(测试指定模型)", command=self.select_model_test_with_info).grid(
            row=1, column=0, padx=5, pady=5)
        ttk.Button(test_frame, text="📊 选择数据测试\n(使用指定数据)", command=self.select_data_test_with_info).grid(
            row=1, column=1, padx=5, pady=5)
        ttk.Button(test_frame, text="🏆 模型对比\n(多模型PK)", command=self.model_comparison_with_info).grid(
            row=1, column=2, padx=5, pady=5)
        
        # 工具按钮
        tool_frame = ttk.LabelFrame(button_frame, text="🔧 实用工具")
        tool_frame.grid(row=0, column=2)
        
        ttk.Button(tool_frame, text="ℹ️ 模型信息\n(查看当前模型)", command=self.show_model_info).grid(
            row=0, column=0, padx=5, pady=5)
        ttk.Button(tool_frame, text="🔄 手动替换\n(安装训练模型)", command=self.manual_replace_with_info).grid(
            row=0, column=1, padx=5, pady=5)
        ttk.Button(tool_frame, text="🧹 清空日志\n(清理界面)", command=self.clear_log).grid(
            row=0, column=2, padx=5, pady=5)
        
        # 模型管理按钮
        model_frame = ttk.LabelFrame(button_frame, text="📦 模型管理")
        model_frame.grid(row=0, column=3, padx=(10, 0))
        
        ttk.Button(model_frame, text="📋 模型列表\n(管理训练模型)", command=self.show_model_manager).grid(
            row=0, column=0, padx=5, pady=5)
        ttk.Button(model_frame, text="📊 性能对比\n(对比模型效果)", command=self.compare_models_on_dataset).grid(
            row=0, column=1, padx=5, pady=5)
        ttk.Button(model_frame, text="📤 导出模型\n(备份最佳模型)", command=self.export_model).grid(
            row=0, column=2, padx=5, pady=5)
    
    def start_training_with_confirm(self):
        """带确认的开始训练"""
        if not self.train_files:
            messagebox.showerror("错误", "❌ 请先选择训练数据文件\n\n操作步骤:\n1. 点击'选择训练文件'或'自动查找'\n2. 确保选择了包含训练数据的CSV文件")
            return
        
        # 显示训练确认对话框
        confirm_msg = f"""🚀 即将开始模型训练

📁 训练文件: {len(self.train_files)} 个
📊 测试文件: {len(self.test_files)} 个  
⚙️ 中性数据策略: {self.neutral_strategy.get()}

训练过程包括:
1. 数据加载和预处理
2. 创建训练语料文件  
3. SnowNLP模型训练
4. 模型性能测试
5. 自动模型替换

预计耗时: 1-5分钟
确定开始训练吗?"""
        
        if messagebox.askyesno("确认训练", confirm_msg):
            self.update_status_guide("training")
            self.start_training()
    
    def quick_test_with_info(self):
        """带说明的快速验证"""
        info_msg = """⚡ 快速验证说明

🎯 功能: 使用内置测试用例验证当前模型
📝 测试内容: 预定义的正面/负面/中性文本
⏱️ 耗时: 约5-10秒
📊 结果: 显示准确率和详细分析

适用场景:
• 快速检查模型是否正常工作
• 训练后的初步验证
• 对比训练前后效果

开始测试吗?"""
        
        if messagebox.askyesno("快速验证", info_msg):
            self.update_status_guide("testing")
            self.quick_test()
    
    def full_test_with_info(self):
        """带说明的完整测试"""
        info_msg = """🔬 完整测试说明

🎯 功能: 综合性模型性能评估
📝 测试内容: 基础测试 + 数据集测试
⏱️ 耗时: 根据数据量而定
📊 结果: 详细性能报告和建议

测试流程:
1. 运行扩展的基础测试用例
2. 使用测试数据集验证(如有)
3. 生成综合性能报告
4. 提供优化建议

开始完整测试吗?"""
        
        if messagebox.askyesno("完整测试", info_msg):
            self.update_status_guide("testing")
            self.full_test()
    
    def dataset_evaluation_with_info(self):
        """带说明的数据集评估"""
        if not self.test_files:
            messagebox.showwarning("提示", "⚠️ 未选择测试数据文件\n\n请先选择测试数据:")
            return
        
        info_msg = f"""📈 数据集评估说明

🎯 功能: 使用测试数据集评估模型性能
📁 数据: {len(self.test_files)} 个测试文件
⏱️ 耗时: 根据数据量而定
📊 结果: 准确率、分类报告

评估内容:
• 总体分类准确率
• 正面/负面样本准确率  
• 数据处理成功率
• 性能评级和建议

开始评估吗?"""
        
        if messagebox.askyesno("数据集评估", info_msg):
            self.update_status_guide("testing")
            self.dataset_evaluation()
    
    def select_model_test_with_info(self):
        """带说明的选择模型测试"""
        info_msg = """🔥🔥🔥 选择模型测试 - 重要说明 🔥🔥🔥

⚠️⚠️⚠️ 重要提醒 ⚠️⚠️⚠️
此功能会临时替换系统模型进行测试！

🎯 功能: 测试指定的模型文件
📝 支持格式: .marshal, .marshal.3, .model文件
⏱️ 测试耗时: 约30秒-2分钟
📊 测试结果: 该模型的详细性能表现

🔄 操作流程:
1. 📁 选择要测试的模型文件
2. 💾 自动备份当前系统模型
3. 🔄 临时替换为测试模型
4. 🧪 运行标准测试用例
5. 📈 显示测试结果
6. 🔙 自动恢复原系统模型

✅ 安全保证:
• 不会永久修改系统模型
• 测试完成后自动恢复
• 创建完整的备份文件

💡 使用场景:
• 验证训练后的模型效果
• 对比不同模型的性能
• 选择最佳训练结果

🚨 注意事项:
• 测试期间请勿关闭程序
• 确保模型文件格式正确
• 测试完成前不要手动操作

📢 确定要开始选择模型进行测试吗？"""
        
        # 使用更强烈的对话框
        result = messagebox.askokcancel(
            "🔥 重要操作 - 选择模型测试 🔥", 
            info_msg,
            icon='warning'
        )
        
        if result:
            # 再次确认
            confirm_msg = """🔔 最终确认

您即将进行模型测试操作！

⚠️ 系统将临时替换当前模型
⚠️ 请确保选择正确的模型文件
⚠️ 测试期间请勿关闭程序

是否确定继续？"""
            
            final_confirm = messagebox.askyesno(
                "⚠️ 最终确认 ⚠️", 
                confirm_msg,
                icon='question'
            )
            
            if final_confirm:
                self.update_status_guide("model_testing")
                # 添加明显的开始提示
                self.log_message("🔥🔥🔥 开始模型测试操作 🔥🔥🔥")
                self.log_message("⚠️ 系统将临时替换模型进行测试")
                self.log_message("⚠️ 请勿在测试期间关闭程序")
                self.log_message("=" * 50)
                self.select_model_test()
            else:
                self.log_message("❌ 用户取消了模型测试操作")
        else:
            self.log_message("❌ 用户取消了模型测试操作")
    
    def select_data_test_with_info(self):
        """带说明的选择数据测试"""
        info_msg = """📊 选择数据测试说明

🎯 功能: 使用指定数据集测试当前模型
📝 支持格式: CSV文件(包含content和sentiment列)
⏱️ 耗时: 根据数据量而定
📊 结果: 在该数据集上的详细性能

测试特点:
• 支持全数据集测试
• 大数据集提供采样选项
• 显示详细分类统计
• 实时进度和时间预估

开始选择数据吗?"""
        
        if messagebox.askyesno("选择数据测试", info_msg):
            self.update_status_guide("data_testing")
            self.select_data_test()
    
    def model_comparison_with_info(self):
        """带说明的模型对比"""
        info_msg = """🔥🔥🔥 模型对比测试 - 重要说明 🔥🔥🔥

⚠️⚠️⚠️ 重要提醒 ⚠️⚠️⚠️
此功能会多次临时替换系统模型进行对比测试！

🎯 功能: 同时测试多个模型并自动排名
📝 要求: 至少选择2个模型文件
⏱️ 测试耗时: 每个模型约1-2分钟
📊 测试结果: 性能排名和最佳推荐

🔄 对比流程:
1. 📁 选择多个模型文件
2. 💾 备份当前系统模型
3. 🔄 逐个替换并测试每个模型
4. 📊 收集每个模型的性能数据
5. 🏆 按准确率自动排名
6. 💡 推荐最佳模型
7. 🔙 恢复原系统模型

🔍 对比内容:
• 统一测试用例保证公平性
• 详细的准确率和性能指标
• 自动识别最佳模型
• 完整的对比报告

✅ 安全保证:
• 所有替换都是临时的
• 对比完成后自动恢复
• 创建完整的备份文件

💡 适用场景:
• 选择最佳训练结果
• 对比不同参数的模型
• 模型优化决策支持

🚨 注意事项:
• 测试时间较长，请耐心等待
• 测试期间请勿关闭程序
• 确保所有模型文件格式正确
• 测试完成前不要手动操作

📢 确定要开始模型对比测试吗？"""
        
        # 使用更强烈的对话框
        result = messagebox.askokcancel(
            "🔥 重要操作 - 模型对比测试 🔥", 
            info_msg,
            icon='warning'
        )
        
        if result:
            # 再次确认
            confirm_msg = """🔔 最终确认

您即将进行模型对比操作！

⚠️ 系统将多次临时替换模型
⚠️ 测试时间可能较长
⚠️ 请确保选择正确的模型文件
⚠️ 测试期间请勿关闭程序

是否确定继续？"""
            
            final_confirm = messagebox.askyesno(
                "⚠️ 最终确认 ⚠️", 
                confirm_msg,
                icon='question'
            )
            
            if final_confirm:
                self.update_status_guide("comparing")
                # 添加明显的开始提示
                self.log_message("🔥🔥🔥 开始模型对比测试操作 🔥🔥🔥")
                self.log_message("⚠️ 系统将多次临时替换模型进行对比")
                self.log_message("⚠️ 测试时间较长，请勿关闭程序")
                self.log_message("=" * 50)
                self.model_comparison()
            else:
                self.log_message("❌ 用户取消了模型对比操作")
        else:
            self.log_message("❌ 用户取消了模型对比操作")
    
    def manual_replace_with_info(self):
        """带说明的手动替换"""
        info_msg = """🔥🔥🔥 手动替换模型 - 重要警告 🔥🔥🔥

⚠️⚠️⚠️ 危险操作警告 ⚠️⚠️⚠️
此功能会永久修改系统模型！！！

🎯 功能: 手动安装训练好的模型
📝 作用: 将训练模型设为系统默认模型
⚠️ 重要: 这是永久性的模型替换操作

🔄 操作流程:
1. 📁 选择要安装的模型文件
2. 🔍 查找并定位系统模型位置
3. 💾 备份当前系统模型文件
4. 🔄 复制新模型到系统位置
5. ✅ 验证替换是否成功
6. 📊 测试新模型基本功能

💡 使用时机:
• 训练完成后永久安装新模型
• 自动替换失败时的备选方案
• 手动安装外部优秀模型
• 恢复之前备份的模型

✅ 安全措施:
• 自动创建原模型备份
• 验证新模型有效性
• 提供模型恢复功能

🚨🚨🚨 严重警告 🚨🚨🚨
• 此操作会永久更改系统模型
• 如无备份，原模型将丢失
• 错误的模型文件可能导致功能异常
• 操作不可自动撤销

⛔ 风险提示:
• 替换后将影响所有使用SnowNLP的程序
• 不正确的模型可能导致预测结果异常
• 系统重装SnowNLP才能恢复默认模型

📢 您确定要进行这个危险的永久替换操作吗？"""
        
        # 使用最强烈的警告对话框
        result = messagebox.askokcancel(
            "🚨 危险操作 - 永久替换系统模型 🚨", 
            info_msg,
            icon='error'
        )
        
        if result:
            # 第一次确认
            first_confirm_msg = """🔔 第一次确认

您确定要进行永久性的模型替换吗？

⚠️ 这将永久更改系统SnowNLP模型
⚠️ 影响所有使用SnowNLP的程序
⚠️ 操作后无法自动撤销

是否继续？"""
            
            first_confirm = messagebox.askyesno(
                "⚠️ 第一次确认 ⚠️", 
                first_confirm_msg,
                icon='warning'
            )
            
            if first_confirm:
                # 最终确认
                final_confirm_msg = """🔔 最终确认

⚠️⚠️⚠️ 最后一次确认 ⚠️⚠️⚠️

您即将进行永久性的系统模型替换！

✋ 请再次确认您了解以下风险:
• 原系统模型将被永久替换
• 所有使用SnowNLP的程序都会受影响
• 只有备份文件可以恢复原模型
• 错误的模型可能导致功能异常

💡 建议: 如果不确定，请先使用"选择模型测试"功能进行验证

🤔 您真的确定要继续这个永久替换操作吗？"""
                
                final_confirm = messagebox.askyesno(
                    "🚨 最终确认 - 永久替换 🚨", 
                    final_confirm_msg,
                    icon='error'
                )
                
                if final_confirm:
                    # 添加明显的开始提示
                    self.log_message("🚨🚨🚨 开始永久性模型替换操作 🚨🚨🚨")
                    self.log_message("⚠️ 这是危险的永久性操作")
                    self.log_message("⚠️ 将永久更改系统SnowNLP模型")
                    self.log_message("=" * 50)
                    self.manual_replace_model()
                else:
                    self.log_message("✅ 用户明智地取消了永久替换操作")
            else:
                self.log_message("✅ 用户取消了永久替换操作")
        else:
            self.log_message("✅ 用户取消了永久替换操作")
    
    def create_log_and_results(self, parent):
        """创建日志和结果显示区域"""
        # 创建Notebook用于分页显示
        notebook = ttk.Notebook(parent)
        notebook.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 日志页面
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📝 运行日志")
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 训练进度页面 (新增强版)
        progress_frame = ttk.Frame(notebook)
        notebook.add(progress_frame, text="🚀 训练进度")
        
        # 创建进度显示区域
        self.create_enhanced_progress_display(progress_frame)
        
        # 结果页面
        result_frame = ttk.Frame(notebook)
        notebook.add(result_frame, text="📈 测试结果")
        
        # 创建结果显示区域
        self.result_text = scrolledtext.ScrolledText(result_frame, height=15, width=80)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_enhanced_progress_display(self, parent):
        """创建增强的进度显示界面"""
        # 主框架
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 训练状态标题
        self.training_status_label = ttk.Label(main_frame, text="准备开始训练...", 
                                              font=("", 14, "bold"), foreground="blue")
        self.training_status_label.pack(pady=(0, 10))
        
        # 时间信息框架
        time_frame = ttk.LabelFrame(main_frame, text="⏱️ 时间信息", padding="10")
        time_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 时间信息网格
        time_info_frame = ttk.Frame(time_frame)
        time_info_frame.pack(fill=tk.X)
        
        # 开始时间
        ttk.Label(time_info_frame, text="开始时间:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.start_time_var = tk.StringVar(value="未开始")
        ttk.Label(time_info_frame, textvariable=self.start_time_var).grid(row=0, column=1, sticky=tk.W)
        
        # 已用时间
        ttk.Label(time_info_frame, text="已用时间:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.elapsed_time_var = tk.StringVar(value="00:00:00")
        ttk.Label(time_info_frame, textvariable=self.elapsed_time_var, font=("", 10, "bold")).grid(row=0, column=3, sticky=tk.W)
        
        # 预估剩余时间
        ttk.Label(time_info_frame, text="预估剩余:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.remaining_time_var = tk.StringVar(value="计算中...")
        ttk.Label(time_info_frame, textvariable=self.remaining_time_var, font=("", 10, "bold")).grid(row=1, column=1, sticky=tk.W)
        
        # 预估完成时间
        ttk.Label(time_info_frame, text="预估完成:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10))
        self.finish_time_var = tk.StringVar(value="计算中...")
        ttk.Label(time_info_frame, textvariable=self.finish_time_var).grid(row=1, column=3, sticky=tk.W)
        
        # 数据信息框架
        data_frame = ttk.LabelFrame(main_frame, text="📊 数据信息", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        data_info_frame = ttk.Frame(data_frame)
        data_info_frame.pack(fill=tk.X)
        
        # 数据统计
        self.data_stats_var = tk.StringVar(value="等待加载数据...")
        ttk.Label(data_info_frame, textvariable=self.data_stats_var, font=("", 9)).pack(anchor=tk.W)
        
        # 训练步骤框架
        steps_frame = ttk.LabelFrame(main_frame, text="📋 训练步骤", padding="10")
        steps_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 步骤列表
        self.training_steps = [
            ("📂", "数据加载", "加载和解析训练数据文件"),
            ("📝", "语料准备", "创建正面和负面语料文件"),
            ("📊", "基线测试", "记录训练前模型性能"),
            ("🧠", "模型训练", "SnowNLP核心算法训练"),
            ("🔄", "模型部署", "替换系统模型文件"),
            ("✅", "完成验证", "验证新模型性能")
        ]
        
        self.step_frames = []
        self.step_progress_bars = []
        self.step_labels = []
        
        for i, (icon, name, desc) in enumerate(self.training_steps):
            step_frame = ttk.Frame(steps_frame)
            step_frame.pack(fill=tk.X, pady=2)
            
            # 步骤图标和状态
            status_label = ttk.Label(step_frame, text="⏳", font=("", 12))
            status_label.pack(side=tk.LEFT, padx=(0, 5))
            
            # 步骤名称
            name_label = ttk.Label(step_frame, text=f"{icon} {name}", font=("", 10, "bold"))
            name_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # 步骤进度条
            step_progress = ttk.Progressbar(step_frame, length=200, mode='determinate')
            step_progress.pack(side=tk.LEFT, padx=(0, 10))
            
            # 步骤描述
            desc_label = ttk.Label(step_frame, text=desc, font=("", 9))
            desc_label.pack(side=tk.LEFT)
            
            self.step_frames.append(step_frame)
            self.step_progress_bars.append(step_progress)
            self.step_labels.append((status_label, name_label, desc_label))
        
        # 总体进度
        overall_frame = ttk.LabelFrame(main_frame, text="🎯 总体进度", padding="10")
        overall_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 总进度条
        ttk.Label(overall_frame, text="整体完成度:").pack(anchor=tk.W)
        self.overall_progress = ttk.Progressbar(overall_frame, mode='determinate', length=400)
        self.overall_progress.pack(fill=tk.X, pady=(5, 0))
        
        # 总进度标签
        self.overall_progress_label = tk.StringVar(value="0% - 准备开始")
        ttk.Label(overall_frame, textvariable=self.overall_progress_label, 
                 font=("", 10, "bold")).pack(pady=(5, 0))
        
        # 当前任务进度
        ttk.Label(overall_frame, text="当前任务:").pack(anchor=tk.W, pady=(10, 0))
        self.current_task = ttk.Progressbar(overall_frame, mode='indeterminate', length=400)
        self.current_task.pack(fill=tk.X, pady=(5, 0))
        
        self.current_task_label = tk.StringVar(value="等待开始...")
        ttk.Label(overall_frame, textvariable=self.current_task_label).pack(pady=(5, 0))
        
        # 性能指标框架
        metrics_frame = ttk.LabelFrame(main_frame, text="📈 性能指标", padding="10")
        metrics_frame.pack(fill=tk.X)
        
        metrics_info_frame = ttk.Frame(metrics_frame)
        metrics_info_frame.pack(fill=tk.X)
        
        # 训练前后对比
        self.performance_var = tk.StringVar(value="训练完成后将显示性能提升情况")
        ttk.Label(metrics_info_frame, textvariable=self.performance_var, font=("", 9)).pack(anchor=tk.W)
        
        # 初始化训练状态
        self.training_start_time = None
        self.current_step = 0
    
    def update_training_status(self, status, step=None):
        """更新训练状态"""
        status_messages = {
            "starting": "🚀 正在启动训练流程...",
            "data_loading": "📂 正在加载训练数据...",
            "data_processing": "📝 正在处理和清理数据...",
            "corpus_creation": "📝 正在创建训练语料...",
            "baseline_testing": "📊 正在进行基线性能测试...",
            "model_training": "🧠 正在训练SnowNLP模型...",
            "model_deploying": "🔄 正在部署新模型...",
            "final_testing": "✅ 正在验证训练结果...",
            "completed": "🎉 训练成功完成！",
            "failed": "❌ 训练失败"
        }
        
        try:
            self.training_status_label.config(text=status_messages.get(status, status))
            
            if step is not None:
                self.current_step = step
                self.log(f"🔧 切换到步骤 {step+1}: {self.training_steps[step][1] if step < len(self.training_steps) else '未知'}")
            
            # 强制刷新界面
            self.root.update_idletasks()
            self.root.update()
            
        except Exception as e:
            self.log(f"❌ 更新训练状态失败: {e}")
    
    def update_step_status(self, step, progress=0, completed=False, failed=False):
        """更新步骤状态"""
        if 0 <= step < len(self.step_labels):
            status_label, name_label, desc_label = self.step_labels[step]
            
            try:
                if failed:
                    status_label.config(text="❌", foreground="red")
                    self.step_progress_bars[step]['value'] = 0
                    self.log(f"🔧 步骤 {step+1} 状态更新: 失败")
                elif completed:
                    status_label.config(text="✅", foreground="green")
                    self.step_progress_bars[step]['value'] = 100
                    self.log(f"🔧 步骤 {step+1} 状态更新: 完成")
                elif progress > 0:
                    status_label.config(text="🔄", foreground="blue")
                    self.step_progress_bars[step]['value'] = progress
                    self.log(f"🔧 步骤 {step+1} 状态更新: 进度 {progress}%")
                else:
                    status_label.config(text="⏳", foreground="orange")
                    self.step_progress_bars[step]['value'] = 0
                    self.log(f"🔧 步骤 {step+1} 状态更新: 等待中")
                
                # 强制刷新界面
                self.root.update_idletasks()
                self.root.update()
                
            except Exception as e:
                self.log(f"❌ 更新步骤状态失败: {e}")
        else:
            self.log(f"❌ 无效步骤索引: {step}, 总步骤数: {len(self.step_labels)}")
    
    def update_enhanced_progress(self, overall_progress=None, step_progress=None, current_task=None):
        """更新增强进度显示"""
        try:
            if overall_progress is not None:
                self.overall_progress['value'] = overall_progress
                self.overall_progress_label.set(f"{overall_progress:.1f}% - {current_task or '进行中...'}")
            
            if step_progress is not None and hasattr(self, 'current_step') and self.current_step < len(self.step_progress_bars):
                self.step_progress_bars[self.current_step]['value'] = step_progress
            
            if current_task is not None:
                self.current_task_label.set(current_task)
            
            # 更新时间信息
            self.update_time_info()
            
            # 强制刷新界面
            self.root.update_idletasks()
            self.root.update()
            
        except Exception as e:
            self.log(f"❌ 更新进度显示失败: {e}")
    
    def update_time_info(self):
        """更新时间信息"""
        if self.training_start_time is None:
            return
        
        import datetime
        
        # 计算已用时间
        elapsed = datetime.datetime.now() - self.training_start_time
        elapsed_str = str(elapsed).split('.')[0]  # 去掉微秒
        self.elapsed_time_var.set(elapsed_str)
        
        # 估算剩余时间
        if hasattr(self, 'overall_progress') and self.overall_progress['value'] > 5:
            progress_percent = self.overall_progress['value'] / 100
            total_estimated = elapsed.total_seconds() / progress_percent
            remaining = total_estimated - elapsed.total_seconds()
            
            if remaining > 0:
                remaining_td = datetime.timedelta(seconds=int(remaining))
                self.remaining_time_var.set(str(remaining_td).split('.')[0])
                
                # 预估完成时间
                finish_time = datetime.datetime.now() + remaining_td
                self.finish_time_var.set(finish_time.strftime("%H:%M:%S"))
            else:
                self.remaining_time_var.set("即将完成")
                self.finish_time_var.set("即将完成")
    
    def update_data_stats(self, stats_text):
        """更新数据统计信息"""
        self.data_stats_var.set(stats_text)
    
    def update_performance_metrics(self, metrics_text):
        """更新性能指标"""
        self.performance_var.set(metrics_text)
    
    def create_test_section(self, parent):
        """创建测试区域"""
        test_frame = ttk.LabelFrame(parent, text="🎮 交互式测试", padding="10")
        test_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        test_frame.columnconfigure(0, weight=1)
        
        # 输入区域
        input_frame = ttk.Frame(test_frame)
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        ttk.Label(input_frame, text="输入测试文本:").grid(row=0, column=0, sticky=tk.W)
        self.test_input = tk.Text(input_frame, height=3, width=70)
        self.test_input.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(input_frame, text="分析情感", command=self.analyze_text).grid(
            row=1, column=1, sticky=tk.N)
        
        # 结果显示
        self.test_result = tk.StringVar(value="等待输入...")
        result_label = ttk.Label(test_frame, textvariable=self.test_result, font=("", 12, "bold"))
        result_label.grid(row=1, column=0, pady=10)
    
    def log(self, message):
        """添加日志"""
        timestamp = time.strftime("[%H:%M:%S]")
        self.log_text.insert(tk.END, f"{timestamp} {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_progress(self, value=None, text=None):
        """更新进度 - 兼容性方法"""
        if value is not None:
            self.overall_progress['value'] = value
        if text is not None:
            if hasattr(self, 'current_task_label'):
                self.current_task_label.set(text)
        self.root.update()
    
    def start_time_updater(self):
        """启动时间更新器"""
        if self.training_running and self.training_start_time:
            self.update_time_info()
            # 每秒更新一次
            self.root.after(1000, self.start_time_updater)
    
    def select_train_files(self):
        """选择训练文件"""
        files = filedialog.askopenfilenames(
            title="选择训练数据文件 - 支持多选",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if files:
            self.train_files = list(files)
            self.train_files_var.set(f"✅ 已选择 {len(files)} 个文件")
            self.log(f"✅ 选择训练文件: {', '.join([os.path.basename(f) for f in files])}")
            self.update_file_status()
        else:
            self.log("❌ 未选择训练文件")
    
    def select_test_files(self):
        """选择测试文件"""
        files = filedialog.askopenfilenames(
            title="选择测试数据文件 - 支持多选",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if files:
            self.test_files = list(files)
            self.test_files_var.set(f"✅ 已选择 {len(files)} 个文件")
            self.log(f"✅ 选择测试文件: {', '.join([os.path.basename(f) for f in files])}")
            self.update_file_status()
        else:
            self.log("❌ 未选择测试文件")
    
    def auto_find_train_files(self):
        """自动查找训练文件"""
        train_patterns = ['train.csv', '训练集.csv', '*train*.csv', '*训练*.csv']
        found_files = []
        
        self.log("🔍 正在自动搜索训练文件...")
        for pattern in train_patterns:
            files = glob(pattern)
            found_files.extend([f for f in files if os.path.exists(f)])
        
        if found_files:
            self.train_files = found_files
            self.train_files_var.set(f"🔍 自动找到 {len(found_files)} 个文件")
            self.log(f"✅ 自动找到训练文件: {', '.join([os.path.basename(f) for f in found_files])}")
            self.update_file_status()
        else:
            self.train_files_var.set("❌ 未找到训练文件")
            self.log("❌ 未找到训练数据文件，请手动选择")
            messagebox.showinfo("提示", "未找到训练数据文件\n\n建议操作:\n• 点击'选择训练文件'手动选择\n• 确保文件名包含'train'或'训练'\n• 检查文件格式是否为CSV")
    
    def auto_find_test_files(self):
        """自动查找测试文件"""
        test_patterns = ['test.csv', '测试集.csv', '*test*.csv', '*测试*.csv']
        found_files = []
        
        self.log("🔍 正在自动搜索测试文件...")
        for pattern in test_patterns:
            files = glob(pattern)
            found_files.extend([f for f in files if os.path.exists(f)])
        
        if found_files:
            self.test_files = found_files
            self.test_files_var.set(f"🔍 自动找到 {len(found_files)} 个文件")
            self.log(f"✅ 自动找到测试文件: {', '.join([os.path.basename(f) for f in found_files])}")
            self.update_file_status()
        else:
            self.test_files_var.set("❌ 未找到测试文件")
            self.log("❌ 未找到测试数据文件，请手动选择")
            messagebox.showinfo("提示", "未找到测试数据文件\n\n建议操作:\n• 点击'选择测试文件'手动选择\n• 确保文件名包含'test'或'测试'\n• 检查文件格式是否为CSV")
    
    def start_training(self):
        """开始训练"""
        if not self.train_files:
            messagebox.showerror("错误", "请先选择训练数据文件")
            return
        
        if self.training_running:
            messagebox.showwarning("提示", "训练已在进行中")
            return
        
        # 禁用训练按钮，启用停止按钮
        self.train_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.training_running = True
        
        # 在新线程中执行训练
        training_thread = threading.Thread(target=self.training_worker)
        training_thread.daemon = True
        training_thread.start()
    
    def stop_training(self):
        """停止训练"""
        self.training_running = False
        self.train_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log("⏹️ 用户手动停止训练")
    
    def training_worker(self):
        """训练工作线程"""
        import datetime
        
        try:
            # 初始化训练状态
            self.training_start_time = datetime.datetime.now()
            self.start_time_var.set(self.training_start_time.strftime("%H:%M:%S"))
            self.current_step = 0  # 确保初始化当前步骤
            
            self.log("🚀 开始训练流程...")
            self.log("🔧 初始化进度显示系统...")
            
            # 启动时间更新器
            self.start_time_updater()
            
            # 初始化所有步骤状态
            for i in range(len(self.training_steps)):
                self.update_step_status(i, 0, False)
            
            self.update_training_status("starting", 0)
            self.update_enhanced_progress(0, 0, "初始化训练环境...")
            self.current_task.start()
            
            # 步骤1: 加载数据 (0-30%)
            self.log("📂 步骤1: 开始加载训练数据...")
            self.current_step = 0
            self.update_training_status("data_loading", 0)
            self.update_enhanced_progress(5, 10, "正在加载训练数据文件...")
            self.update_step_status(0, 10)
            
            train_texts, train_labels = self.load_data(self.train_files, "训练")
            
            if not train_texts:
                self.log("❌ 训练数据加载失败")
                self.update_training_status("failed")
                return
            
            # 更新数据统计
            pos_count = sum(1 for label in train_labels if label == 1)
            neg_count = sum(1 for label in train_labels if label == 0)
            data_stats = f"训练样本: {len(train_texts)} 个 (正面: {pos_count}, 负面: {neg_count})"
            self.update_data_stats(data_stats)
            
            self.update_enhanced_progress(15, 80, "数据加载完成，开始处理...")
            self.update_step_status(0, 80)
            
            # 完成步骤1
            self.update_enhanced_progress(25, 100, "数据加载完成")
            self.update_step_status(0, 100, True)  # 完成数据加载
            self.log("✅ 步骤1: 数据加载完成")
            
            # 步骤2: 创建语料文件 (30-40%)
            self.log("📝 步骤2: 开始创建语料文件...")
            self.current_step = 1
            self.update_training_status("corpus_creation", 1)
            self.update_enhanced_progress(30, 20, "正在创建训练语料文件...")
            self.update_step_status(1, 20)
            
            pos_path = 'temp_data/pos.txt'
            neg_path = 'temp_data/neg.txt'
            pos_file_count, neg_file_count = self.create_sentiment_files(train_texts, train_labels, pos_path, neg_path)
            
            if pos_file_count == 0 or neg_file_count == 0:
                self.log("❌ 正面或负面样本数量为0，无法训练")
                self.update_training_status("failed")
                return
            
            self.update_enhanced_progress(35, 100, "语料文件创建完成")
            self.update_step_status(1, 100, True)
            self.log("✅ 步骤2: 语料文件创建完成")
            
            # 步骤3: 训练前基线测试 (40-50%)
            self.log("📊 步骤3: 开始基线性能测试...")
            self.current_step = 2
            base_acc = None
            test_texts = None
            test_labels = None
            
            if self.test_files:
                self.update_training_status("baseline_testing", 2)
                self.update_enhanced_progress(40, 30, "正在进行基线性能测试...")
                self.update_step_status(2, 30)
                
                test_texts, test_labels = self.load_data(self.test_files, "测试")
                if test_texts:
                    self.update_enhanced_progress(45, 70, "正在评估当前模型性能...")
                    self.update_step_status(2, 70)
                    
                    base_acc = self.evaluate_model_simple(test_texts, test_labels)
                    self.log(f"📊 训练前基线准确率: {base_acc:.2%}")
                    
                    self.update_performance_metrics(f"基线准确率: {base_acc:.2%} | 训练中...")
                    self.update_enhanced_progress(50, 100, "基线测试完成")
                    self.update_step_status(2, 100, True)
                    self.log("✅ 步骤3: 基线测试完成")
            else:
                self.update_enhanced_progress(50, 100, "跳过基线测试（无测试数据）")
                self.update_step_status(2, 100, True)
                self.log("⏭️ 步骤3: 跳过基线测试（无测试数据）")
            
            # 步骤4: 模型训练 (50-85%)
            self.log("🧠 步骤4: 开始SnowNLP模型训练...")
            self.current_step = 3
            self.update_training_status("model_training", 3)
            self.update_enhanced_progress(55, 10, "初始化SnowNLP训练...")
            self.update_step_status(3, 10)
            
            # 估算训练时间（基于样本数量）
            estimated_training_time = max(30, len(train_texts) * 0.0001)  # 至少30秒
            self.log(f"📊 预估训练时间: {estimated_training_time:.1f} 秒")
            
            # 开始模型训练
            self.update_enhanced_progress(60, 30, "SnowNLP核心算法训练中...")
            self.update_step_status(3, 30)
            
            success = self.train_and_replace_model(neg_path, pos_path)
            
            if not success:
                self.log("❌ 模型训练失败")
                self.update_training_status("failed")
                self.update_step_status(3, 0, False, True)  # 显示失败状态
                messagebox.showerror("训练失败", "❌ 模型训练失败\n\n请检查:\n• 数据文件格式是否正确\n• 是否有足够的正负面样本\n• 查看日志了解详细错误")
                return
            
            self.update_enhanced_progress(75, 100, "模型训练完成")
            self.update_step_status(3, 100, True)
            self.log("✅ 步骤4: 模型训练完成")
            
            # 步骤5: 模型部署 (85-95%)
            self.log("🔄 步骤5: 开始部署新模型...")
            self.current_step = 4
            self.update_training_status("model_deploying", 4)
            self.update_enhanced_progress(80, 50, "正在部署新模型...")
            self.update_step_status(4, 50)
            
            self.update_enhanced_progress(85, 100, "新模型部署完成")
            self.update_step_status(4, 100, True)
            self.log("✅ 步骤5: 模型部署完成")
            
            # 步骤6: 验证训练结果 (95-100%)
            self.log("✅ 步骤6: 开始验证训练结果...")
            self.current_step = 5
            self.update_training_status("final_testing", 5)
            self.update_enhanced_progress(90, 50, "正在验证新模型性能...")
            self.update_step_status(5, 50)
            
            if self.test_files and test_texts:
                trained_acc = self.evaluate_model_simple(test_texts, test_labels)
                self.log(f"📊 训练后模型准确率: {trained_acc:.2%}")
                
                if base_acc is not None:
                    improvement = (trained_acc - base_acc) * 100
                    self.log(f"📈 准确率提升: {improvement:.2f}%")
                    
                    # 更新性能指标
                    perf_text = f"基线: {base_acc:.2%} → 训练后: {trained_acc:.2%} | 提升: {improvement:.2f}%"
                    if improvement > 0:
                        perf_text += " 🎉"
                    self.update_performance_metrics(perf_text)
                else:
                    self.update_performance_metrics(f"训练后准确率: {trained_acc:.2%}")
            else:
                self.update_performance_metrics("训练完成 - 建议使用测试功能验证效果")
            
            # 完成训练
            self.update_enhanced_progress(100, 100, "训练成功完成！")
            self.update_step_status(5, 100, True)
            self.update_training_status("completed")
            self.log("✅ 步骤6: 验证完成")
            
            self.log("✅ 模型训练和替换成功!")
            self.log("🔄 建议重启Python解释器以确保使用新模型")
            
            # 显示完成时间
            finish_time = datetime.datetime.now()
            total_time = finish_time - self.training_start_time
            self.log(f"⏱️ 总训练时间: {str(total_time).split('.')[0]}")
            
            # 显示成功对话框
            messagebox.showinfo("训练完成", 
                f"🎉 模型训练成功完成！\n\n" +
                f"⏱️ 训练时间: {str(total_time).split('.')[0]}\n" +
                f"📊 训练样本: {len(train_texts)} 个\n" +
                "✅ 新模型已安装\n" +
                "📊 可以使用测试功能验证效果\n" +
                "🔄 建议重启程序以确保使用新模型")
                
        except Exception as e:
            self.log(f"❌ 训练过程出错: {e}")
            self.update_training_status("failed")
            self.update_enhanced_progress(None, None, "训练异常中断")
            
            # 标记当前步骤为失败
            if hasattr(self, 'current_step') and self.current_step < len(self.training_steps):
                self.update_step_status(self.current_step, 0, False, True)
            
            import traceback
            self.log(f"详细错误: {traceback.format_exc()}")
            
            messagebox.showerror("训练异常", f"❌ 训练过程出现异常:\n{e}\n\n建议:\n• 检查数据文件完整性\n• 重启程序后重试\n• 查看日志了解详细信息")
        finally:
            self.training_running = False
            self.train_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.current_task.stop()
            
            # 最终时间更新
            if self.training_start_time:
                self.remaining_time_var.set("已完成")
                self.finish_time_var.set("已完成")
    
    def update_file_status(self):
        """更新文件状态"""
        if self.train_files:
            if self.test_files:
                self.update_status_guide("files_selected", "训练和测试文件已就绪 - 可以开始训练")
            else:
                self.update_status_guide("files_selected", "训练文件已选择 - 建议也选择测试文件")
        else:
            self.update_status_guide("ready")
    
    def load_data(self, filepaths, data_type="数据"):
        """加载数据文件"""
        try:
            neutral_strategy = self.neutral_strategy.get()
            self.log(f"开始加载 {len(filepaths)} 个{data_type}文件...")
            self.log(f"中性数据处理策略: {neutral_strategy}")
            
            # 扩展的标签映射 - 包含更多可能的标签格式
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
                
                # 情感标签 (从JSON数据来的)
                'angry': 0, 'sad': 0, 'fear': 0,
                'happy': 1, 'surprise': 1,
            }

            all_texts, all_labels = [], []
            neutral_texts = []
            unknown_labels = set()  # 记录未知标签
            total_rows = 0  # 总行数
            
            for path in filepaths:
                if not os.path.exists(path):
                    self.log(f"文件不存在: {path}")
                    continue
                
                try:
                    self.log(f"正在加载文件: {path}")

                    from data_io import read_sentiment_csv
                    result = read_sentiment_csv(path)
                    df = result.df
                    self.log(f"成功加载: 编码={result.encoding}, 分隔符={repr(result.sep)}")
                    
                    self.log(f"成功加载，共 {len(df)} 行数据")
                    total_rows += len(df)
                        
                except Exception as e:
                    self.log(f"读取文件失败 {path}: {e}")
                    continue
                    
                texts = df['content'].astype(str).tolist()
                labels = []
                valid_indices = []
                neutral_indices = []
                
                # 统计这个文件中的标签分布
                file_label_counts = {}

                for i, label in enumerate(df['sentiment']):
                    # 处理各种可能的标签格式
                    if pd.isna(label):
                        continue
                        
                    # 如果是数字，直接使用
                    if isinstance(label, (int, float)):
                        label_key = int(label)
                    else:
                        label_key = str(label).strip().lower()
                    
                    # 统计标签出现次数
                    file_label_counts[label_key] = file_label_counts.get(label_key, 0) + 1
                    
                    mapped = label_mapping.get(label_key, None)

                    if mapped == 'neutral':
                        neutral_indices.append(i)
                    elif mapped is not None:
                        labels.append(mapped)
                        valid_indices.append(i)
                    else:
                        # 记录未知标签
                        unknown_labels.add(str(label_key))
                
                # 报告这个文件的标签分布
                self.log(f"文件 {os.path.basename(path)} 标签分布:")
                for label_key, count in sorted(file_label_counts.items()):
                    mapped = label_mapping.get(label_key, "未知")
                    self.log(f"  '{label_key}' -> {mapped}: {count} 个")

                all_texts.extend([texts[i] for i in valid_indices])
                all_labels.extend(labels)
                neutral_texts.extend([texts[i] for i in neutral_indices])

            # 报告数据加载摘要
            self.log(f"\n📊 数据加载摘要:")
            self.log(f"总文件数: {len(filepaths)}")
            self.log(f"总行数: {total_rows}")
            self.log(f"有效样本: {len(all_texts)}")
            self.log(f"中性样本: {len(neutral_texts)}")
            self.log(f"数据利用率: {(len(all_texts) + len(neutral_texts)) / total_rows * 100:.1f}%")
            
            if unknown_labels:
                self.log(f"\n⚠️ 发现未知标签 (被跳过的数据):")
                for label in sorted(unknown_labels):
                    self.log(f"  '{label}'")
                self.log(f"\n💡 建议: 如果这些标签应该被处理，请联系开发者添加标签映射")

            # 处理中性数据
            current_pos = sum(1 for label in all_labels if label == 1)
            current_neg = sum(1 for label in all_labels if label == 0)
            neutral_count = len(neutral_texts)
            
            self.log(f"\n原始数据统计:")
            self.log(f"  正面样本: {current_pos}")
            self.log(f"  负面样本: {current_neg}")  
            self.log(f"  中性样本: {neutral_count}")

            if neutral_count > 0 and neutral_strategy != 'exclude':
                self.log(f"正在处理 {neutral_count} 个中性样本...")
                
                if neutral_strategy == 'random':
                    for text in neutral_texts:
                        label = random.choice([0, 1])
                        all_texts.append(text)
                        all_labels.append(label)
                elif neutral_strategy == 'balance':
                    if current_pos < current_neg:
                        for text in neutral_texts:
                            all_texts.append(text)
                            all_labels.append(1)
                        self.log(f"  中性样本全部分配给正面类别(平衡数据)")
                    else:
                        for text in neutral_texts:
                            all_texts.append(text)
                            all_labels.append(0)
                        self.log(f"  中性样本全部分配给负面类别(平衡数据)")
                elif neutral_strategy == 'positive':
                    for text in neutral_texts:
                        all_texts.append(text)
                        all_labels.append(1)
                    self.log(f"  中性样本全部分配给正面类别")
                elif neutral_strategy == 'negative':
                    for text in neutral_texts:
                        all_texts.append(text)
                        all_labels.append(0)
                    self.log(f"  中性样本全部分配给负面类别")
                elif neutral_strategy == 'split':
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
                        
                    self.log(f"  中性样本按比例分配: {len(pos_neutrals)}个给正面, {len(neg_neutrals)}个给负面")

            final_pos = sum(1 for label in all_labels if label == 1)
            final_neg = sum(1 for label in all_labels if label == 0)
            
            self.log(f"\n最终数据统计:")
            self.log(f"  正面样本: {final_pos}")
            self.log(f"  负面样本: {final_neg}")
            self.log(f"  总样本数: {len(all_texts)}")
            
            # 记录训练样本数量供模型管理器使用
            if data_type == "训练":
                self.current_train_samples = len(all_texts)
            
            self.log(f"✅ {data_type}加载完成: {len(all_texts)} 个样本")
            return all_texts, all_labels
            
        except Exception as e:
            self.log(f"❌ {data_type}加载失败: {e}")
            import traceback
            self.log(f"详细错误: {traceback.format_exc()}")
            return [], []
    
    def create_sentiment_files(self, texts, labels, pos_path, neg_path):
        """创建情感语料文件"""
        try:
            os.makedirs(os.path.dirname(pos_path), exist_ok=True)
            os.makedirs(os.path.dirname(neg_path), exist_ok=True)

            with open(pos_path, 'w', encoding='utf-8') as f_pos, \
                 open(neg_path, 'w', encoding='utf-8') as f_neg:

                pos_count, neg_count = 0, 0
                for text, label in zip(texts, labels):
                    clean_text = text.replace('\n', '').replace('\r', '').strip()
                    if len(clean_text) > 0:
                        if label == 1:
                            f_pos.write(clean_text + '\n')
                            pos_count += 1
                        elif label == 0:
                            f_neg.write(clean_text + '\n')
                            neg_count += 1

            self.log(f"✅ 语料文件创建完成: {pos_count} 正面, {neg_count} 负面")
            return pos_count, neg_count
            
        except Exception as e:
            self.log(f"❌ 语料文件创建失败: {e}")
            return 0, 0
    
    def train_and_replace_model(self, neg_path, pos_path):
        """训练并替换模型"""
        try:
            self.log("🔧 开始模型训练和替换...")
            
            # 1. 先进行基础训练
            self.log("正在训练模型...")
            sentiment.train(neg_path, pos_path)
            self.log("✅ 模型训练完成")
            
            # 2. 查找生成的模型文件
            possible_model_files = [
                'custom_sentiment.marshal.3',
                'sentiment.marshal',
                'sentiment.marshal.3',
                'custom_sentiment.model'
            ]
            
            source_file = None
            for fname in possible_model_files:
                if os.path.exists(fname):
                    source_file = fname
                    self.log(f"找到训练生成的模型文件: {fname}")
                    break
            
            if not source_file:
                self.log("❌ 未找到训练生成的模型文件")
                return False
            
            # 3. 检查源文件
            file_size = os.path.getsize(source_file)
            self.log(f"模型文件大小: {file_size:,} 字节")
            
            if file_size < 50000:  # 小于50KB可能不是有效模型
                self.log("⚠️ 警告：模型文件大小较小")
            
            # 4. 创建带时间戳的模型副本
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_copy = f"model_{timestamp}.marshal.3"
            shutil.copy2(source_file, model_copy)
            self.log(f"✅ 创建模型副本: {model_copy}")
            
            # 5. 获取SnowNLP系统路径
            import snownlp
            snownlp_dir = os.path.dirname(snownlp.__file__)
            sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
            
            self.log(f"SnowNLP系统路径: {sentiment_dir}")
            
            # 6. 查找目标文件
            target_files = []
            for fname in ['sentiment.marshal', 'sentiment.marshal.3']:
                fpath = os.path.join(sentiment_dir, fname)
                if os.path.exists(fpath):
                    target_files.append(fpath)
                    self.log(f"找到目标文件: {fname}")
            
            if not target_files:
                self.log("❌ 未找到目标模型文件")
                return False
            
            # 7. 备份原文件
            for target_file in target_files:
                backup_file = target_file + '.backup_gui'
                if not os.path.exists(backup_file):
                    shutil.copy2(target_file, backup_file)
                    self.log(f"✅ 创建备份: {os.path.basename(backup_file)}")
                else:
                    self.log(f"备份已存在: {os.path.basename(backup_file)}")
            
            # 8. 复制新模型到系统位置
            success_count = 0
            for target_file in target_files:
                try:
                    shutil.copy2(source_file, target_file)
                    new_size = os.path.getsize(target_file)
                    fname = os.path.basename(target_file)
                    self.log(f"✅ 模型替换成功: {fname} ({new_size:,} 字节)")
                    success_count += 1
                except Exception as e:
                    fname = os.path.basename(target_file)
                    self.log(f"❌ 模型替换失败 {fname}: {e}")
            
            if success_count > 0:
                # 9. 保存模型到管理器
                model_info = {
                    'name': f"训练模型_{timestamp}",
                    'train_files': [os.path.basename(f) for f in self.train_files],
                    'test_files': [os.path.basename(f) for f in self.test_files],
                    'train_samples': getattr(self, 'current_train_samples', 0),
                    'neutral_strategy': self.neutral_strategy.get(),
                    'notes': f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
                
                model_id = self.model_manager.add_model(model_copy, model_info)
                self.log(f"📦 模型已保存到管理器: {model_id}")
                
                self.log(f"🎉 成功替换 {success_count} 个模型文件！")
                return True
            else:
                self.log("❌ 所有模型文件替换都失败了")
                return False
                
        except Exception as e:
            self.log(f"❌ 模型训练替换失败: {e}")
            import traceback
            self.log(f"详细错误: {traceback.format_exc()}")
            return False
    
    def evaluate_model_simple(self, test_texts, test_labels):
        """简单模型评估"""
        try:
            correct = 0
            total = min(len(test_texts), 500)  # 限制测试样本数量
            
            for i in range(total):
                if not self.training_running:
                    break
                    
                text, label = test_texts[i], test_labels[i]
                try:
                    s = SnowNLP(text)
                    score = s.sentiments
                    pred_label = 1 if score > 0.5 else 0
                    if pred_label == label:
                        correct += 1
                except:
                    continue
            
            return correct / total if total > 0 else 0
            
        except Exception as e:
            self.log(f"❌ 模型评估失败: {e}")
            return 0
    
    def quick_test(self):
        """快速验证"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "🚀 快速验证测试\n" + "="*50 + "\n\n")
        
        test_cases = [
            ("这个产品质量很好，非常满意！", "正面"),
            ("服务态度太差了，很不满意", "负面"),
            ("还可以吧，一般般", "中性"),
            ("物流速度很快，包装也不错", "正面"),
            ("价格有点贵，但质量确实好", "正面")
        ]
        
        correct = 0
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
                
                status = "✅" if predicted == expected or expected == "中性" else "❌"
                if predicted == expected or expected == "中性":
                    correct += 1
                
                self.result_text.insert(tk.END, f"{status} [{i}] {score:.4f} ({predicted}) | {text}\n")
                
            except Exception as e:
                self.result_text.insert(tk.END, f"❌ [{i}] 测试失败: {e}\n")
        
        accuracy = correct / len(test_cases)
        self.result_text.insert(tk.END, f"\n📊 准确率: {accuracy:.2%}\n")
        
        if accuracy >= 0.8:
            self.result_text.insert(tk.END, "🎉 优秀！模型表现很好\n")
        elif accuracy >= 0.6:
            self.result_text.insert(tk.END, "👍 良好！模型表现不错\n")
        else:
            self.result_text.insert(tk.END, "😐 一般！模型需要改进\n")
    
    def full_test(self):
        """完整测试"""
        # 在新线程中执行，避免界面冻结
        test_thread = threading.Thread(target=self.full_test_worker)
        test_thread.daemon = True
        test_thread.start()
    
    def full_test_worker(self):
        """完整测试工作线程"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "🧪 完整模型测试\n" + "="*50 + "\n\n")
        
        # 基础测试
        self.result_text.insert(tk.END, "进行基础情感分析测试...\n")
        self.root.update()
        
        # 这里可以调用更完整的测试逻辑
        basic_accuracy = self.run_basic_test()
        
        # 数据集测试
        if self.test_files:
            self.result_text.insert(tk.END, "\n进行数据集评估...\n")
            self.root.update()
            dataset_accuracy = self.run_dataset_test()
        else:
            dataset_accuracy = None
        
        # 总结
        self.result_text.insert(tk.END, f"\n{'='*50}\n")
        self.result_text.insert(tk.END, "📋 测试总结\n")
        self.result_text.insert(tk.END, f"{'='*50}\n")
        
        if basic_accuracy is not None:
            self.result_text.insert(tk.END, f"基础测试准确率: {basic_accuracy:.2%}\n")
        
        if dataset_accuracy is not None:
            self.result_text.insert(tk.END, f"数据集测试准确率: {dataset_accuracy:.2%}\n")
            
        if basic_accuracy and dataset_accuracy:
            avg_accuracy = (basic_accuracy + dataset_accuracy) / 2
            self.result_text.insert(tk.END, f"平均准确率: {avg_accuracy:.2%}\n")
            
            if avg_accuracy >= 0.75:
                self.result_text.insert(tk.END, "🎉 模型表现优秀！\n")
            elif avg_accuracy >= 0.6:
                self.result_text.insert(tk.END, "👍 模型表现良好！\n")
            else:
                self.result_text.insert(tk.END, "😐 模型需要进一步优化\n")
    
    def run_basic_test(self):
        """运行基础测试"""
        # 扩展的测试用例
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
                
                # 只计算非中性样本的准确率
                if expected != "中性":
                    total += 1
                    if predicted == expected:
                        correct += 1
                
                status = "✅" if predicted == expected or expected == "中性" else "❌"
                self.result_text.insert(tk.END, f"{status} [{i:2d}] {score:.4f} ({predicted:^4}) | {text}\n")
                self.root.update()
                
            except Exception as e:
                self.result_text.insert(tk.END, f"❌ [{i:2d}] 测试失败: {e}\n")
        
        accuracy = correct / total if total > 0 else 0
        self.result_text.insert(tk.END, f"\n📊 基础测试结果: {correct}/{total} 正确，准确率: {accuracy:.2%}\n")
        
        return accuracy
    
    def run_dataset_test(self):
        """运行数据集测试"""
        try:
            # 加载测试数据
            test_texts, test_labels = self.load_data(self.test_files, "测试")
            
            if not test_texts:
                self.result_text.insert(tk.END, "❌ 测试数据加载失败\n")
                return None
            
            self.result_text.insert(tk.END, f"测试样本数: {len(test_texts)}\n")
            
            # 如果数据量很大，询问用户是否采样
            use_sampling = False
            if len(test_texts) > 5000:
                self.result_text.insert(tk.END, f"⚠️ 检测到大数据集({len(test_texts)}个样本)\n")
                
                # 创建对话框询问用户
                from tkinter import messagebox
                choice = messagebox.askyesnocancel(
                    "数据集选项", 
                    f"检测到大数据集({len(test_texts)}个样本)\n\n"
                    "选择测试方式:\n"
                    "• 是(Yes): 采样5000个样本快速测试\n" 
                    "• 否(No): 测试全部数据(可能较慢)\n"
                    "• 取消: 停止测试"
                )
                
                if choice is None:  # 用户选择取消
                    self.result_text.insert(tk.END, "❌ 用户取消测试\n")
                    return None
                elif choice:  # 用户选择采样
                    use_sampling = True
                    max_samples = 5000
                    self.result_text.insert(tk.END, f"✅ 用户选择采样测试({max_samples}个样本)\n")
                else:  # 用户选择全部测试
                    self.result_text.insert(tk.END, f"✅ 用户选择测试全部数据({len(test_texts)}个样本)\n")
            
            # 根据用户选择进行采样
            if use_sampling and len(test_texts) > max_samples:
                indices = random.sample(range(len(test_texts)), max_samples)
                test_texts = [test_texts[i] for i in indices]
                test_labels = [test_labels[i] for i in indices]
                self.result_text.insert(tk.END, f"已随机采样 {len(test_texts)} 个样本进行测试\n")
            
            # 统计标签分布
            pos_count = sum(1 for label in test_labels if label == 1)
            neg_count = sum(1 for label in test_labels if label == 0)
            
            self.result_text.insert(tk.END, f"数据分布: 正面 {pos_count}, 负面 {neg_count}\n")
            self.result_text.insert(tk.END, "开始评估...\n")
            
            # 添加预估时间
            if len(test_texts) > 1000:
                estimated_time = len(test_texts) * 0.01  # 估算每个样本0.01秒
                self.result_text.insert(tk.END, f"预估测试时间: {estimated_time:.1f}秒\n")
            
            self.root.update()
            
            # 评估
            correct = 0
            total_processed = 0
            
            for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
                try:
                    s = SnowNLP(text)
                    score = s.sentiments
                    pred_label = 1 if score > 0.5 else 0
                    
                    if pred_label == true_label:
                        correct += 1
                    
                    total_processed += 1
                    
                    # 根据数据量调整进度更新频率
                    update_freq = max(100, len(test_texts) // 20)  # 最少100个，最多20次更新
                    if (i + 1) % update_freq == 0:
                        progress = (i + 1) / len(test_texts) * 100
                        current_acc = correct / total_processed if total_processed > 0 else 0
                        self.result_text.insert(tk.END, f"进度: {progress:.1f}% ({i+1}/{len(test_texts)}) 当前准确率: {current_acc:.2%}\n")
                        self.root.update()
                        
                except Exception:
                    continue
            
            accuracy = correct / total_processed if total_processed > 0 else 0
            self.result_text.insert(tk.END, f"\n📊 数据集测试结果: {correct}/{total_processed} 正确，准确率: {accuracy:.2%}\n")
            
            if total_processed != len(test_texts):
                success_rate = total_processed / len(test_texts)
                self.result_text.insert(tk.END, f"处理成功率: {success_rate:.2%}\n")
            
            return accuracy
            
        except Exception as e:
            self.result_text.insert(tk.END, f"❌ 数据集测试失败: {e}\n")
            return None
    
    def dataset_evaluation(self):
        """数据集评估"""
        if not self.test_files:
            messagebox.showwarning("提示", "请先选择测试数据文件")
            return
        
        # 在新线程中执行
        eval_thread = threading.Thread(target=self.run_dataset_test_display)
        eval_thread.daemon = True
        eval_thread.start()
    
    def run_dataset_test_display(self):
        """运行数据集测试并显示"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "📊 数据集评估\n" + "="*50 + "\n\n")
        accuracy = self.run_dataset_test()
        
        if accuracy is not None:
            if accuracy >= 0.8:
                self.result_text.insert(tk.END, "🎉 优秀！模型表现很好\n")
            elif accuracy >= 0.6:
                self.result_text.insert(tk.END, "👍 良好！模型表现不错\n")
            elif accuracy >= 0.4:
                self.result_text.insert(tk.END, "😐 一般！模型需要改进\n")
            else:
                self.result_text.insert(tk.END, "😞 较差！建议重新训练\n")
    
    def show_model_info(self):
        """显示模型信息"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "ℹ️ SnowNLP模型信息\n" + "="*50 + "\n\n")
        
        try:
            import snownlp
            snownlp_dir = os.path.dirname(snownlp.__file__)
            sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
            
            self.result_text.insert(tk.END, f"SnowNLP安装路径: {snownlp_dir}\n")
            self.result_text.insert(tk.END, f"Sentiment模块路径: {sentiment_dir}\n\n")
            
            # 检查模型文件
            model_files = ['sentiment.marshal', 'sentiment.marshal.3']
            for fname in model_files:
                fpath = os.path.join(sentiment_dir, fname)
                if os.path.exists(fpath):
                    size = os.path.getsize(fpath)
                    mtime = os.path.getmtime(fpath)
                    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                    self.result_text.insert(tk.END, f"模型文件: {fname}\n")
                    self.result_text.insert(tk.END, f"  大小: {size:,} 字节\n")
                    self.result_text.insert(tk.END, f"  修改时间: {mtime_str}\n")
                    
                    # 检查备份文件
                    backup_files = [f for f in os.listdir(sentiment_dir) if fname in f and 'backup' in f]
                    if backup_files:
                        self.result_text.insert(tk.END, f"  备份文件: {len(backup_files)} 个\n")
                    self.result_text.insert(tk.END, "\n")
            
            # 快速测试
            test_text = "这是一个测试文本"
            s = SnowNLP(test_text)
            score = s.sentiments
            self.result_text.insert(tk.END, f"快速测试: '{test_text}' → {score:.4f}\n")
            
        except Exception as e:
            self.result_text.insert(tk.END, f"❌ 获取模型信息失败: {e}\n")
    
    def analyze_text(self):
        """分析输入文本的情感"""
        text = self.test_input.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("提示", "请输入要分析的文本")
            return
        
        try:
            s = SnowNLP(text)
            score = s.sentiments
            
            if score > 0.6:
                sentiment = "正面 😊"
                color = "green"
            elif score < 0.4:
                sentiment = "负面 😞"
                color = "red"
            else:
                sentiment = "中性 😐"
                color = "orange"
            
            result = f"得分: {score:.4f} | 情感: {sentiment}"
            
            # 额外提示
            if score > 0.8:
                result += " (强烈正面)"
            elif score < 0.2:
                result += " (强烈负面)"
            elif 0.45 <= score <= 0.55:
                result += " (情感模糊)"
            
            self.test_result.set(result)
            
        except Exception as e:
            self.test_result.set(f"分析失败: {e}")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.test_result.set("等待输入...")
    
    def select_model_test(self):
        """选择模型文件进行测试"""
        # 在新线程中执行
        test_thread = threading.Thread(target=self.select_model_test_worker)
        test_thread.daemon = True
        test_thread.start()
    
    def select_model_test_worker(self):
        """选择模型测试工作线程"""
        # 让用户选择模型文件
        model_file = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[
                ("Marshal文件", "*.marshal*"),
                ("模型文件", "*.model"),
                ("所有文件", "*.*")
            ]
        )
        
        if not model_file:
            return
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "🔄 选择模型测试\n" + "="*50 + "\n\n")
        self.result_text.insert(tk.END, f"选择的模型文件: {os.path.basename(model_file)}\n")
        
        # 临时替换模型进行测试
        success = self.temp_replace_model(model_file)
        
        if success:
            self.result_text.insert(tk.END, "✅ 模型临时替换成功，开始测试...\n\n")
            
            # 运行基础测试
            accuracy = self.run_basic_test()
            
            self.result_text.insert(tk.END, f"\n📊 使用模型 {os.path.basename(model_file)} 的测试结果:\n")
            self.result_text.insert(tk.END, f"准确率: {accuracy:.2%}\n")
            
            if accuracy >= 0.8:
                self.result_text.insert(tk.END, "🎉 该模型表现优秀！\n")
            elif accuracy >= 0.6:
                self.result_text.insert(tk.END, "👍 该模型表现良好！\n")
            else:
                self.result_text.insert(tk.END, "😐 该模型需要改进\n")
        else:
            self.result_text.insert(tk.END, "❌ 模型替换失败，无法进行测试\n")
    
    def select_data_test(self):
        """选择数据集进行测试"""
        # 让用户选择测试数据文件
        data_files = filedialog.askopenfilenames(
            title="选择测试数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if not data_files:
            return
        
        # 在新线程中执行测试
        test_thread = threading.Thread(target=self.select_data_test_worker, args=(data_files,))
        test_thread.daemon = True
        test_thread.start()
    
    def select_data_test_worker(self, data_files):
        """选择数据测试工作线程"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "📊 选择数据集测试\n" + "="*50 + "\n\n")
        
        for i, data_file in enumerate(data_files, 1):
            self.result_text.insert(tk.END, f"数据文件 {i}: {os.path.basename(data_file)}\n")
        
        self.result_text.insert(tk.END, "\n开始加载测试数据...\n")
        
        # 加载测试数据
        test_texts, test_labels = self.load_data(list(data_files), "测试")
        
        if not test_texts:
            self.result_text.insert(tk.END, "❌ 测试数据加载失败\n")
            return
        
        # 统计数据分布
        pos_count = sum(1 for label in test_labels if label == 1)
        neg_count = sum(1 for label in test_labels if label == 0)
        
        self.result_text.insert(tk.END, f"测试样本统计:\n")
        self.result_text.insert(tk.END, f"  总计: {len(test_texts)} 个样本\n")
        self.result_text.insert(tk.END, f"  正面: {pos_count} 个\n")
        self.result_text.insert(tk.END, f"  负面: {neg_count} 个\n\n")
        
        # 如果数据量很大，询问用户是否采样
        use_sampling = False
        if len(test_texts) > 5000:
            self.result_text.insert(tk.END, f"⚠️ 检测到大数据集({len(test_texts)}个样本)\n")
            
            # 创建对话框询问用户
            from tkinter import messagebox
            choice = messagebox.askyesnocancel(
                "数据集选项", 
                f"检测到大数据集({len(test_texts)}个样本)\n\n"
                "选择测试方式:\n"
                "• 是(Yes): 采样5000个样本快速测试\n" 
                "• 否(No): 测试全部数据(可能较慢)\n"
                "• 取消: 停止测试"
            )
            
            if choice is None:  # 用户选择取消
                self.result_text.insert(tk.END, "❌ 用户取消测试\n")
                return
            elif choice:  # 用户选择采样
                use_sampling = True
                max_samples = 5000
                self.result_text.insert(tk.END, f"✅ 用户选择采样测试({max_samples}个样本)\n")
            else:  # 用户选择全部测试
                self.result_text.insert(tk.END, f"✅ 用户选择测试全部数据({len(test_texts)}个样本)\n")
        
        # 根据用户选择进行采样
        if use_sampling and len(test_texts) > max_samples:
            indices = random.sample(range(len(test_texts)), max_samples)
            test_texts = [test_texts[i] for i in indices]
            test_labels = [test_labels[i] for i in indices]
            self.result_text.insert(tk.END, f"已随机采样 {len(test_texts)} 个样本进行测试\n")
        
        # 开始评估
        self.result_text.insert(tk.END, "开始评估当前模型...\n")
        
        correct = 0
        total_processed = 0
        
        # 添加预估时间
        if len(test_texts) > 1000:
            estimated_time = len(test_texts) * 0.01  # 估算每个样本0.01秒
            self.result_text.insert(tk.END, f"预估测试时间: {estimated_time:.1f}秒\n")
        
        for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
            try:
                s = SnowNLP(text)
                score = s.sentiments
                pred_label = 1 if score > 0.5 else 0
                
                if pred_label == true_label:
                    correct += 1
                
                total_processed += 1
                
                # 根据数据量调整进度更新频率
                update_freq = max(50, len(test_texts) // 20)  # 最少50个，最多20次更新
                if (i + 1) % update_freq == 0:
                    progress = (i + 1) / len(test_texts) * 100
                    current_acc = correct / total_processed if total_processed > 0 else 0
                    self.result_text.insert(tk.END, f"进度: {progress:.1f}% ({i+1}/{len(test_texts)}) 当前准确率: {current_acc:.2%}\n")
                    self.root.update()
                    
            except Exception:
                continue
        
        accuracy = correct / total_processed if total_processed > 0 else 0
        
        # 计算各类别准确率
        pos_correct = neg_correct = 0
        pos_total = neg_total = 0
        
        for text, true_label in zip(test_texts, test_labels):
            try:
                s = SnowNLP(text)
                score = s.sentiments
                pred_label = 1 if score > 0.5 else 0
                
                if true_label == 1:
                    pos_total += 1
                    if pred_label == 1:
                        pos_correct += 1
                else:
                    neg_total += 1
                    if pred_label == 0:
                        neg_correct += 1
            except:
                continue
        
        pos_acc = pos_correct / pos_total if pos_total > 0 else 0
        neg_acc = neg_correct / neg_total if neg_total > 0 else 0
        
        # 显示详细结果
        self.result_text.insert(tk.END, f"\n📊 详细测试结果:\n")
        self.result_text.insert(tk.END, f"处理样本数: {total_processed}/{len(test_texts)}\n")
        self.result_text.insert(tk.END, f"总体准确率: {accuracy:.2%} ({correct}/{total_processed})\n")
        self.result_text.insert(tk.END, f"正面准确率: {pos_acc:.2%} ({pos_correct}/{pos_total})\n")
        self.result_text.insert(tk.END, f"负面准确率: {neg_acc:.2%} ({neg_correct}/{neg_total})\n")
        
        # 添加更详细的统计
        if total_processed != len(test_texts):
            success_rate = total_processed / len(test_texts)
            self.result_text.insert(tk.END, f"处理成功率: {success_rate:.2%}\n")
        
        if accuracy >= 0.8:
            self.result_text.insert(tk.END, "\n🎉 在该数据集上表现优秀！\n")
        elif accuracy >= 0.6:
            self.result_text.insert(tk.END, "\n👍 在该数据集上表现良好！\n")
        elif accuracy >= 0.4:
            self.result_text.insert(tk.END, "\n😐 在该数据集上表现一般\n")
        else:
            self.result_text.insert(tk.END, "\n😞 在该数据集上表现较差\n")
    
    def model_comparison(self):
        """模型对比测试"""
        # 让用户选择多个模型文件
        model_files = filedialog.askopenfilenames(
            title="选择要对比的模型文件",
            filetypes=[
                ("Marshal文件", "*.marshal*"),
                ("模型文件", "*.model"),
                ("所有文件", "*.*")
            ]
        )
        
        if len(model_files) < 2:
            messagebox.showwarning("提示", "请至少选择2个模型文件进行对比")
            return
        
        # 在新线程中执行对比
        compare_thread = threading.Thread(target=self.model_comparison_worker, args=(model_files,))
        compare_thread.daemon = True
        compare_thread.start()
    
    def model_comparison_worker(self, model_files):
        """模型对比工作线程"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "🔄 模型对比测试\n" + "="*50 + "\n\n")
        
        # 显示要对比的模型
        self.result_text.insert(tk.END, f"对比模型数量: {len(model_files)}\n")
        for i, model_file in enumerate(model_files, 1):
            size = os.path.getsize(model_file) if os.path.exists(model_file) else 0
            self.result_text.insert(tk.END, f"模型 {i}: {os.path.basename(model_file)} ({size:,} 字节)\n")
        
        # 准备测试用例
        test_cases = [
            ("这个产品质量非常好，强烈推荐大家购买！", "正面"),
            ("服务态度超棒，物流也很快，非常满意", "正面"),
            ("性价比很高，用了一段时间效果很不错", "正面"),
            ("质量太差了，完全不值这个价格", "负面"),
            ("服务态度恶劣，客服回复很慢很敷衍", "负面"),
            ("物流超级慢，包装也很粗糙", "负面"),
            ("价格有点贵，但是质量确实不错", "正面"),
            ("功能很好，就是界面有点丑", "正面"),
            ("还可以吧，凑合能用", "中性"),
            ("和描述基本一致", "中性")
        ]
        
        # 备份原始模型
        original_model_backup = self.backup_current_model()
        
        results = {}
        
        # 测试每个模型
        for i, model_file in enumerate(model_files, 1):
            self.result_text.insert(tk.END, f"\n正在测试模型 {i}: {os.path.basename(model_file)}\n")
            self.result_text.insert(tk.END, "-" * 40 + "\n")
            
            # 临时替换模型
            if self.temp_replace_model(model_file):
                correct = 0
                total = 0
                model_results = []
                
                for text, expected in test_cases:
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
                        model_results.append((text, score, predicted, expected, is_correct))
                        self.result_text.insert(tk.END, f"{status} {score:.4f} ({predicted}) | {text}\n")
                        
                    except Exception as e:
                        self.result_text.insert(tk.END, f"❌ 测试失败: {e}\n")
                
                accuracy = correct / total if total > 0 else 0
                results[os.path.basename(model_file)] = {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'results': model_results
                }
                
                self.result_text.insert(tk.END, f"准确率: {accuracy:.2%} ({correct}/{total})\n")
                
            else:
                self.result_text.insert(tk.END, "❌ 模型加载失败\n")
                results[os.path.basename(model_file)] = None
        
        # 恢复原始模型
        if original_model_backup:
            self.restore_model(original_model_backup)
        
        # 显示对比总结
        self.result_text.insert(tk.END, f"\n{'='*50}\n")
        self.result_text.insert(tk.END, "📊 模型对比总结\n")
        self.result_text.insert(tk.END, f"{'='*50}\n")
        
        # 按准确率排序
        valid_results = [(name, data) for name, data in results.items() if data is not None]
        valid_results.sort(key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (model_name, data) in enumerate(valid_results, 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            self.result_text.insert(tk.END, f"{rank_emoji} {model_name}: {data['accuracy']:.2%}\n")
        
        if valid_results:
            best_model = valid_results[0]
            self.result_text.insert(tk.END, f"\n🏆 最佳模型: {best_model[0]}\n")
            self.result_text.insert(tk.END, f"准确率: {best_model[1]['accuracy']:.2%}\n")
    
    def temp_replace_model(self, model_file):
        """临时替换模型文件"""
        try:
            if not os.path.exists(model_file):
                return False
            
            # 获取SnowNLP系统路径
            import snownlp
            snownlp_dir = os.path.dirname(snownlp.__file__)
            sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
            
            # 查找目标文件
            target_files = []
            for fname in ['sentiment.marshal', 'sentiment.marshal.3']:
                fpath = os.path.join(sentiment_dir, fname)
                if os.path.exists(fpath):
                    target_files.append(fpath)
            
            if not target_files:
                return False
            
            # 替换模型文件
            for target_file in target_files:
                shutil.copy2(model_file, target_file)
            
            return True
            
        except Exception as e:
            self.log(f"临时模型替换失败: {e}")
            return False
    
    def backup_current_model(self):
        """备份当前模型"""
        try:
            import snownlp
            snownlp_dir = os.path.dirname(snownlp.__file__)
            sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
            
            backup_files = []
            for fname in ['sentiment.marshal', 'sentiment.marshal.3']:
                fpath = os.path.join(sentiment_dir, fname)
                if os.path.exists(fpath):
                    backup_path = fpath + '.temp_backup'
                    shutil.copy2(fpath, backup_path)
                    backup_files.append(backup_path)
            
            return backup_files
            
        except Exception:
            return None
    
    def restore_model(self, backup_files):
        """恢复模型"""
        try:
            import snownlp
            snownlp_dir = os.path.dirname(snownlp.__file__)
            sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
            
            for backup_file in backup_files:
                if os.path.exists(backup_file):
                    original_file = backup_file.replace('.temp_backup', '')
                    shutil.copy2(backup_file, original_file)
                    os.remove(backup_file)  # 删除临时备份
            
        except Exception as e:
            self.log(f"模型恢复失败: {e}")

    def manual_replace_model(self):
        """手动替换模型"""
        # 在新线程中执行
        replace_thread = threading.Thread(target=self.manual_replace_worker)
        replace_thread.daemon = True
        replace_thread.start()
    
    def manual_replace_worker(self):
        """手动替换工作线程"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "🔧 手动模型文件替换\n" + "="*50 + "\n\n")
        
        try:
            # 1. 检查可能的源文件
            possible_files = [
                'custom_sentiment.marshal.3',
                'sentiment.marshal',
                'sentiment.marshal.3',
                'custom_sentiment.model'
            ]
            
            source_file = None
            for fname in possible_files:
                if os.path.exists(fname):
                    file_size = os.path.getsize(fname)
                    self.result_text.insert(tk.END, f"找到模型文件: {fname} ({file_size:,} 字节)\n")
                    if file_size > 50000:  # 选择较大的文件
                        source_file = fname
                        break
            
            if not source_file:
                self.result_text.insert(tk.END, "❌ 未找到有效的模型文件\n")
                self.result_text.insert(tk.END, "请先完成模型训练\n")
                return
            
            self.result_text.insert(tk.END, f"✅ 选择源文件: {source_file}\n\n")
            
            # 2. 获取SnowNLP路径
            import snownlp
            snownlp_dir = os.path.dirname(snownlp.__file__)
            sentiment_dir = os.path.join(snownlp_dir, 'sentiment')
            
            self.result_text.insert(tk.END, f"SnowNLP目录: {snownlp_dir}\n")
            self.result_text.insert(tk.END, f"Sentiment目录: {sentiment_dir}\n\n")
            
            # 3. 查找目标文件
            target_files = []
            for fname in ['sentiment.marshal', 'sentiment.marshal.3']:
                fpath = os.path.join(sentiment_dir, fname)
                if os.path.exists(fpath):
                    target_files.append(fpath)
                    self.result_text.insert(tk.END, f"找到目标文件: {fname}\n")
            
            if not target_files:
                self.result_text.insert(tk.END, "❌ 未找到目标模型文件\n")
                return
            
            # 4. 备份原文件
            self.result_text.insert(tk.END, "\n开始备份原文件...\n")
            for target_file in target_files:
                backup_file = target_file + '.backup_manual'
                if not os.path.exists(backup_file):
                    shutil.copy2(target_file, backup_file)
                    fname = os.path.basename(backup_file)
                    self.result_text.insert(tk.END, f"✅ 备份完成: {fname}\n")
                else:
                    fname = os.path.basename(backup_file)
                    self.result_text.insert(tk.END, f"备份已存在: {fname}\n")
            
            # 5. 执行替换
            self.result_text.insert(tk.END, "\n开始替换模型文件...\n")
            success_count = 0
            
            for target_file in target_files:
                try:
                    shutil.copy2(source_file, target_file)
                    new_size = os.path.getsize(target_file)
                    fname = os.path.basename(target_file)
                    self.result_text.insert(tk.END, f"✅ 替换成功: {fname} ({new_size:,} 字节)\n")
                    success_count += 1
                except Exception as e:
                    fname = os.path.basename(target_file)
                    self.result_text.insert(tk.END, f"❌ 替换失败 {fname}: {e}\n")
            
            # 6. 结果报告
            if success_count > 0:
                self.result_text.insert(tk.END, f"\n🎉 成功替换 {success_count} 个模型文件！\n")
                self.result_text.insert(tk.END, "\n重要提示:\n")
                self.result_text.insert(tk.END, "1. 模型文件已成功替换\n")
                self.result_text.insert(tk.END, "2. 建议重启程序以确保使用新模型\n")
                self.result_text.insert(tk.END, "3. 可以使用测试功能验证新模型效果\n")
                
                messagebox.showinfo("成功", "模型替换成功！\n建议重启程序使用新模型。")
            else:
                self.result_text.insert(tk.END, "\n❌ 模型替换失败\n")
                self.result_text.insert(tk.END, "可能需要管理员权限或检查文件权限\n")
                
                messagebox.showerror("失败", "模型替换失败！\n请查看详细信息。")
                
        except Exception as e:
            self.result_text.insert(tk.END, f"\n❌ 操作失败: {e}\n")
            import traceback
            self.result_text.insert(tk.END, f"详细错误:\n{traceback.format_exc()}\n")
    
    def show_model_manager(self):
        """显示模型管理器"""
        # 创建新窗口
        manager_window = tk.Toplevel(self.root)
        manager_window.title("📦 模型管理器")
        manager_window.geometry("1000x600")
        manager_window.transient(self.root)
        
        # 创建主框架
        main_frame = ttk.Frame(manager_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="📦 训练模型管理器", font=("", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        # 工具栏
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="🔄 刷新列表", command=lambda: self.refresh_model_list(tree)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="✏️ 重命名", command=lambda: self.rename_model(tree)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="📝 添加备注", command=lambda: self.edit_model_notes(tree)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="🧪 测试模型", command=lambda: self.test_selected_model(tree)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="🗑️ 删除模型", command=lambda: self.delete_model(tree)).pack(side=tk.LEFT, padx=(0, 10))
        
        # 模型列表
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Treeview
        columns = ("name", "created_time", "train_files", "samples", "accuracy", "strategy", "size", "notes")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # 设置列标题
        tree.heading("name", text="模型名称")
        tree.heading("created_time", text="创建时间")
        tree.heading("train_files", text="训练数据")
        tree.heading("samples", text="样本数")
        tree.heading("accuracy", text="测试准确率")
        tree.heading("strategy", text="中性策略")
        tree.heading("size", text="文件大小")
        tree.heading("notes", text="备注")
        
        # 设置列宽
        tree.column("name", width=150)
        tree.column("created_time", width=120)
        tree.column("train_files", width=150)
        tree.column("samples", width=80)
        tree.column("accuracy", width=80)
        tree.column("strategy", width=80)
        tree.column("size", width=80)
        tree.column("notes", width=200)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 加载模型列表
        self.refresh_model_list(tree)
        
        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        models = self.model_manager.get_model_list()
        status_text = f"共管理 {len(models)} 个训练模型"
        ttk.Label(status_frame, text=status_text).pack(side=tk.LEFT)
        
        ttk.Button(status_frame, text="关闭", command=manager_window.destroy).pack(side=tk.RIGHT)
    
    def refresh_model_list(self, tree):
        """刷新模型列表"""
        # 清空现有项目
        for item in tree.get_children():
            tree.delete(item)
        
        # 获取模型列表
        models = self.model_manager.get_model_list()
        
        # 按创建时间倒序排列
        sorted_models = sorted(models.items(), 
                             key=lambda x: x[1].get('created_time', ''), 
                             reverse=True)
        
        for model_id, info in sorted_models:
            # 格式化显示信息
            name = info.get('name', '未命名模型')
            created_time = info.get('created_time', '')
            if created_time:
                try:
                    dt = datetime.fromisoformat(created_time)
                    created_time = dt.strftime('%m-%d %H:%M')
                except:
                    created_time = created_time[:16]
            
            train_files = ', '.join(info.get('train_files', []))
            if len(train_files) > 30:
                train_files = train_files[:30] + "..."
            
            samples = info.get('train_samples', 0)
            accuracy = info.get('test_accuracy', 0)
            accuracy_str = f"{accuracy:.1%}" if accuracy > 0 else "-"
            
            strategy = info.get('neutral_strategy', '')
            strategy_map = {
                'balance': '平衡',
                'random': '随机',
                'positive': '正面',
                'negative': '负面',
                'split': '分割',
                'exclude': '排除'
            }
            strategy = strategy_map.get(strategy, strategy)
            
            file_size = info.get('file_size', 0)
            size_str = f"{file_size//1024}KB" if file_size > 0 else "-"
            
            notes = info.get('notes', '')
            if len(notes) > 50:
                notes = notes[:50] + "..."
            
            # 插入到树形控件
            tree.insert("", tk.END, iid=model_id, values=(
                name, created_time, train_files, samples, 
                accuracy_str, strategy, size_str, notes
            ))
    
    def rename_model(self, tree):
        """重命名模型"""
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选择一个模型")
            return
        
        model_id = selected[0]
        models = self.model_manager.get_model_list()
        current_name = models[model_id]['name']
        
        # 弹出输入对话框
        new_name = tk.simpledialog.askstring("重命名模型", 
                                            f"当前名称: {current_name}\n\n请输入新名称:",
                                            initialvalue=current_name)
        if new_name and new_name.strip():
            self.model_manager.update_model(model_id, {'name': new_name.strip()})
            self.refresh_model_list(tree)
            messagebox.showinfo("成功", "模型重命名成功")
    
    def edit_model_notes(self, tree):
        """编辑模型备注"""
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选择一个模型")
            return
        
        model_id = selected[0]
        models = self.model_manager.get_model_list()
        current_notes = models[model_id].get('notes', '')
        
        # 创建备注编辑窗口
        notes_window = tk.Toplevel(self.root)
        notes_window.title("编辑模型备注")
        notes_window.geometry("500x300")
        notes_window.transient(self.root)
        
        frame = ttk.Frame(notes_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="模型备注:").pack(anchor=tk.W)
        
        notes_text = tk.Text(frame, height=10, width=60)
        notes_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        notes_text.insert(1.0, current_notes)
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X)
        
        def save_notes():
            new_notes = notes_text.get(1.0, tk.END).strip()
            self.model_manager.update_model(model_id, {'notes': new_notes})
            self.refresh_model_list(tree)
            notes_window.destroy()
            messagebox.showinfo("成功", "备注保存成功")
        
        ttk.Button(button_frame, text="保存", command=save_notes).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="取消", command=notes_window.destroy).pack(side=tk.RIGHT)
    
    def test_selected_model(self, tree):
        """测试选中的模型"""
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选择一个模型")
            return
        
        model_id = selected[0]
        models = self.model_manager.get_model_list()
        model_path = models[model_id]['path']
        
        # 临时替换模型并运行测试
        if os.path.exists(model_path):
            # 创建测试线程
            test_thread = threading.Thread(target=self.test_model_worker, args=(model_path, models[model_id]['name']))
            test_thread.daemon = True
            test_thread.start()
        else:
            messagebox.showerror("错误", "模型文件不存在")
    
    def test_model_worker(self, model_path, model_name):
        """测试模型的工作线程"""
        self.update_status_guide("model_testing", f"正在测试模型: {model_name}")
        
        # 备份当前模型
        original_model_backup = self.backup_current_model()
        
        try:
            # 临时替换模型
            if self.temp_replace_model(model_path):
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"🧪 测试模型: {model_name}\n" + "="*50 + "\n\n")
                
                # 运行基础测试
                accuracy = self.run_basic_test()
                
                self.result_text.insert(tk.END, f"\n📊 模型 '{model_name}' 测试结果:\n")
                self.result_text.insert(tk.END, f"准确率: {accuracy:.2%}\n")
                
                if accuracy >= 0.8:
                    self.result_text.insert(tk.END, "🎉 该模型表现优秀！\n")
                elif accuracy >= 0.6:
                    self.result_text.insert(tk.END, "👍 该模型表现良好！\n")
                else:
                    self.result_text.insert(tk.END, "😐 该模型需要改进\n")
            else:
                self.result_text.insert(tk.END, "❌ 模型加载失败\n")
        finally:
            # 恢复原始模型
            if original_model_backup:
                self.restore_model(original_model_backup)
            self.update_status_guide("ready")
    
    def delete_model(self, tree):
        """删除模型"""
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选择一个模型")
            return
        
        model_id = selected[0]
        models = self.model_manager.get_model_list()
        model_name = models[model_id]['name']
        
        # 确认删除
        if messagebox.askyesno("确认删除", 
                              f"确定要删除模型 '{model_name}' 吗？\n\n"
                              "注意: 这将删除模型文件和所有相关信息，"
                              "此操作不可恢复！"):
            
            model_path = self.model_manager.delete_model(model_id)
            if model_path and os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    messagebox.showinfo("成功", f"模型 '{model_name}' 已删除")
                except Exception as e:
                    messagebox.showerror("错误", f"删除模型文件失败: {e}")
            
            self.refresh_model_list(tree)
    
    def compare_models_on_dataset(self):
        """基于统一数据集对比多个模型性能"""
        # 首先选择测试数据集
        data_files = filedialog.askopenfilenames(
            title="选择测试数据集进行模型对比",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if not data_files:
            return
        
        models = self.model_manager.get_model_list()
        if len(models) < 2:
            messagebox.showwarning("提示", "至少需要2个已训练的模型才能进行对比")
            return
        
        # 创建模型选择窗口
        selection_window = tk.Toplevel(self.root)
        selection_window.title("选择要对比的模型")
        selection_window.geometry("600x400")
        selection_window.transient(self.root)
        
        frame = ttk.Frame(selection_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="请选择要对比的模型（至少选择2个）:", font=("", 12, "bold")).pack(pady=(0, 10))
        
        # 创建模型复选框列表
        model_vars = {}
        listbox_frame = ttk.Frame(frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 添加滚动条
        scrollbar_select = ttk.Scrollbar(listbox_frame)
        scrollbar_select.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas = tk.Canvas(listbox_frame, yscrollcommand=scrollbar_select.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_select.config(command=canvas.yview)
        
        checkbox_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=checkbox_frame, anchor=tk.NW)
        
        # 按创建时间倒序排列
        sorted_models = sorted(models.items(), 
                             key=lambda x: x[1].get('created_time', ''), 
                             reverse=True)
        
        for model_id, info in sorted_models:
            var = tk.BooleanVar()
            model_vars[model_id] = var
            
            # 格式化显示信息
            name = info.get('name', '未命名模型')
            created_time = info.get('created_time', '')
            if created_time:
                try:
                    dt = datetime.fromisoformat(created_time)
                    created_time = dt.strftime('%m-%d %H:%M')
                except:
                    created_time = created_time[:16]
            
            accuracy = info.get('test_accuracy', 0)
            accuracy_str = f"准确率: {accuracy:.1%}" if accuracy > 0 else "准确率: 未测试"
            
            text = f"{name} ({created_time}) - {accuracy_str}"
            ttk.Checkbutton(checkbox_frame, text=text, variable=var).pack(anchor=tk.W, pady=2)
        
        # 更新画布滚动区域
        checkbox_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # 按钮框架
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X)
        
        def start_comparison():
            selected_models = [model_id for model_id, var in model_vars.items() if var.get()]
            if len(selected_models) < 2:
                messagebox.showwarning("提示", "请至少选择2个模型进行对比")
                return
            
            selection_window.destroy()
            # 在新线程中执行对比
            compare_thread = threading.Thread(target=self.compare_models_worker, 
                                            args=(selected_models, data_files))
            compare_thread.daemon = True
            compare_thread.start()
        
        ttk.Button(button_frame, text="开始对比", command=start_comparison).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="取消", command=selection_window.destroy).pack(side=tk.RIGHT)
    
    def compare_models_worker(self, selected_model_ids, data_files):
        """模型对比工作线程"""
        self.update_status_guide("comparing", "正在进行模型性能对比...")
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "📊 模型性能对比测试\n" + "="*60 + "\n\n")
        
        # 加载测试数据
        self.result_text.insert(tk.END, "📂 加载测试数据集...\n")
        test_texts, test_labels = self.load_data(list(data_files), "对比测试")
        
        if not test_texts:
            self.result_text.insert(tk.END, "❌ 测试数据加载失败\n")
            return
        
        self.result_text.insert(tk.END, f"✅ 成功加载 {len(test_texts)} 个测试样本\n\n")
        
        # 备份原始模型
        original_model_backup = self.backup_current_model()
        
        results = []
        models = self.model_manager.get_model_list()
        
        try:
            for i, model_id in enumerate(selected_model_ids, 1):
                model_info = models[model_id]
                model_path = model_info['path']
                model_name = model_info['name']
                
                self.result_text.insert(tk.END, f"🧪 测试模型 {i}/{len(selected_model_ids)}: {model_name}\n")
                self.result_text.insert(tk.END, "-" * 50 + "\n")
                
                if self.temp_replace_model(model_path):
                    # 评估模型
                    correct = 0
                    total_processed = 0
                    
                    for j, (text, true_label) in enumerate(zip(test_texts, test_labels)):
                        try:
                            s = SnowNLP(text)
                            score = s.sentiments
                            pred_label = 1 if score > 0.5 else 0
                            
                            if pred_label == true_label:
                                correct += 1
                            total_processed += 1
                            
                            # 显示进度
                            if (j + 1) % 500 == 0:
                                progress = (j + 1) / len(test_texts) * 100
                                current_acc = correct / total_processed if total_processed > 0 else 0
                                self.result_text.insert(tk.END, f"  进度: {progress:.1f}% - 当前准确率: {current_acc:.2%}\n")
                                self.root.update()
                        except:
                            continue
                    
                    accuracy = correct / total_processed if total_processed > 0 else 0
                    
                    results.append({
                        'id': model_id,
                        'name': model_name,
                        'accuracy': accuracy,
                        'correct': correct,
                        'total': total_processed,
                        'created_time': model_info.get('created_time', ''),
                        'train_files': model_info.get('train_files', [])
                    })
                    
                    self.result_text.insert(tk.END, f"✅ 准确率: {accuracy:.2%} ({correct}/{total_processed})\n\n")
                    
                    # 更新模型记录中的测试准确率
                    self.model_manager.update_model(model_id, {'test_accuracy': accuracy})
                else:
                    self.result_text.insert(tk.END, "❌ 模型加载失败\n\n")
        finally:
            # 恢复原始模型
            if original_model_backup:
                self.restore_model(original_model_backup)
        
        # 显示对比结果
        self.result_text.insert(tk.END, "🏆 对比结果汇总\n" + "="*60 + "\n")
        
        # 按准确率排序
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        for i, result in enumerate(results, 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            
            created_time = result['created_time']
            if created_time:
                try:
                    dt = datetime.fromisoformat(created_time)
                    created_time = dt.strftime('%m-%d %H:%M')
                except:
                    created_time = created_time[:16]
            
            train_files = ', '.join(result['train_files'])
            
            self.result_text.insert(tk.END, f"{rank_emoji} {result['name']}\n")
            self.result_text.insert(tk.END, f"   准确率: {result['accuracy']:.2%}\n")
            self.result_text.insert(tk.END, f"   训练时间: {created_time}\n")
            self.result_text.insert(tk.END, f"   训练数据: {train_files}\n\n")
        
        if results:
            best_model = results[0]
            self.result_text.insert(tk.END, f"🎯 推荐使用: {best_model['name']}\n")
            self.result_text.insert(tk.END, f"   最佳准确率: {best_model['accuracy']:.2%}\n")
        
        self.update_status_guide("ready")
    
    def export_model(self):
        """导出模型"""
        models = self.model_manager.get_model_list()
        if not models:
            messagebox.showwarning("提示", "没有可导出的模型")
            return
        
        # 选择要导出的模型
        model_names = [f"{info['name']} ({info.get('created_time', '')[:16]})" 
                      for info in models.values()]
        model_ids = list(models.keys())
        
        selection = tk.simpledialog.askstring("选择模型", 
            f"请输入要导出的模型序号 (1-{len(models)}):\n\n" + 
            "\n".join([f"{i+1}. {name}" for i, name in enumerate(model_names)]))
        
        if not selection or not selection.isdigit():
            return
        
        try:
            index = int(selection) - 1
            if 0 <= index < len(model_ids):
                model_id = model_ids[index]
                model_info = models[model_id]
                model_path = model_info['path']
                
                # 选择导出位置
                export_path = filedialog.asksaveasfilename(
                    title="选择导出位置",
                    defaultextension=".marshal.3",
                    filetypes=[("Marshal文件", "*.marshal.3"), ("所有文件", "*.*")],
                    initialvalue=f"{model_info['name']}.marshal.3"
                )
                
                if export_path:
                    try:
                        shutil.copy2(model_path, export_path)
                        
                        # 同时导出模型信息
                        info_path = export_path + ".info.json"
                        with open(info_path, 'w', encoding='utf-8') as f:
                            json.dump(model_info, f, ensure_ascii=False, indent=2)
                        
                        messagebox.showinfo("导出成功", 
                            f"模型已导出到:\n{export_path}\n\n"
                            f"模型信息已导出到:\n{info_path}")
                    except Exception as e:
                        messagebox.showerror("导出失败", f"导出模型失败: {e}")
            else:
                messagebox.showerror("错误", "无效的序号")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置样式
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")
    
    # 创建应用
    app = SnowNLPTrainerGUI(root)
    
    # 启动界面
    root.mainloop()

if __name__ == "__main__":
    main() 