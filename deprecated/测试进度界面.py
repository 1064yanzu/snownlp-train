# -*- coding: utf-8 -*-
"""
测试进度界面功能
"""

import tkinter as tk
from tkinter import ttk
import time
import threading

def test_progress_interface():
    """测试进度界面"""
    root = tk.Tk()
    root.title("测试进度界面")
    root.geometry("800x600")
    
    # 创建模拟的步骤
    training_steps = [
        ("📂", "数据加载", "加载和解析训练数据文件"),
        ("📝", "语料准备", "创建正面和负面语料文件"),
        ("📊", "基线测试", "记录训练前模型性能"),
        ("🧠", "模型训练", "SnowNLP核心算法训练"),
        ("🔄", "模型部署", "替换系统模型文件"),
        ("✅", "完成验证", "验证新模型性能")
    ]
    
    # 创建界面
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # 训练步骤框架
    steps_frame = ttk.LabelFrame(main_frame, text="📋 训练步骤", padding="10")
    steps_frame.pack(fill=tk.X, pady=(0, 10))
    
    step_progress_bars = []
    step_labels = []
    
    for i, (icon, name, desc) in enumerate(training_steps):
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
        
        step_progress_bars.append(step_progress)
        step_labels.append((status_label, name_label, desc_label))
    
    # 控制按钮
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)
    
    def simulate_training():
        """模拟训练过程"""
        def worker():
            for i, (icon, name, desc) in enumerate(training_steps):
                # 开始步骤
                status_label, name_label, desc_label = step_labels[i]
                status_label.config(text="🔄", foreground="blue")
                
                # 模拟进度
                for progress in range(0, 101, 20):
                    step_progress_bars[i]['value'] = progress
                    root.update()
                    time.sleep(0.1)
                
                # 完成步骤
                status_label.config(text="✅", foreground="green")
                step_progress_bars[i]['value'] = 100
                root.update()
                time.sleep(0.5)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
    
    def reset_steps():
        """重置所有步骤"""
        for i in range(len(training_steps)):
            status_label, name_label, desc_label = step_labels[i]
            status_label.config(text="⏳", foreground="black")
            step_progress_bars[i]['value'] = 0
        root.update()
    
    ttk.Button(button_frame, text="开始测试", command=simulate_training).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(button_frame, text="重置", command=reset_steps).pack(side=tk.LEFT)
    
    # 显示说明
    info_label = ttk.Label(main_frame, 
                          text="这是进度界面测试。点击'开始测试'查看步骤进度动画。",
                          font=("", 10))
    info_label.pack(pady=10)
    
    print("✅ 测试界面启动成功")
    print("📝 如果看到步骤进度条，说明界面正常")
    print("🚀 点击'开始测试'按钮测试动画效果")
    
    root.mainloop()

if __name__ == "__main__":
    test_progress_interface() 