# -*- coding: utf-8 -*-
"""
修复版 GUI 工具 - 专门解决步骤显示问题
"""

import tkinter as tk
from tkinter import ttk
import time
import threading
from datetime import datetime

class FixedProgressGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("修复版进度测试")
        self.root.geometry("1000x700")
        
        # 初始化变量
        self.training_start_time = None
        self.current_step = 0
        self.training_running = False
        
        self.create_interface()
        
    def create_interface(self):
        """创建界面"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="🚀 修复版训练进度界面", 
                               font=("", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 时间信息
        self.create_time_info(main_frame)
        
        # 数据信息
        self.create_data_info(main_frame)
        
        # 训练步骤 - 核心部分
        self.create_training_steps(main_frame)
        
        # 总体进度
        self.create_overall_progress(main_frame)
        
        # 控制按钮
        self.create_controls(main_frame)
        
        # 测试按钮
        test_frame = ttk.Frame(main_frame)
        test_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(test_frame, text="🧪 测试步骤动画", 
                  command=self.test_steps).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(test_frame, text="🔄 重置步骤", 
                  command=self.reset_steps).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(test_frame, text="⏰ 测试时间更新", 
                  command=self.test_time).pack(side=tk.LEFT)
        
        print("✅ 修复版界面创建完成")
        print(f"📊 训练步骤数量: {len(self.training_steps)}")
        print(f"📋 步骤标签数量: {len(self.step_labels)}")
        print(f"📈 进度条数量: {len(self.step_progress_bars)}")
    
    def create_time_info(self, parent):
        """创建时间信息"""
        time_frame = ttk.LabelFrame(parent, text="⏱️ 时间信息", padding="10")
        time_frame.pack(fill=tk.X, pady=(0, 10))
        
        info_frame = ttk.Frame(time_frame)
        info_frame.pack(fill=tk.X)
        
        # 开始时间
        ttk.Label(info_frame, text="开始时间:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.start_time_var = tk.StringVar(value="未开始")
        ttk.Label(info_frame, textvariable=self.start_time_var).grid(row=0, column=1, sticky=tk.W)
        
        # 已用时间
        ttk.Label(info_frame, text="已用时间:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.elapsed_time_var = tk.StringVar(value="00:00:00")
        ttk.Label(info_frame, textvariable=self.elapsed_time_var, 
                 font=("", 10, "bold")).grid(row=0, column=3, sticky=tk.W)
    
    def create_data_info(self, parent):
        """创建数据信息"""
        data_frame = ttk.LabelFrame(parent, text="📊 数据信息", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.data_stats_var = tk.StringVar(value="等待加载数据...")
        ttk.Label(data_frame, textvariable=self.data_stats_var, font=("", 9)).pack(anchor=tk.W)
    
    def create_training_steps(self, parent):
        """创建训练步骤 - 关键部分"""
        steps_frame = ttk.LabelFrame(parent, text="📋 训练步骤", padding="10")
        steps_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 定义步骤
        self.training_steps = [
            ("📂", "数据加载", "加载和解析训练数据文件"),
            ("📝", "语料准备", "创建正面和负面语料文件"),
            ("📊", "基线测试", "记录训练前模型性能"),
            ("🧠", "模型训练", "SnowNLP核心算法训练"),
            ("🔄", "模型部署", "替换系统模型文件"),
            ("✅", "完成验证", "验证新模型性能")
        ]
        
        # 初始化列表
        self.step_frames = []
        self.step_progress_bars = []
        self.step_labels = []
        
        print(f"🔧 开始创建 {len(self.training_steps)} 个训练步骤...")
        
        for i, (icon, name, desc) in enumerate(self.training_steps):
            print(f"  创建步骤 {i+1}: {name}")
            
            # 创建步骤框架
            step_frame = ttk.Frame(steps_frame)
            step_frame.pack(fill=tk.X, pady=3)
            
            # 状态图标
            status_label = ttk.Label(step_frame, text="⏳", font=("", 14))
            status_label.pack(side=tk.LEFT, padx=(0, 8))
            
            # 步骤信息
            info_frame = ttk.Frame(step_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # 步骤名称
            name_label = ttk.Label(info_frame, text=f"{icon} {name}", 
                                  font=("", 11, "bold"))
            name_label.pack(anchor=tk.W)
            
            # 步骤描述
            desc_label = ttk.Label(info_frame, text=desc, font=("", 9))
            desc_label.pack(anchor=tk.W)
            
            # 进度条
            progress_frame = ttk.Frame(step_frame)
            progress_frame.pack(side=tk.RIGHT, padx=(10, 0))
            
            step_progress = ttk.Progressbar(progress_frame, length=250, mode='determinate')
            step_progress.pack()
            
            # 进度百分比
            progress_label = ttk.Label(progress_frame, text="0%", font=("", 9))
            progress_label.pack()
            
            # 保存引用
            self.step_frames.append(step_frame)
            self.step_progress_bars.append(step_progress)
            self.step_labels.append((status_label, name_label, desc_label, progress_label))
        
        print(f"✅ 成功创建所有步骤界面")
    
    def create_overall_progress(self, parent):
        """创建总体进度"""
        overall_frame = ttk.LabelFrame(parent, text="🎯 总体进度", padding="10")
        overall_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 总进度条
        ttk.Label(overall_frame, text="整体完成度:").pack(anchor=tk.W)
        self.overall_progress = ttk.Progressbar(overall_frame, mode='determinate', length=500)
        self.overall_progress.pack(fill=tk.X, pady=(5, 0))
        
        # 总进度标签
        self.overall_progress_label = tk.StringVar(value="0% - 准备开始")
        ttk.Label(overall_frame, textvariable=self.overall_progress_label, 
                 font=("", 11, "bold")).pack(pady=(5, 0))
    
    def create_controls(self, parent):
        """创建控制区域"""
        control_frame = ttk.LabelFrame(parent, text="🎮 控制台", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 状态显示
        self.status_var = tk.StringVar(value="🏁 系统就绪")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, 
                                font=("", 12, "bold"), foreground="blue")
        status_label.pack(pady=(0, 10))
        
        # 日志显示
        log_frame = ttk.Frame(control_frame)
        log_frame.pack(fill=tk.X)
        
        ttk.Label(log_frame, text="📝 操作日志:").pack(anchor=tk.W)
        self.log_text = tk.Text(log_frame, height=6, width=80, font=("Consolas", 9))
        self.log_text.pack(fill=tk.X)
        
        self.log("✅ 修复版GUI启动完成")
        self.log("🔧 步骤显示系统已就绪")
    
    def log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_text.insert(tk.END, f"{timestamp} {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_step_status(self, step, progress=0, completed=False, failed=False):
        """更新步骤状态"""
        if 0 <= step < len(self.step_labels):
            status_label, name_label, desc_label, progress_label = self.step_labels[step]
            
            try:
                if failed:
                    status_label.config(text="❌", foreground="red")
                    self.step_progress_bars[step]['value'] = 0
                    progress_label.config(text="失败", foreground="red")
                    self.log(f"❌ 步骤 {step+1} 执行失败")
                elif completed:
                    status_label.config(text="✅", foreground="green")
                    self.step_progress_bars[step]['value'] = 100
                    progress_label.config(text="100%", foreground="green")
                    self.log(f"✅ 步骤 {step+1} 执行完成")
                elif progress > 0:
                    status_label.config(text="🔄", foreground="blue")
                    self.step_progress_bars[step]['value'] = progress
                    progress_label.config(text=f"{progress}%", foreground="blue")
                else:
                    status_label.config(text="⏳", foreground="orange")
                    self.step_progress_bars[step]['value'] = 0
                    progress_label.config(text="等待", foreground="gray")
                
                # 强制更新
                self.root.update_idletasks()
                self.root.update()
                
            except Exception as e:
                self.log(f"❌ 更新步骤 {step+1} 状态失败: {e}")
        else:
            self.log(f"❌ 无效步骤索引: {step}")
    
    def test_steps(self):
        """测试步骤动画"""
        def worker():
            self.log("🧪 开始测试步骤动画...")
            self.status_var.set("🧪 正在测试步骤动画")
            
            for i in range(len(self.training_steps)):
                icon, name, desc = self.training_steps[i]
                self.log(f"🔄 测试步骤 {i+1}: {name}")
                
                # 模拟进度
                for progress in range(0, 101, 25):
                    self.update_step_status(i, progress)
                    time.sleep(0.2)
                
                # 完成步骤
                self.update_step_status(i, 100, True)
                time.sleep(0.5)
            
            self.log("✅ 步骤动画测试完成")
            self.status_var.set("✅ 测试完成")
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
    
    def reset_steps(self):
        """重置所有步骤"""
        self.log("🔄 重置所有步骤状态...")
        for i in range(len(self.training_steps)):
            self.update_step_status(i, 0)
        self.overall_progress['value'] = 0
        self.overall_progress_label.set("0% - 已重置")
        self.status_var.set("🔄 已重置")
        self.log("✅ 步骤重置完成")
    
    def test_time(self):
        """测试时间更新"""
        self.training_start_time = datetime.now()
        self.start_time_var.set(self.training_start_time.strftime("%H:%M:%S"))
        self.log("⏰ 开始时间更新测试...")
        self.status_var.set("⏰ 时间更新测试中")
        
        def update_time():
            if self.training_start_time:
                elapsed = datetime.now() - self.training_start_time
                elapsed_str = str(elapsed).split('.')[0]
                self.elapsed_time_var.set(elapsed_str)
                self.root.after(1000, update_time)
        
        update_time()

def main():
    root = tk.Tk()
    app = FixedProgressGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 