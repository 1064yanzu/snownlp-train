# -*- coding: utf-8 -*-
"""
SnowNLP情感分析训练工具 - Web界面版本
基于Flask的Web界面，适用于远程服务器和云环境
"""

import os
import sys
import time
import json
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_file
import threading
import pandas as pd
from glob import glob

from app_logger import get_logger, runtime_summary

WEB_LOGGER = get_logger("web")
try:
    WEB_LOGGER.info("web_start runtime=%s", runtime_summary())
except Exception:
    pass

# 检查并安装Flask
try:
    import flask
except ImportError:
    print("正在安装Flask...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    import flask

app = Flask(__name__)
app.secret_key = "snownlp_web_tool_2024"

# 全局变量
training_status = {
    'running': False,
    'progress': 0,
    'message': '准备就绪',
    'log': []
}

def log_message(message):
    """添加日志"""
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    log_entry = f"{timestamp} {message}"
    training_status['log'].append(log_entry)
    print(log_entry)
    try:
        WEB_LOGGER.info("%s", str(message))
    except Exception:
        pass

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SnowNLP情感分析训练工具 - Web版</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #4CAF50, #2196F3);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .section {
            background: #f8f9fa;
            margin: 20px 0;
            padding: 25px;
            border-radius: 15px;
            border-left: 5px solid #4CAF50;
        }
        
        .section h2 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .status-card {
            background: #e8f5e8;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        
        .status-running {
            background: #fff3cd;
            border-color: #ffc107;
        }
        
        .status-error {
            background: #f8d7da;
            border-color: #dc3545;
        }
        
        .btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
            transition: all 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #2196F3, #1976D2);
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #f44336, #d32f2f);
        }
        
        .progress-bar {
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .progress-fill {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            height: 30px;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .log-container {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            max-height: 400px;
            overflow-y: auto;
            margin: 20px 0;
        }
        
        .form-group {
            margin: 15px 0;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .form-control {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        .form-control:focus {
            outline: none;
            border-color: #4CAF50;
        }
        
        .file-info {
            background: #e3f2fd;
            border: 1px solid #2196F3;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .test-result {
            background: #f0f0f0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            font-family: monospace;
        }
        
        .emoji {
            font-size: 1.2em;
            margin-right: 5px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 10px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .content {
                padding: 20px;
            }
            
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SnowNLP情感分析训练工具</h1>
            <p>Web版本 - 适用于Linux云环境和远程服务器</p>
        </div>
        
        <div class="content">
            <!-- 状态显示 -->
            <div class="section">
                <h2><span class="emoji">📊</span>系统状态</h2>
                <div id="status-card" class="status-card">
                    <h3 id="status-message">准备就绪</h3>
                    <div class="progress-bar">
                        <div id="progress-fill" class="progress-fill" style="width: 0%">0%</div>
                    </div>
                    <p>当前时间: <span id="current-time"></span></p>
                </div>
            </div>
            
            <div class="grid">
                <!-- 数据文件信息 -->
                <div class="section">
                    <h2><span class="emoji">📁</span>数据文件</h2>
                    <button class="btn btn-secondary" onclick="checkDataFiles()">
                        <span class="emoji">🔍</span>检查数据文件
                    </button>
                    <div id="data-files-info"></div>
                </div>
                
                <!-- 训练控制 -->
                <div class="section">
                    <h2><span class="emoji">🎯</span>模型训练</h2>
                    <div class="form-group">
                        <label for="neutral-strategy">中性数据处理策略:</label>
                        <select id="neutral-strategy" class="form-control">
                            <option value="balance">自动平衡(推荐)</option>
                            <option value="split">比例分配</option>
                            <option value="exclude">排除中性</option>
                        </select>
                    </div>
                    <button id="train-btn" class="btn" onclick="startTraining()">
                        <span class="emoji">🚀</span>开始训练
                    </button>
                    <button id="stop-btn" class="btn btn-danger" onclick="stopTraining()" disabled>
                        <span class="emoji">⏹️</span>停止训练
                    </button>
                </div>
            </div>
            
            <div class="grid">
                <!-- 模型测试 -->
                <div class="section">
                    <h2><span class="emoji">🧪</span>模型测试</h2>
                    <button class="btn btn-secondary" onclick="quickTest()">
                        <span class="emoji">⚡</span>快速验证
                    </button>
                    <button class="btn btn-secondary" onclick="interactiveTest()">
                        <span class="emoji">🎮</span>交互测试
                    </button>
                    <div class="form-group">
                        <label for="test-input">输入测试文本:</label>
                        <textarea id="test-input" class="form-control" rows="3" 
                                placeholder="输入要分析的文本..."></textarea>
                        <button class="btn" onclick="analyzeText()">
                            <span class="emoji">🔍</span>分析情感
                        </button>
                    </div>
                    <div id="test-results"></div>
                </div>
                
                <!-- 系统信息 -->
                <div class="section">
                    <h2><span class="emoji">💻</span>系统信息</h2>
                    <button class="btn btn-secondary" onclick="getSystemInfo()">
                        <span class="emoji">📋</span>查看系统信息
                    </button>
                    <button class="btn btn-secondary" onclick="clearLog()">
                        <span class="emoji">🧹</span>清空日志
                    </button>
                    <div id="system-info"></div>
                </div>
            </div>
            
            <!-- 日志显示 -->
            <div class="section">
                <h2><span class="emoji">📝</span>运行日志</h2>
                <div id="log-container" class="log-container">
                    <div>等待操作...</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 更新当前时间
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleString('zh-CN');
        }
        
        // 检查训练状态
        function checkStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateStatus(data);
                })
                .catch(error => console.error('Error:', error));
        }
        
        // 更新状态显示
        function updateStatus(data) {
            const statusCard = document.getElementById('status-card');
            const statusMessage = document.getElementById('status-message');
            const progressFill = document.getElementById('progress-fill');
            const trainBtn = document.getElementById('train-btn');
            const stopBtn = document.getElementById('stop-btn');
            
            statusMessage.textContent = data.message;
            progressFill.style.width = data.progress + '%';
            progressFill.textContent = data.progress + '%';
            
            if (data.running) {
                statusCard.className = 'status-card status-running';
                trainBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                statusCard.className = 'status-card';
                trainBtn.disabled = false;
                stopBtn.disabled = true;
            }
            
            // 更新日志
            const logContainer = document.getElementById('log-container');
            if (data.log && data.log.length > 0) {
                logContainer.innerHTML = data.log.slice(-20).map(log => `<div>${log}</div>`).join('');
                logContainer.scrollTop = logContainer.scrollHeight;
            }
        }
        
        // 检查数据文件
        function checkDataFiles() {
            fetch('/api/check_files')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('data-files-info');
                    let html = '';
                    
                    if (data.train_files && data.train_files.length > 0) {
                        html += '<div class="file-info"><h4>🚀 训练文件:</h4>';
                        data.train_files.forEach(file => {
                            html += `<p>${file.name} (${file.size} 字节)</p>`;
                        });
                        html += '</div>';
                    } else {
                        html += '<div class="file-info"><h4>⚠️ 未找到训练文件</h4></div>';
                    }
                    
                    if (data.test_files && data.test_files.length > 0) {
                        html += '<div class="file-info"><h4>🧪 测试文件:</h4>';
                        data.test_files.forEach(file => {
                            html += `<p>${file.name} (${file.size} 字节)</p>`;
                        });
                        html += '</div>';
                    } else {
                        html += '<div class="file-info"><h4>⚠️ 未找到测试文件</h4></div>';
                    }
                    
                    container.innerHTML = html;
                });
        }
        
        // 开始训练
        function startTraining() {
            const strategy = document.getElementById('neutral-strategy').value;
            
            fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    neutral_strategy: strategy
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('训练已开始！');
                } else {
                    alert('训练启动失败: ' + data.message);
                }
            });
        }
        
        // 停止训练
        function stopTraining() {
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                });
        }
        
        // 快速测试
        function quickTest() {
            fetch('/api/quick_test')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('test-results');
                    let html = '<div class="test-result"><h4>⚡ 快速测试结果:</h4>';
                    
                    if (data.results) {
                        data.results.forEach((result, index) => {
                            const status = result.correct ? '✅' : '❌';
                            html += `<p>${status} [${index + 1}] ${result.score.toFixed(4)} (${result.predicted}) | ${result.text}</p>`;
                        });
                        html += `<p><strong>📊 准确率: ${data.accuracy}</strong></p>`;
                    }
                    html += '</div>';
                    container.innerHTML = html;
                });
        }
        
        // 分析文本
        function analyzeText() {
            const text = document.getElementById('test-input').value.trim();
            if (!text) {
                alert('请输入要分析的文本');
                return;
            }
            
            fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({text: text})
            })
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById('test-results');
                let html = '<div class="test-result"><h4>🔍 情感分析结果:</h4>';
                
                if (data.success) {
                    const sentiment = data.score > 0.6 ? '正面 😊' : 
                                     data.score < 0.4 ? '负面 😞' : '中性 😐';
                    html += `<p><strong>文本:</strong> ${data.text}</p>`;
                    html += `<p><strong>得分:</strong> ${data.score.toFixed(4)}</p>`;
                    html += `<p><strong>情感:</strong> ${sentiment}</p>`;
                } else {
                    html += `<p>❌ 分析失败: ${data.message}</p>`;
                }
                html += '</div>';
                container.innerHTML = html;
            });
        }
        
        // 获取系统信息
        function getSystemInfo() {
            fetch('/api/system_info')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('system-info');
                    let html = '<div class="test-result"><h4>💻 系统信息:</h4>';
                    
                    for (const [key, value] of Object.entries(data)) {
                        html += `<p><strong>${key}:</strong> ${value}</p>`;
                    }
                    html += '</div>';
                    container.innerHTML = html;
                });
        }
        
        // 清空日志
        function clearLog() {
            fetch('/api/clear_log', {method: 'POST'})
                .then(() => {
                    document.getElementById('log-container').innerHTML = '<div>日志已清空</div>';
                });
        }
        
        // 初始化
        document.addEventListener('DOMContentLoaded', function() {
            updateTime();
            setInterval(updateTime, 1000);
            checkStatus();
            setInterval(checkStatus, 2000);
            checkDataFiles();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    """获取训练状态"""
    return jsonify(training_status)

@app.route('/api/check_files')
def api_check_files():
    """检查数据文件"""
    train_patterns = ['train.csv', '训练*.csv', '*train*.csv']
    test_patterns = ['test.csv', '测试*.csv', '*test*.csv']
    
    train_files = []
    for pattern in train_patterns:
        train_files.extend(glob(pattern))
    
    test_files = []
    for pattern in test_patterns:
        test_files.extend(glob(pattern))
    
    result = {
        'train_files': [{'name': f, 'size': os.path.getsize(f)} for f in train_files],
        'test_files': [{'name': f, 'size': os.path.getsize(f)} for f in test_files]
    }
    
    return jsonify(result)

@app.route('/api/train', methods=['POST'])
def api_train():
    """开始训练"""
    if training_status['running']:
        return jsonify({'success': False, 'message': '训练已在进行中'})
    
    data = request.json
    neutral_strategy = data.get('neutral_strategy', 'balance')
    
    # 在新线程中执行训练
    def training_worker():
        try:
            training_status['running'] = True
            training_status['progress'] = 0
            training_status['message'] = '开始训练...'
            training_status['log'] = []

            try:
                WEB_LOGGER.info(
                    "web_training_begin neutral_strategy=%s cwd=%s",
                    neutral_strategy,
                    os.getcwd(),
                )
            except Exception:
                pass
            
            log_message("🚀 开始模型训练流程...")
            
            # 导入必要模块
            from 命令行训练工具 import (
                find_data_files, load_data_with_progress, 
                create_sentiment_files, train_model, replace_model
            )
            
            # 查找数据文件
            log_message("📁 查找数据文件...")
            training_status['progress'] = 10
            training_status['message'] = '查找数据文件...'
            
            train_files, test_files = find_data_files()
            if not train_files:
                log_message("❌ 未找到训练数据文件")
                training_status['message'] = '未找到训练数据文件'
                return
            
            # 加载训练数据
            log_message("📂 加载训练数据...")
            training_status['progress'] = 30
            training_status['message'] = '加载训练数据...'
            
            train_texts, train_labels = load_data_with_progress(
                train_files, "训练", neutral_strategy)
            
            if not train_texts:
                log_message("❌ 训练数据加载失败")
                training_status['message'] = '训练数据加载失败'
                return
            
            # 创建语料文件
            log_message("📝 创建语料文件...")
            training_status['progress'] = 50
            training_status['message'] = '创建语料文件...'
            
            pos_path = 'temp_data/pos.txt'
            neg_path = 'temp_data/neg.txt'
            pos_count, neg_count = create_sentiment_files(
                train_texts, train_labels, pos_path, neg_path)
            
            if pos_count == 0 or neg_count == 0:
                log_message("❌ 正面或负面样本数量为0，无法训练")
                training_status['message'] = '样本数量不足'
                return
            
            # 训练模型
            log_message("🧠 开始模型训练...")
            training_status['progress'] = 70
            training_status['message'] = '模型训练中...'
            
            model_file = train_model(neg_path, pos_path)
            if not model_file:
                log_message("❌ 模型训练失败")
                training_status['message'] = '模型训练失败'
                return
            
            # 替换模型
            log_message("🔄 部署新模型...")
            training_status['progress'] = 90
            training_status['message'] = '部署新模型...'
            
            if replace_model(model_file):
                log_message("🎉 模型训练和部署完成!")
                training_status['progress'] = 100
                training_status['message'] = '训练完成！'
            else:
                log_message("❌ 模型部署失败")
                training_status['message'] = '模型部署失败'
            
            # 清理临时文件
            try:
                if os.path.exists('temp_data'):
                    import shutil
                    shutil.rmtree('temp_data')
            except:
                pass
                
        except Exception as e:
            log_message(f"❌ 训练异常: {e}")
            training_status['message'] = f'训练异常: {e}'
            try:
                WEB_LOGGER.exception("web_training_exception")
            except Exception:
                pass
        finally:
            training_status['running'] = False
    
    thread = threading.Thread(target=training_worker)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': '训练已开始'})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """停止训练"""
    training_status['running'] = False
    training_status['message'] = '训练已停止'
    log_message("⏹️ 用户停止训练")
    return jsonify({'success': True, 'message': '训练已停止'})

@app.route('/api/quick_test')
def api_quick_test():
    """快速测试"""
    try:
        from snownlp import SnowNLP
        
        test_cases = [
            ("这个产品质量非常好，强烈推荐！", "正面"),
            ("服务态度太差了，很不满意", "负面"),
            ("还可以吧，一般般", "中性"),
            ("物流速度很快，包装也不错", "正面"),
            ("价格有点贵，但质量确实好", "正面")
        ]
        
        results = []
        correct = 0
        total = 0
        
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
                
                results.append({
                    'text': text,
                    'score': score,
                    'predicted': predicted,
                    'expected': expected,
                    'correct': is_correct
                })
                
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e)
                })
        
        accuracy = f"{correct}/{total} ({correct/total:.2%})" if total > 0 else "N/A"
        
        return jsonify({
            'success': True,
            'results': results,
            'accuracy': accuracy
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """分析文本情感"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'success': False, 'message': '文本不能为空'})
        
        from snownlp import SnowNLP
        s = SnowNLP(text)
        score = s.sentiments
        
        return jsonify({
            'success': True,
            'text': text,
            'score': score
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/system_info')
def api_system_info():
    """获取系统信息"""
    import platform
    
    info = {
        'Python版本': sys.version,
        '操作系统': f"{platform.system()} {platform.release()}",
        '系统架构': platform.machine(),
        '当前目录': os.getcwd(),
        'DISPLAY环境变量': os.environ.get('DISPLAY', '未设置'),
        '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 检查依赖
    dependencies = ['pandas', 'snownlp', 'tqdm', 'numpy', 'flask']
    installed = []
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            installed.append(dep)
        except ImportError:
            missing.append(dep)
    
    info['已安装依赖'] = ', '.join(installed)
    if missing:
        info['缺失依赖'] = ', '.join(missing)
    
    return jsonify(info)

@app.route('/api/clear_log', methods=['POST'])
def api_clear_log():
    """清空日志"""
    training_status['log'] = []
    return jsonify({'success': True})

def main():
    """主函数"""
    print("=" * 60)
    print("🌐 SnowNLP情感分析训练工具 - Web版本")
    print("=" * 60)
    print("🌟 专为Linux云环境和远程服务器设计")
    print("🖥️ 通过浏览器访问图形界面")
    print("=" * 60)
    
    # 检查依赖
    try:
        import pandas, snownlp, tqdm, numpy
        print("✅ 核心依赖检查完成")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install pandas snownlp tqdm numpy")
        return
    
    # 获取主机信息
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    
    port = 5000
    
    print(f"\n🚀 启动Web服务...")
    print(f"📡 本地访问: http://127.0.0.1:{port}")
    print(f"🌍 网络访问: http://{local_ip}:{port}")
    print(f"☁️ 云服务器访问: http://YOUR_SERVER_IP:{port}")
    print(f"\n💡 使用提示:")
    print(f"• 在浏览器中打开上述链接")
    print(f"• 支持所有现代浏览器")
    print(f"• 可以通过SSH端口转发访问")
    print(f"• 按 Ctrl+C 停止服务")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Web服务已停止")
    except Exception as e:
        print(f"\n❌ Web服务启动失败: {e}")

if __name__ == "__main__":
    main() 