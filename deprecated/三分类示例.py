# -*- coding: utf-8 -*-
"""
三分类情感分析使用示例
演示如何使用三分类方法进行情感分析
"""

from snownlp import SnowNLP

class ThreeClassSentiment:
    """三分类情感分析器"""
    
    def __init__(self, negative_threshold=0.4, positive_threshold=0.6):
        self.negative_threshold = negative_threshold
        self.positive_threshold = positive_threshold
        
    def classify(self, text):
        """三分类预测"""
        s = SnowNLP(text)
        score = s.sentiments
        
        if score < self.negative_threshold:
            return score, 0, "负面"
        elif score > self.positive_threshold:
            return score, 2, "正面"
        else:
            return score, 1, "中性"

def demo_basic_usage():
    """基本使用示例"""
    print("=" * 50)
    print("基本使用示例")
    print("=" * 50)
    
    # 创建分类器
    classifier = ThreeClassSentiment()
    
    # 测试文本
    test_texts = [
        "这个电影真的太棒了！",
        "完全是浪费时间的垃圾",
        "还可以吧，没什么特别的",
        "演员表演很精彩",
        "剧情很无聊",
        "一般般的作品",
        "强烈推荐大家去看",
        "不推荐，很失望",
        "普通的电影"
    ]
    
    print(f"分类阈值: 负面 < {classifier.negative_threshold}, "
          f"中性 {classifier.negative_threshold}-{classifier.positive_threshold}, "
          f"正面 > {classifier.positive_threshold}\n")
    
    for i, text in enumerate(test_texts, 1):
        score, label, label_name = classifier.classify(text)
        print(f"{i:2d}. {text}")
        print(f"    得分: {score:.4f} → {label_name}")
        print()

def demo_custom_thresholds():
    """自定义阈值示例"""
    print("=" * 50)
    print("自定义阈值示例")
    print("=" * 50)
    
    text = "这个产品还可以"
    
    # 不同阈值设置
    thresholds = [
        (0.3, 0.7, "保守策略"),
        (0.4, 0.6, "平衡策略"),
        (0.45, 0.55, "激进策略")
    ]
    
    print(f"测试文本: {text}\n")
    
    for neg_thresh, pos_thresh, strategy_name in thresholds:
        classifier = ThreeClassSentiment(neg_thresh, pos_thresh)
        score, label, label_name = classifier.classify(text)
        
        print(f"{strategy_name} (阈值: {neg_thresh}-{pos_thresh}):")
        print(f"  得分: {score:.4f} → {label_name}")
        print()

def demo_batch_analysis():
    """批量分析示例"""
    print("=" * 50)
    print("批量分析示例")
    print("=" * 50)
    
    # 产品评论数据
    product_reviews = [
        "质量很好，物超所值",
        "包装精美，服务周到",
        "价格偏高，性价比一般",
        "产品有瑕疵，不满意",
        "配送很快，商品不错",
        "还行吧，没有期待那么好",
        "超出预期，非常满意",
        "质量太差，要求退货",
        "中规中矩的产品"
    ]
    
    classifier = ThreeClassSentiment()
    
    # 统计各类别数量
    negative_count = 0
    neutral_count = 0
    positive_count = 0
    
    print("产品评论情感分析结果:\n")
    
    for i, review in enumerate(product_reviews, 1):
        score, label, label_name = classifier.classify(review)
        
        print(f"{i:2d}. {review}")
        print(f"    {label_name} ({score:.4f})")
        print()
        
        if label == 0:
            negative_count += 1
        elif label == 1:
            neutral_count += 1
        else:
            positive_count += 1
    
    # 统计结果
    total = len(product_reviews)
    print("=" * 30)
    print("统计结果:")
    print("=" * 30)
    print(f"正面评论: {positive_count} ({positive_count/total:.1%})")
    print(f"中性评论: {neutral_count} ({neutral_count/total:.1%})")
    print(f"负面评论: {negative_count} ({negative_count/total:.1%})")

def demo_score_distribution():
    """得分分布示例"""
    print("=" * 50)
    print("得分分布分析示例")
    print("=" * 50)
    
    texts = [
        "非常棒！",        # 高正面
        "很好",           # 正面
        "不错",           # 偏正面
        "还可以",         # 中性
        "一般",           # 中性
        "不太好",         # 偏负面
        "很差",           # 负面
        "太糟糕了！"      # 高负面
    ]
    
    classifier = ThreeClassSentiment()
    
    print("得分分布分析:\n")
    
    for text in texts:
        score, label, label_name = classifier.classify(text)
        
        # 生成可视化的得分条
        bar_length = 20
        bar_pos = int(score * bar_length)
        bar = ['·'] * bar_length
        bar[bar_pos] = '●'
        bar_str = ''.join(bar)
        
        print(f"{text:8} | {score:.3f} | {bar_str} | {label_name}")
    
    print(f"\n图例: 0.0 {'·'*20} 1.0")
    print(f"      负面区间 | 中性区间 | 正面区间")
    print(f"      0.0-0.4  | 0.4-0.6  | 0.6-1.0")

if __name__ == "__main__":
    print("三分类情感分析演示")
    print("使用 SnowNLP + 阈值设定实现三分类")
    print()
    
    # 运行各种示例
    demo_basic_usage()
    demo_custom_thresholds()
    demo_batch_analysis()
    demo_score_distribution()
    
    print("=" * 50)
    print("演示完成！")
    print("您可以根据实际需求调整阈值参数")
    print("建议根据您的数据特点选择合适的策略") 