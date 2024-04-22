import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plotting(variance1, variance2, mean1, mean2, model_name_or_path):
    # 计算标准差
    std_dev1 = np.sqrt(variance1)
    std_dev2 = np.sqrt(variance2)

    # 生成x的值，考虑到两个分布的范围
    x = np.linspace(min(mean1 - 3*std_dev1, mean2 - 3*std_dev2), max(mean1 + 3*std_dev1, mean2 + 3*std_dev2), 1000)

    # 计算两个正态分布的PDF值
    y1 = norm.pdf(x, mean1, std_dev1)
    y2 = norm.pdf(x, mean2, std_dev2)

    # 设置图形大小为8x6英寸
    plt.figure(figsize=(8, 6))

    # 绘制两个正态分布
    plt.fill_between(x, y1, color="#ECCE61", alpha=0.5)
    plt.plot(x, y1, color="#ECCE61", label="Memorized, $\mu={}, \sigma^2=0.0156$".format(mean1, variance1))

    plt.fill_between(x, y2, color="#B8CAD6", alpha=0.5)
    plt.plot(x, y2, color="#B8CAD6", label="Non-memorized, $\mu={}, \sigma^2={}$".format(mean2, variance2))

    # 添加标题和标签,图例
    plt.xlabel('predicted probability', fontsize=14)
    plt.xlim(-0.2, 1.0)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=10)

    # 隐藏整个y轴（包括轴线和标签）
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().spines['left'].set_visible(False)  # 隐藏y轴的竖线

    # 隐藏上方和右侧的边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.savefig('../figure/poetry_prob_{}.pdf'.format(model_name_or_path.split('/')[-1]), dpi=300, bbox_inches='tight')
    # 显示图形
    plt.show()


def get_numbers(model):
    if model == '/home/incoming/LLM/qwen1_5/qwen1_5-7b':
        mean1, variance1 = 0.8815310775080694, 0.009514372188555251  # memorized
        mean2, variance2 = 0.5706145638865455, 0.041165736641014736  # non-memorized
    elif model == '/home/incoming/LLM/qwen1_5/qwen1_5-14b':
        mean1, variance1 = 0.7736181988277369, 0.07123299873708827  # ?
        mean2, variance2 = 0.30352748847670025, 0.051063538288446536
    else:
        raise ValueError('Model not found')
    return variance1, variance2, mean1, mean2

if __name__ == '__main__':
    from load_LLMs import MODELS
    # random_splits = ['llama2-7b-random', 'llama2-13b-random']
    for model_name_or_path in MODELS:
        variance1, variance2, mean1, mean2 = get_numbers(model_name_or_path)
        plotting(variance1, variance2, mean1, mean2, model_name_or_path)

