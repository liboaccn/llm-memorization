import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plotting(variance1, variance2, mean1, mean2, random_mean1, random_variance1, random_mean2, random_variance2,
             model_name_or_path):
    # 计算标准差
    std_dev1 = np.sqrt(variance1)
    std_dev2 = np.sqrt(variance2)
    random_std_dev1 = np.sqrt(random_variance1)
    random_std_dev2 = np.sqrt(random_variance2)

    # 生成x的值，考虑到所有四个分布的范围
    x = np.linspace(min(mean1 - 3 * std_dev1, mean2 - 3 * std_dev2, random_mean1 - 3 * random_std_dev1,
                        random_mean2 - 3 * random_std_dev2),
                    max(mean1 + 3 * std_dev1, mean2 + 3 * std_dev2, random_mean1 + 3 * random_std_dev1,
                        random_mean2 + 3 * random_std_dev2), 1000)

    # 计算两个原始正态分布的PDF值
    y1 = norm.pdf(x, mean1, std_dev1)
    y2 = norm.pdf(x, mean2, std_dev2)

    # 计算两个新的正态分布的PDF值
    y3 = norm.pdf(x, random_mean1, random_std_dev1)
    y4 = norm.pdf(x, random_mean2, random_std_dev2)

    # 设置图形大小为8x6英寸
    plt.figure(figsize=(8, 6))

    # 绘制四个正态分布
    plt.fill_between(x, y1, color="#dde1ad", alpha=0.5)
    plt.plot(x, y1, color="#fab0c4", linestyle='--',
             label="Memorized, $\mu={:.4f}, \sigma^2={:.4f}$".format(mean1, variance1))
    plt.fill_between(x, y2, color="#fbd1de", alpha=0.5)
    plt.plot(x, y2, color="#fbd1de", linestyle='--',
             label="Non-memorized, $\mu={:.4f}, \sigma^2={:.4f}$".format(mean2, variance2))

    # 新增绘制
    plt.fill_between(x, y3, color="#dde1ad", alpha=0.5)
    plt.plot(x, y3, color="#a7c456",
             label="Split 1, $\mu={:.4f}, \sigma^2={:.4f}$".format(random_mean1, random_variance1))
    plt.fill_between(x, y4, color="#6a9a4a", alpha=0.5)
    plt.plot(x, y4, color="#6a9a4a",
             label="Split 2, $\mu={:.4f}, \sigma^2={:.4f}$".format(random_mean2, random_variance2))

    # 添加标题和标签, 图例
    plt.xlabel('Predicted Probability', fontsize=18)
    plt.xlim(-0.2, 1.0)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=14)

    # 隐藏整个y轴（包括轴线和标签）
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().spines['left'].set_visible(False)  # 隐藏y轴的竖线

    # 隐藏上方和右侧的边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.savefig('../figure/idiom_prob_rand{}.pdf'.format(model_name_or_path), dpi=300, bbox_inches='tight')
    # 显示图形
    plt.show()


def get_numbers(model):
    if model == 'llama2-13b':
        mean1, variance1 = 0.4254, 0.0168  # memorized
        mean2, variance2 = 0.2880, 0.0268
        random_mean1, random_variance1 = 0.4000, 0.0196
        random_mean2, random_variance2 = 0.4105, 0.0222
    else:
        raise ValueError('Model not found')
    return variance1, variance2, random_variance1, random_variance2, \
           mean1, mean2, random_mean1, random_mean2


if __name__ == '__main__':
    from load_LLMs import MODELS
    random_splits = ['llama2-13b']
    for model_name_or_path in random_splits:
        variance1, variance2, rand_var1, rand_var2, \
        mean1, mean2, rand_mean1, rand_mean2 = get_numbers(model_name_or_path)
        plotting(variance1, variance2, mean1, mean2, rand_mean1, rand_var1, rand_mean2, rand_var2, model_name_or_path)

