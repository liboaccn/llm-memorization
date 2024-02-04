from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import json


idiom_y = []
idiom_n = []

explanation_y = []
explanation_n = []

similarity_y = []
similarity_n = []
distance_y = []
distance_n = []
# Read in the data from a file
with open('edata127.jsonl', 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        idiom_v = data['idiom_v']
        explanation_v = data['explanation_v']
        match = data['match']
        if match == 'Y':
            group_idiom_y = np.array(idiom_v)
            group_explanation_y = np.array(explanation_v)

            similarity = cosine_similarity([group_idiom_y], [group_explanation_y])
            # print(similarity[0, 0])
            similarity_y.append(similarity[0, 0])
            dist = distance.euclidean(group_idiom_y, group_explanation_y)
            distance_y.append(dist)


        else:
            group_idiom_n = np.array(idiom_v)
            group_explanation_n = np.array(explanation_v)

            similarity = cosine_similarity([group_idiom_n], [group_explanation_n])
            dist = distance.euclidean(group_idiom_n, group_explanation_n)
            # print(similarity[0, 0])
            similarity_n.append(similarity[0, 0])
            distance_n.append(dist)


# print(similarity_y)
print('y_mean', np.mean(similarity_y))
print('n_mean', np.mean(similarity_n))
print('y_mean1', np.mean(distance_y))
print('n_mean1', np.mean(distance_n))
# print(similarity_n)

similarity = np.concatenate((similarity_y, similarity_n), axis=0)
print(len(similarity))
s_len = len(similarity)

dist = np.concatenate((distance_y, distance_n), axis=0)
print(len(dist))
d_len = len(dist)
 
# data = np.zeros((s_len, s_len))  
# for i in range(852):
#     for j in range(852):
#         if i == j:
#             data[i, j] = similarity[i]  







# one_dimensional_array = np.arange(852)

# 使用reshape将其转换为二维正方形数组
# 这里使用int(np.sqrt(len(one_dimensional_array)))来确定每个维度的大小
# square_array = similarity.reshape((int(np.sqrt(len(similarity))), -1))



# # 找到最接近852的平方数
# square_size = int(np.sqrt(s_len))
# while 852 % square_size != 0:
#     square_size -= 1

# # 使用reshape将其转换为二维正方形数组
# data = similarity[:square_size**2].reshape((square_size, -1))

# 找到最接近852的平方数
square_size = int(np.sqrt(d_len))
while 852 % square_size != 0:
    square_size -= 1

# 使用reshape将其转换为二维正方形数组
data = dist[:square_size**2].reshape((square_size, -1))


# 打印结果
print(data)


# exit()
# # 创建一个随机的2D数组作为热力图的数据
# # data = np.random.rand(10, 10)

# 绘制热力图
plt.imshow(data, cmap='viridis')  # viridis是一种预定义的颜色映射，你可以根据需要选择其他的cmap
plt.colorbar()  # 添加颜色条

# 添加标题和轴标签
plt.title('Heatmap Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示热力图
# plt.show()
plt.savefig('./heatmap3.png')