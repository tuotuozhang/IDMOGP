import numpy as np
import matplotlib.pyplot as plt

# 创建 x 值范围，确保 x 大于 0
# x = np.linspace(0.01, 1, 8)  # 从0.01到5之间生成400个点
x = [0.001, 0.001, 0.001, 0.997]
# 计算对应的 y 值
y = x * np.log(x)
f = -sum(y)
print(f)
# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='y = x * log(x)', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function y = x * log(x)')
plt.grid(True)
plt.legend()
plt.show()
