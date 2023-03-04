import matplotlib.pyplot as plt

x = [1, 2, 3]
y1 = [1, 2, 4]
y2 = [1, 4, 8]

plt.figure()
plt.plot(x, y1, color="red", label="red")
plt.legend()  # 无此语句会不显示右下角label
plt.show()

plt.figure()
plt.plot(x, y2, color="green", label="green")
plt.legend()  # 无此语句会不显示右上角label
plt.show()