from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt


def restore_image(cb, cluster, shape):
    row, col, dummy = shape
    image = np.empty((row, col, dummy))
    for r in range(row):
        for c in range(col):
            image[r, c] = cb[cluster[r * col + c]]
    return image


if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    im = Image.open('Tiger_Woods_0023.jpg')
    image = np.array(im).astype(np.float) / 255
    image = image[:, :, :3]
    print(image.shape)

    plt.figure()
    plt.subplot(141)
    plt.axis('off')
    plt.title(u'原始图片')
    plt.imshow(image)
    # 可以使用plt.savefig('原始图片.png') ，保存原始图片并对比

    for i,n_clusters  in enumerate([2,6,30]):
        # 聚类数2,6,30
        image_v = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=n_clusters , init='k-means++')

        N = image_v.shape[0]  # 图像像素总数
        # 选择样本，计算聚类中心
        idx = np.random.randint(0, N, size=int(N * 0.7))
        image_sample = image_v[idx]
        kmeans.fit(image_sample)
        result = kmeans.predict(image_v)  # 聚类结果
        print('聚类结果:\n', result)
        print(result.shape)
        print('聚类中心%d个\n' % n_clusters, kmeans.cluster_centers_)


        plt.subplot(1,4,i+2)
        vq_image = restore_image(kmeans.cluster_centers_, result, image.shape)
        plt.axis('off')
        plt.title(u'聚类个数:%d' % n_clusters )
        plt.imshow(vq_image)
        # 可以使用plt.savefig('矢量化图片.png')，保存处理后的图片并对比

    plt.show()