import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt

def restore_image(cb,cluster,shape):
    row,col,dummy = shape
    image = np.empty((row,col,dummy))
    for r in range(row):
        for c in range(col):
            image[r,c]=cb[cluster[r*col+c]]
    return image

if __name__=='__main__':
    matplotlib.rcParams['font.sans-serif']=[u'SimHei']
    matplotlib.rcParams['axes.unicode_minus']=False
    # 聚类数2,6,30
    im = Image.open('Lenna.png')
    image = np.array(im).astype(np.float) / 255
    image = image[:, :, :3]
    image_v = image.reshape((-1, 3))

    N = image_v.shape[0]
    #选择样本，计算聚类中心
    idx = np.random.randint(0,N,size=int(N*0.7))
    image_sample = image_v[idx]

    for i in range(3):
        K_set = [2,6,30]
        num_vq = K_set[i]
        # num_vq = i
        # im = Image.open('Lenna.png')
        # image = np.array(im).astype(np.float)/255
        # image = image[:,:,:3]
        # image_v =image.reshape((-1,3))
        kmeans = KMeans(n_clusters=num_vq,init='k-means++')

        # N = image_v.shape[0]
        # #选择样本，计算聚类中心
        # idx = np.random.randint(0,N,size=int(N*0.7))
        # image_sample = image_v[idx]
        kmeans.fit(image_sample)
        result = kmeans.predict(image_v)
        # print('聚类结果\n',result)
        # print('聚类中心\n',kmeans.cluster_centers_)

        plt.figure(figsize=(15,8),facecolor='w')
        plt.subplot(1,4,1)
        plt.axis('off')
        plt.title(u'原始图片',fontsize = 18)
        plt.imshow(image)

        plt.subplot(1,4,i+2)
        vq_image = restore_image(kmeans.cluster_centers_,result,image.shape)
        plt.axis('off')
        plt.title(u'聚类个数:%d' % num_vq,fontsize = 20)
        plt.imshow(vq_image)

        # plt.tight_layout(1.2)
    plt.show()

