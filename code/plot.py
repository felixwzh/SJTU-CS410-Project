# coding=utf-8
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
if __name__ == '__main__':

    fig = plt.figure(frameon=False)
    fig.set_size_inches(6, 4)

    plt.grid()
    # fig = pylab.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    # plt.title('Learning curve')



    # ['MaterialType-2','Sex-2','DiseaseState-16','BioSourceType-7']
    plt.plot([100,200,400,500,1000,1200,2000,3400],
             [0.990841276,0.996099251,0.9981342994,0.99847357,0.9983040784,0.9986432052,0.9983040784,0.9984737137],
    '-ro', label="MaterialType")

    plt.plot([100,200,400,500,1000,1200,2000,3400],
             [0.8459642963, 0.8691378654, 0.8945154194, 0.8977854393, 0.9192795804, 0.9244891916, 0.930995389,0.9433647785],
            '-go', label="Sex")

    plt.plot([100,200,400,500,1000,1200,2000,3400],
             [0.9333657253,0.9414707966,0.9482235044,0.9509241826,0.9518281203,0.9536278976,0.9581344266,0.9594847657],
    '-bo', label="DiseaseState")

    plt.plot([100,200,400,500,1000,1200,2000,3400],
             [ 0.9697360156,0.9811679823,0.9838598874,0.984195456,0.9865495289,0.9862133944,0.9878946326,0.9885674674],
    '-yo', label="BioSourceType")


    # plt.xticks([0.01, 0.1, 0.25, 0.5],
    #            ['1%','10%','25%','50%'])
    # plt.plot([0.01, 0.1, 0.25, 0.5], [0.6538, 0.7407, 0.7865, 0.8297], '-^r', label="Our method")


    # plt.xlim(-0.02, 0.53)
    # plt.ylim(0.5, 0.85)
    plt.xlabel('Dimension')
    plt.ylabel('F1')
    plt.legend(loc='lower right')



    plt.savefig('./../fig/PCA_Dimension.pdf')
    plt.show()