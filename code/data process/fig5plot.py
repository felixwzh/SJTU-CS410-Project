# coding=utf-8
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from  math import log10
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
    plt.plot([log10(0.001),log10(0.01),log10(0.1),log10(1.0),log10(4.0),log10(8.0),log10(16.0),log10(32.0)],
             [0.997456046,0.997964808,0.997964808,0.997964808,0.998134299,0.998134299,0.998134299,0.998134299],
    '-ro', label="MaterialType")

    plt.plot([log10(0.001),log10(0.01),log10(0.1),log10(1.0),log10(4.0),log10(8.0),log10(16.0),log10(32.0)],
             [0.943360548
,0.927735945
,0.925781547
,0.925781547
,0.925781547
,0.925781547
,0.925781547
,0.925781547],
            '-go', label="Sex")

    plt.plot([log10(0.001),log10(0.01),log10(0.1),log10(1.0),log10(4.0),log10(8.0),log10(16.0),log10(32.0)],
             [0.949573844
,0.952726997
,0.954979249
,0.954979249
,0.954979249
,0.954979249
,0.954979249
,0.954979249],
    '-bo', label="DiseaseState")

    plt.plot([log10(0.001),log10(0.01),log10(0.1),log10(1.0),log10(4.0),log10(8.0),log10(16.0),log10(32.0)],
             [0.983522621
,0.983524319
,0.982178649
,0.981841949
,0.981841949
,0.981841949
,0.981841949
,0.981841949],
    '-yo', label="BioSourceType")


    # plt.xticks([0.01, 0.1, 0.25, 0.5],
    #            ['1%','10%','25%','50%'])
    # plt.plot([0.01, 0.1, 0.25, 0.5], [0.6538, 0.7407, 0.7865, 0.8297], '-^r', label="Our method")


    # plt.xlim(-0.02, 0.53)
    # plt.ylim(0.5, 0.85)
    plt.xlabel('Log10(C)')
    plt.ylabel('F1')
    plt.legend(loc='lower right',prop={'size':5})



    plt.savefig('LR_C.pdf')
    plt.show()