import os
from scipy import stats
import numpy as np
from scipy.stats import mannwhitneyu, stats

def tTest(data1, data2):
    data11 = np.divide(data1, 100)
    data12 = np.divide(data2, 100)
    if ((data11 == data12).all()):
        return str(' ')
    else:
        # print('data1',data1)
        # print('data2',data2)
        t, p = stats.tTest(data11, data12)
        mean_data1 = np.mean(data11)
        mean_data2 = np.mean(data12)
        # print(t,p,mean_data2,mean_data1)
        if p < 0.05:
            if mean_data1 > mean_data2:
                return str('+')
            else:
                return str('-')
        elif p >= 0.05:
            return str(' ')
        else:
            return str('/')


def wilcoxonT(data1, data2):
    ##    print(data1.shape, data2.shape)
    ##    print(re_data2)
    ##    print(re_data2.shape, re_data2)
    data11 = np.divide(data1, 100)
    data12 = np.divide(data2, 100)
    #print(data12)
    if ((data11 == data12).all()):
        return str(' ')
    else:
        # print('data1',data1)
        # print('data2',data2)
        t, p = stats.wilcoxon(data11, data12)
        mean_data1 = np.mean(data11)
        mean_data2 = np.mean(data12)
        # print(t,p,mean_data2,mean_data1)
        if p < 0.05:
            if mean_data1 > mean_data2:
                return str('+')
            else:
                return str('--')
        elif p >= 0.05:
            return str('=')
        else:
            return str('/')


def wilcoxonT_unpaired(data1, data2):
    #data11 = np.divide(data1, 100)
    #data12 = np.divide(data2, 100)
    data11 = data1
    data12 = data2
    if ((data11 == data12).all()):
        return str(' same')
    else:
        # print('data1',data1)
        # print('data2',data2)
        t, p = stats.ranksums(data11, data12)
        mean_data1 = np.mean(data11)
        mean_data2 = np.mean(data12)
        # print(t,p,mean_data2,mean_data1)
        if p < 0.05:
            if mean_data1 > mean_data2:
                return str('+')
            else:
                return str('--')
        elif p >= 0.05:
            return str('=')
        else:
            return str('/')




def Wilcoxon_Rank_Sum_Test(x1, x2):
    statistic, pvalue = mannwhitneyu(x1, x2)
    alpha = 0.05
    if pvalue < alpha:
        if np.mean(x1) < np.mean(x2):     # 方法2优于方法1
            result = '+'
        else:
            result = '-'
    else:
        result = '='

    return result




data1 = [65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29, 65.29]
data2 = [66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86, 66.86]


print('(',Wilcoxon_Rank_Sum_Test(data2, data1),')')