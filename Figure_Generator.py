import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns


# the results of system with Uti 0.9

def Generator(uti, file_name, BBN_sur, Alan_sur, Droppable_task_num, EU_difference_holistic):
    for i in BBN_sur:
        for j in i:
            for k in range(len(j)):
                if j[k] < 0:
                    j[k] = 0
    e = 0
    file_name_p = copy.deepcopy(file_name)

    for i in range(len(file_name)):
        if file_name[i] == 555:
            # print(i)
            del file_name_p[i - e]
            del BBN_sur[i - e]
            del Alan_sur[i - e]
            del Droppable_task_num[i - e]
            del EU_difference_holistic[i - e]
            e += 1

    # print(len(BBN_sur), len(Alan_sur))
    better_EU = []
    same_EU = []
    for i in EU_difference_holistic:
        count = 0

        for j in range(len(i)):
            if i[j] > 0:
                count += 1
        better_EU.append(count)
        same_EU.append(len(i) - count)
    # uti = 0.3
    # print(better_EU, same_EU)


    # plt.suptitle(" Systems with Utilisation %s" % (uti), fontsize=16, y=1, weight="bold")
    print("=======================================")
    plt.subplot(231)
    x = np.arange(1, len(EU_difference_holistic) + 1, 1)
    plt.bar(x, better_EU, label='Improved', tick_label=x, color='cadetblue')
    plt.bar(x, same_EU, bottom=better_EU, label='Same', color='peru')
    plt.legend(loc='best', prop={'size': 11})
    plt.ylim([0, 10])
    plt.ylim([-1, 34])
    plt.tick_params(labelsize=14)
    # plt.xlabel("The system ID", fontsize=14)
    plt.ylabel("The proportion", fontsize=14)
    plt.title("The proportion of system with higher EU value", fontsize=14, pad=10)
    # plt.show()

    for i in range(len(EU_difference_holistic)):
        for j in range(len(EU_difference_holistic[i])):
            if np.isinf(EU_difference_holistic[i][j]):
                EU_difference_holistic[i][j] = 0
            if np.isnan(EU_difference_holistic[i][j]):
                EU_difference_holistic[i][j] = 0

    for i in range(len(EU_difference_holistic)):
        EU_difference_holistic[i] = list(filter(lambda x: x != 0, EU_difference_holistic[i]))

    EU_median = []
    for i in range(len(EU_difference_holistic)):
        # print(EU_difference_holistic[i])
        if len(EU_difference_holistic[i]) > 2:
            EU_difference_holistic[i].remove(max(EU_difference_holistic[i]))
            EU_difference_holistic[i].remove(min(EU_difference_holistic[i]))
        # print(EU_difference_holistic[i])
    # print(EU_difference_holistic[1])
    # print(np.mean(EU_difference_holistic[1]))
    EU_me=[]
    for i in EU_difference_holistic:
        EU_median.append(np.median(i))
        EU_me.append(np.mean(i))

    ana = []
    for i in EU_difference_holistic:
        for j in i:
            ana.append(j)
    print("The mean and median EU value", np.mean(ana), np.median(ana))
    # print(EU_mean)

    plt.subplot(232)
    bp = plt.boxplot(EU_difference_holistic, medianprops=None, showmeans=True, meanline=True, showfliers=True)
    # plt.setp(axes, xticklabels=file_name)
    plt.tick_params(labelsize=14)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], prop={'size': 11})
    # plt.xlabel("The system ID", fontsize=14)
    plt.ylabel("The EU value difference (%)", fontsize=14)
    plt.title("The distribution of EU value difference \n of improved system", fontsize=14, pad=10)
    # plt.show()

    plt.subplot(233)
    plt.bar(x, EU_median, tick_label=x, color='cadetblue')
    # plt.ylim([0, 10])
    # plt.ylim([-1, 33])
    plt.tick_params(labelsize=14)
    # plt.xlabel("The system number", fontsize=14)
    plt.ylabel("The EU value difference (%)", fontsize=14)
    plt.title("The median of EU value difference of improved system", fontsize=14, pad=10)
    # plt.show()

    for i in range(len(Alan_sur)):
        for j in range(len(Alan_sur[i])):
            if not Alan_sur[i][j]:
                Alan_sur[i][j] = [0]

    for i in range(len(BBN_sur)):
        for j in range(len(BBN_sur[i])):
            if not BBN_sur[i][j]:
                BBN_sur[i][j] = [0]

    Pro_diff = []
    for i in range(len(Droppable_task_num)):
        temp_0 = []
        for j in range(len(Droppable_task_num[i])):
            temp = []
            for k in range(len(BBN_sur[i][j])):
                temp.append(((BBN_sur[i][j][k] - Alan_sur[i][j][k]) / Droppable_task_num[i][j]) * 100)
            temp_0.append(np.mean(temp))
        Pro_diff.append(temp_0)

    better = []
    same = []
    for i in Pro_diff:
        count = 0
        for j in range(len(i)):
            if i[j] > 0:
                count += 1
        better.append(count)
        same.append(len(i) - count)

    # print(better, same)

    # file_name.remove(555)

    plt.subplot(234)
    x = np.arange(1, len(Pro_diff) + 1, 1)
    plt.bar(x, better, label='Improved', tick_label=x, color='grey')
    plt.bar(x, same, bottom=better, label='Same', color='rosybrown')
    plt.legend(loc='best', prop={'size': 11})
    plt.ylim([0, 10])
    plt.ylim([-1, 33])
    plt.tick_params(labelsize=14)
    plt.xlabel("The system ID", fontsize=14)
    plt.ylabel("The proportion", fontsize=14)
    plt.title("The proportion of system with improved survivability", fontsize=14, pad=10)
    # plt.show()

    for i in range(len(Pro_diff)):
        Pro_diff[i] = list(filter(lambda x: x != 0, Pro_diff[i]))

    ana2 = []
    for i in Pro_diff:
        for j in i:
            ana2.append(j)
    print("the survive mean and median", np.mean(ana2), np.median(ana2))

    # print(Pro_diff)
    plt.subplot(235)
    bp = plt.boxplot(Pro_diff, medianprops=None, showmeans=True, meanline=True, showfliers=True)
    # plt.setp(axes, xticklabels=file_name)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], prop={'size': 11})
    plt.tick_params(labelsize=14)
    plt.xlabel("The system ID", fontsize=14)
    plt.ylabel("The percentage difference (%)", fontsize=14)
    plt.title("The distribution of survived percentage difference \n of improved system", fontsize=14, pad=10)
    # plt.show()

    Pro_mean = []
    for i in Pro_diff:
        Pro_mean.append(np.median(i))
    # print(Pro_mean)
    plt.subplot(236)
    plt.bar(x, Pro_mean, tick_label=x, color='grey')
    # plt.ylim([0, 10])
    # plt.ylim([-1, 33])
    plt.tick_params(labelsize=14)
    plt.xlabel("The system ID", fontsize=14)
    plt.ylabel("The percentage difference (%)", fontsize=14)
    plt.title("The median of survived percentage difference \n of improved system", fontsize=14, pad=10)
    plt.show()

    return better_EU, same_EU, EU_difference_holistic, better, same, Pro_diff


def Integration(Better_EU, Same_EU, EU_diff, Better, Same, Pro_Diff, better_EU, same_EU, EU_difference_holistic, better,
                same, Pro_diff):

    Better_EU.append(sum(better_EU))
    Same_EU.append(sum(same_EU))
    temp = []
    for i in EU_difference_holistic:
        for j in i:
            temp.append(j)
    EU_diff.append(temp)

    Better.append(sum(better))
    Same.append(sum(same))
    temp = []
    for i in Pro_diff:
        for j in i:
            temp.append(j)
    Pro_Diff.append(temp)

    # Better_EU, Same_EU, EU_diff, Better, Same, Pro_Diff
    return


if __name__ == "__main__":

    # Data load 03
    with open('/home/jiezou/Documents/Context_aware MCS/L_results/file_name_03_30.pickle', 'rb') as handle:
        file_name_03 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_BBN_sur_03_30.pickle', 'rb') as handle:
        BBN_sur_03 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Alan_sur_03_30.pickle', 'rb') as handle:
        Alan_sur_03 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Drop_all_03_30.pickle', 'rb') as handle:
        Droppable_task_num_03 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/EU_difference_holistic_03_30.pickle', 'rb') as handle:
        EU_difference_holistic_03 = pickle.load(handle)

    # Data load 04
    with open('/home/jiezou/Documents/Context_aware MCS/L_results/file_name_04_30.pickle', 'rb') as handle:
        file_name_04 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_BBN_sur_04_30.pickle', 'rb') as handle:
        BBN_sur_04 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Alan_sur_04_30.pickle', 'rb') as handle:
        Alan_sur_04 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Drop_all_04_30.pickle', 'rb') as handle:
        Droppable_task_num_04 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/EU_difference_holistic_04_30.pickle', 'rb') as handle:
        EU_difference_holistic_04 = pickle.load(handle)

    # Data load 05

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/file_name_05_30.pickle', 'rb') as handle:
        file_name_05 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_BBN_sur_05_30.pickle', 'rb') as handle:
        BBN_sur_05 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Alan_sur_05_30.pickle', 'rb') as handle:
        Alan_sur_05 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Drop_all_05_30.pickle', 'rb') as handle:
        Droppable_task_num_05 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/EU_difference_holistic_05_30.pickle', 'rb') as handle:
        EU_difference_holistic_05 = pickle.load(handle)

    # Data load 05

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/file_name_06_30.pickle', 'rb') as handle:
        file_name_06 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_BBN_sur_06_30.pickle', 'rb') as handle:
        BBN_sur_06 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Alan_sur_06_30.pickle', 'rb') as handle:
        Alan_sur_06 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Drop_all_06_30.pickle', 'rb') as handle:
        Droppable_task_num_06 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/EU_difference_holistic_06_30.pickle', 'rb') as handle:
        EU_difference_holistic_06 = pickle.load(handle)

    # Data load 07

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/file_name_07_30.pickle', 'rb') as handle:
        file_name_07 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_BBN_sur_07_30.pickle', 'rb') as handle:
        BBN_sur_07 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Alan_sur_07_30.pickle', 'rb') as handle:
        Alan_sur_07 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Drop_all_07_30.pickle', 'rb') as handle:
        Droppable_task_num_07 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/EU_difference_holistic_07_30.pickle',
              'rb') as handle:
        EU_difference_holistic_07 = pickle.load(handle)

    # Data load 08

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/file_name_08_30.pickle', 'rb') as handle:
        file_name_08 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_BBN_sur_08_30.pickle', 'rb') as handle:
        BBN_sur_08 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Alan_sur_08_30.pickle', 'rb') as handle:
        Alan_sur_08 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Drop_all_08_30.pickle', 'rb') as handle:
        Droppable_task_num_08 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/EU_difference_holistic_08_30.pickle',
              'rb') as handle:
        EU_difference_holistic_08 = pickle.load(handle)

    # Data load 09

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/file_name_09_30.pickle', 'rb') as handle:
        file_name_09 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_BBN_sur_09_30.pickle', 'rb') as handle:
        BBN_sur_09 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Alan_sur_09_30.pickle', 'rb') as handle:
        Alan_sur_09 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Drop_all_09_30.pickle', 'rb') as handle:
        Droppable_task_num_09 = pickle.load(handle)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/EU_difference_holistic_09_30.pickle',
              'rb') as handle:
        EU_difference_holistic_09 = pickle.load(handle)

    Better_EU = []
    Same_EU = []
    EU_diff = []

    Better = []
    Same = []
    Pro_Diff = []



    better_EU_03, same_EU_03, EU_difference_holistic_03, better_03, same_03, Pro_diff_03 = Generator(0.3, file_name_03,
                                                                                                     BBN_sur_03,
                                                                                                     Alan_sur_03,
                                                                                                     Droppable_task_num_03,
                                                                                                     EU_difference_holistic_03)

    Integration(Better_EU, Same_EU, EU_diff, Better, Same, Pro_Diff, better_EU_03, same_EU_03,
                EU_difference_holistic_03, better_03,same_03, Pro_diff_03)


    better_EU_04, same_EU_04, EU_difference_holistic_04, better_04, same_04, Pro_diff_04 = Generator(0.4, file_name_04,
                                                                                                     BBN_sur_04,
                                                                                                     Alan_sur_04,
                                                                                                     Droppable_task_num_04,
                                                                                                     EU_difference_holistic_04)

    Integration(Better_EU, Same_EU, EU_diff, Better, Same, Pro_Diff, better_EU_04, same_EU_04,
                EU_difference_holistic_04, better_04,same_04, Pro_diff_04)



    better_EU_05, same_EU_05, EU_difference_holistic_05, better_05, same_05, Pro_diff_05 = Generator(0.5, file_name_05,
                                                                                                     BBN_sur_05,
                                                                                                     Alan_sur_05,
                                                                                                     Droppable_task_num_05,
                                                                                                     EU_difference_holistic_05)
    Integration(Better_EU, Same_EU, EU_diff, Better, Same, Pro_Diff, better_EU_05, same_EU_05,
                EU_difference_holistic_05, better_05, same_05, Pro_diff_05)

    better_EU_06, same_EU_06, EU_difference_holistic_06, better_06, same_06, Pro_diff_06 = Generator(0.6, file_name_06,
                                                                                                     BBN_sur_06,
                                                                                                     Alan_sur_06,
                                                                                                     Droppable_task_num_06,
                                                                                                     EU_difference_holistic_06)

    Integration(Better_EU, Same_EU, EU_diff, Better, Same, Pro_Diff, better_EU_06, same_EU_06,
                EU_difference_holistic_06, better_06, same_06, Pro_diff_06)

    better_EU_07, same_EU_07, EU_difference_holistic_07, better_07, same_07, Pro_diff_07 = Generator(0.7, file_name_07,
                                                                                                     BBN_sur_07,
                                                                                                     Alan_sur_07,
                                                                                                     Droppable_task_num_07,
                                                                                                     EU_difference_holistic_07)
    Integration(Better_EU, Same_EU, EU_diff, Better, Same, Pro_Diff, better_EU_07, same_EU_07,
                EU_difference_holistic_07, better_07, same_07, Pro_diff_07)


    better_EU_08, same_EU_08, EU_difference_holistic_08, better_08, same_08, Pro_diff_08 = Generator(0.8, file_name_08,
                                                                                                     BBN_sur_08,
                                                                                                     Alan_sur_08,
                                                                                                     Droppable_task_num_08,
                                                                                                     EU_difference_holistic_08)
    Integration(Better_EU, Same_EU, EU_diff, Better, Same, Pro_Diff, better_EU_08, same_EU_08,
                EU_difference_holistic_08, better_08, same_08, Pro_diff_08)

    better_EU_09, same_EU_09, EU_difference_holistic_09, better_09, same_09, Pro_diff_09 = Generator(0.9, file_name_09,
                                                                                                     BBN_sur_09,
                                                                                                     Alan_sur_09,
                                                                                                     Droppable_task_num_09,
                                                                                                     EU_difference_holistic_09)
    Integration(Better_EU, Same_EU, EU_diff, Better, Same, Pro_Diff, better_EU_09, same_EU_09,
                    EU_difference_holistic_09, better_09, same_09, Pro_diff_09)
    # print(Better_EU)
    # print(Same_EU)



    plt.subplot(231)
    Uti = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.bar(range(len(Uti)), Better_EU, label='Improved', tick_label=Uti, color='lightsteelblue')
    plt.bar(range(len(Uti)), Same_EU, bottom=Better_EU, label='Same', color='thistle')
    plt.legend(loc='best', prop={'size': 11})
    # plt.ylim([0, 10])
    plt.ylim([-1, 330])
    plt.tick_params(labelsize=14)
    # plt.xlabel("System Utilisation", fontsize=14)
    plt.ylabel("The proportion", fontsize=14)
    plt.title("The proportion of system with higher EU value", fontsize=14, pad=10)

    axes = plt.subplot(232)
    bt=plt.boxplot(EU_diff, medianprops=None, showmeans=True, meanline=True, showfliers=True)
    plt.setp(axes, xticklabels=Uti)
    plt.tick_params(labelsize=14)
    plt.legend([bt['medians'][0], bt['means'][0]], ['median', 'mean'], prop={'size': 11})
    # plt.xlabel("System Utilisation", fontsize=14)
    plt.ylabel("The EU value difference (%)", fontsize=14)
    plt.title("The distribution of EU value difference \n of improved system", fontsize=14, pad=10)

    EU_median = []
    for i in EU_diff:
        EU_median.append(np.median(i))

    plt.subplot(233)
    plt.bar(range(len(Uti)), EU_median, tick_label=Uti, color='cadetblue')
    # plt.ylim([0, 10])
    # plt.ylim([-1, 33])
    plt.tick_params(labelsize=14)
    # plt.xlabel("System Utilisation", fontsize=14)
    plt.ylabel("The EU value difference (%)", fontsize=14)
    plt.title("The median of EU value difference of improved system", fontsize=14, pad=10)

    plt.subplot(234)
    plt.bar(range(len(Uti)), Better, label='Improved', tick_label=Uti, color='silver')
    plt.bar(range(len(Uti)), Same, bottom=Better, label='Same', color='burlywood')
    plt.legend(loc='best', prop={'size': 11})
    # plt.ylim([0, 10])
    plt.ylim([-1, 330])
    plt.tick_params(labelsize=14)
    plt.xlabel("System Utilisation", fontsize=14)
    plt.ylabel("The proportion", fontsize=14)
    plt.title("The proportion of system with improved survivability", fontsize=14, pad=10)

    axes = plt.subplot(235)
    bp = plt.boxplot(Pro_Diff, medianprops=None, showmeans=True, meanline=True, showfliers=True)
    plt.setp(axes, xticklabels=Uti)
    plt.tick_params(labelsize=14)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], prop={'size': 11})
    plt.xlabel("System Utilisation", fontsize=14)
    plt.ylabel("The percentage difference (%)", fontsize=14)
    plt.title("The distribution of survived percentage difference \n of improved system", fontsize=14, pad=10)

    Pro_median = []
    for i in Pro_Diff:
        Pro_median.append(np.median(i))
    # print(Pro_mean)
    plt.subplot(236)
    plt.bar(range(len(Uti)), Pro_median, tick_label=Uti, color='grey')
    # plt.ylim([0, 10])
    # plt.ylim([-1, 33])
    plt.tick_params(labelsize=14)
    plt.xlabel("System Utilisation", fontsize=14)
    plt.ylabel("The percentage difference (%)", fontsize=14)
    plt.title("The median of survived percentage difference \n of improved system", fontsize=14, pad=10)

    plt.show()