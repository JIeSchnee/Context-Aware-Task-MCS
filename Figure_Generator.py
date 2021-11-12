import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

# the results of system with Uti 0.9


with open('/home/jiezou/Documents/Context_aware MCS/L_results/file_name_09_25.pickle', 'rb') as handle:
    file_name = pickle.load(handle)

with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_BBN_sur_09_25.pickle', 'rb') as handle:
    BBN_sur = pickle.load(handle)
for i in BBN_sur:
    for j in i:
        for k in range(len(j)):
            if j[k] < 0:
                j[k] = 0

with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Alan_sur_09_25.pickle', 'rb') as handle:
    Alan_sur = pickle.load(handle)

with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Drop_all_09_25.pickle', 'rb') as handle:
    Droppable_task_num = pickle.load(handle)

with open('/home/jiezou/Documents/Context_aware MCS/L_results/EU_difference_holistic_09_25.pickle', 'rb') as handle:
    EU_difference_holistic = pickle.load(handle)


for i in range(len(file_name)):
    if file_name[i] == 555:
        # print(i)
        del BBN_sur[i]
        del Alan_sur[i]
        del Droppable_task_num[i]
        del EU_difference_holistic[i]

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

# print(better_EU, same_EU)

x = np.arange(1, len(EU_difference_holistic)+1, 1)
plt.bar(x, better_EU, label='Improved', tick_label=x)
plt.bar(x, same_EU, bottom=better_EU, label='Same')
plt.legend(loc='best', prop={'size': 26})
plt.ylim([0, 10])
plt.ylim([-1, 24])
plt.tick_params(labelsize=14)
plt.xlabel("The graph number", fontsize=26)
plt.ylabel("The proportion (%)", fontsize=26)
plt.title("The proportion of system with higher EU value", fontsize=26, pad=24)
plt.show()
for i in range(len(EU_difference_holistic)):
    for j in range(len(EU_difference_holistic[i])):
        if np.isinf(EU_difference_holistic[i][j]):
            EU_difference_holistic[i][j] = 0

for i in range(len(EU_difference_holistic)):
    EU_difference_holistic[i] = list(filter(lambda x: x != 0, EU_difference_holistic[i]))

EU_mean = []
for i in range(len(EU_difference_holistic)):

    if len(EU_difference_holistic[i]) > 2:
        EU_difference_holistic[i].remove(max(EU_difference_holistic[i]))
        EU_difference_holistic[i].remove(min(EU_difference_holistic[i]))
# print(EU_difference_holistic[1])
# print(np.mean(EU_difference_holistic[1]))

for i in EU_difference_holistic:
    EU_mean.append(np.mean(i))

# print(EU_mean)

fig1, axes = plt.subplots()
plt.boxplot(EU_difference_holistic, medianprops=None, showmeans=True, meanline=True, showfliers=False)
# plt.setp(axes, xticklabels=file_name)
plt.tick_params(labelsize=14)
plt.xlabel("The graph number", fontsize=26)
plt.ylabel("The distribution of EU value difference (%)", fontsize=26)
plt.title("The distribution of EU value difference during degradation", fontsize=26, pad=24)
plt.show()

plt.bar(x, EU_mean, tick_label=x)
# plt.ylim([0, 10])
# plt.ylim([-1, 24])
plt.tick_params(labelsize=14)
plt.xlabel("The graph number", fontsize=26)
plt.ylabel("The average of EU value difference (%)", fontsize=26)
plt.title("The average of EU value difference during degradation", fontsize=26, pad=24)
plt.show()


Pro_diff = []
for i in range(len(Droppable_task_num)):
    temp_0 = []
    for j in range(len(Droppable_task_num[i])):
        temp = []
        for k in range(len(BBN_sur[i][j])):
            temp.append(((BBN_sur[i][j][k] - Alan_sur[i][j][k]) / Droppable_task_num[i][j]) * 100)
        temp_0.append(np.mean(temp))
    Pro_diff.append(temp_0)

# print(Pro_diff)

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

file_name.remove(555)
x = np.arange(1, len(Pro_diff)+1, 1)
plt.bar(x, better, label='Improved', tick_label=x)
plt.bar(x, same, bottom=better, label='Same')
plt.legend(loc='best', prop={'size': 26})
plt.ylim([0, 10])
plt.ylim([-1, 24])
plt.tick_params(labelsize=14)
plt.xlabel("The graph number", fontsize=26)
plt.ylabel("The proportion (%)", fontsize=26)
plt.title("The proportion of system with improved survivability", fontsize=26, pad=24)
plt.show()

for i in range(len(Pro_diff)):
    Pro_diff[i] = list(filter(lambda x: x != 0, Pro_diff[i]))

# print(Pro_diff)
fig1, axes = plt.subplots()
plt.boxplot(Pro_diff, medianprops=None, showmeans=False, meanline=True, showfliers=False)
# plt.setp(axes, xticklabels=file_name)
plt.tick_params(labelsize=14)
plt.xlabel("The graph number", fontsize=26)
plt.ylabel("The distribution of survived percentage difference (%)", fontsize=26)
plt.title("The distribution of survived percentage difference during degradation", fontsize=26, pad=24)
plt.show()

Pro_mean = []
for i in Pro_diff:
    Pro_mean.append(np.mean(i))
# print(Pro_mean)

plt.bar(x, Pro_mean, tick_label=x)
# plt.ylim([0, 10])
# plt.ylim([-1, 24])
plt.tick_params(labelsize=14)
plt.xlabel("The graph number", fontsize=26)
plt.ylabel("The average of survived percentage difference (%)", fontsize=26)
plt.title("The average of survived percentage difference during degradation", fontsize=26, pad=24)
plt.show()
