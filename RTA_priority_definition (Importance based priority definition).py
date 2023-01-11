import numpy as np
import random
import math
from random import sample
from prettytable import PrettyTable
import os, sys

from operator import itemgetter, attrgetter

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Task:
    criticality = ''

    def __init__(self, task_id, deadline, period, execution_time_LO, execution_time_HI, priority,
                 importance, criticality):
        self.task_id = task_id
        self.deadline = deadline
        self.period = period
        self.execution_time_LO = execution_time_LO
        self.execution_time_HI = execution_time_HI
        self.priority = priority
        self.importance = importance  # under different mode the value of importance may be varied
        self.criticality = criticality  # task criticality definition


def table_print(tasks):
    table = PrettyTable(
        ['task_id', 'deadline', 'period', 'execution_time_LO', 'execution_time_HI',
         'priority', 'importance', 'criticality'])
    for i in tasks:
        table.add_row(
            [i.task_id, i.deadline, i.period, i.execution_time_LO, i.execution_time_HI,
             i.priority, i.importance, i.criticality])
    print(table)


def response_time_calculation_LO(task, task_set):
    # print("##########################################################")
    print("The analysed task:", task.task_id)
    print("the execution time of analysed task:", task.execution_time_LO)
    response_time = task.execution_time_LO
    hp_tasks = []
    for i in task_set:
        if i.priority < task.priority:
            hp_tasks.append(i)
    print(" ----------------tasks with higher priority----------------")
    table_print(hp_tasks)
    rep = response_time
    print("start recursive")
    k = 1
    while k:
        response_time = recursive_LO(response_time, hp_tasks, task)
        print("response time", response_time)
        if response_time != rep:
            rep = response_time
        else:
            k = 0
    return response_time


def recursive_LO(response_time, hp_tasks, task):
    temp = 0
    execution_time = task.execution_time_LO
    for j in hp_tasks:
        temp += math.ceil(response_time / j.period) * j.execution_time_LO
        print("temp", temp)

    response_time = execution_time + temp

    return response_time


def recursive_HI(response_time, hp_tasks, task):
    temp = 0
    execution_time = task.execution_time_HI
    for j in hp_tasks:
        temp += math.ceil(response_time / j.period) * j.execution_time_HI
        print("temp", temp)

    response_time = execution_time + temp

    return response_time


def response_time_calculation_HI(task, task_set):
    # print("##########################################################")
    print("The analysed task:", task.task_id)
    print("the execution time of analysed task:", task.execution_time_HI)
    response_time = task.execution_time_HI
    hp_tasks = []
    for i in task_set:
        if i.priority < task.priority and i.criticality == 'HI':
            hp_tasks.append(i)
    print(" ----------------tasks with higher priority----------------")
    table_print(hp_tasks)
    rep = response_time
    print("start recursive")
    k = 1
    while k:
        response_time = recursive_HI(response_time, hp_tasks, task)
        print("response time", response_time)
        if response_time != rep:
            rep = response_time
        else:
            k = 0
    return response_time


def response_time_calculation_HI_MC(task, task_set, time_point):
    response_time = task.execution_time_HI
    MC_time_point = time_point
    hp_tasks = []
    for i in task_set:
        if i.priority < task.priority:
            hp_tasks.append(i)
    print(" ----------------tasks with higher priority----------------")
    table_print(hp_tasks)
    rep = response_time
    print("start recursive")
    require = "HI"
    k = 1
    while k:
        response_time = recursive_HI_MC(response_time, hp_tasks, task, require, MC_time_point)
        print("response time", response_time)
        if response_time != rep:
            rep = response_time
        else:
            k = 0
    return response_time


def recursive_HI_MC(response_time, hp_tasks, task, require, MC_time_point):

    temp0 = 0
    temp1 = 0
    execution_time = task.execution_time_HI
    for j in hp_tasks:
        if j.criticality == "LO":
            temp0 += (math.floor(MC_time_point / j.period) + 1) * j.execution_time_LO
        else:
            temp1 += math.ceil(MC_time_point / j.period) * j.execution_time_LO + math.ceil(
                (response_time - MC_time_point) / j.period) * (j.execution_time_HI - j.execution_time_LO)
    temp = temp0 + temp1
    print("temp", temp)

    response_time = execution_time + temp

    return response_time


def response_time_Mode_change(task, Test_tasks):

    response_time_LO_task = response_time_calculation_LO(task, Test_tasks)
    print("the LO MODE response time:", response_time_LO_task)

    response_time_set = []
    MC_candidate = []
    for i in range(response_time_LO_task):
        print("+++++++++++++++++++++++++++++++++++++")
        s = i
        print("tested mode change time point:", s)
        response_time_HI_task = response_time_calculation_HI_MC(task, Test_tasks, s)
        print("-------------------------------------")
        print("convergent response time:", response_time_HI_task, '\n')
        response_time_set.append(response_time_HI_task)
        MC_candidate.append(s)

    print("+++++++++++++++++++++++++++++++++++++", '\n')

    worst_case = max(response_time_set)
    index = response_time_set.index(worst_case)
    # print("index", index)
    print("The mode change time point:", MC_candidate[index], "with worst case response time:", worst_case)
    return worst_case


def schedulability_check(task, Test_tasks):

    print("LO MODE response time analysis for PRIORITY")
    with HiddenPrints():
        response_time_LO = response_time_calculation_LO(task, Test_tasks)
    print("response time for LO MODE:", response_time_LO)
    if response_time_LO <= task.deadline:
        safisty = 1
    else:
        safisty = 0

    if task.criticality == 'HI':
        print("-------------------------------------------")
        print("HI MODE response time analysis for PRIORITY")
        print("-------------------------------------------")
        with HiddenPrints():
            response_time_HI = response_time_calculation_HI(task, Test_tasks)
        print("response time for HI MODE:", response_time_HI)

        with HiddenPrints():
            response_timeMC = response_time_Mode_change(task, Test_tasks)
        print("response time for mode change", response_timeMC)
        print("-------------------------------------")
        if response_time_HI <= task.deadline and response_timeMC <= task.deadline:
            safisty = 1
        else:
            safisty = 0

    return safisty


def priority_recursive(priority_temp, Test_tasks):

    for i in range(len(Test_tasks)):

        if Test_tasks[i].priority != -1:
            for i in range(len(Test_tasks)):
                 if Test_tasks[i].priority == -1:
                    index = i
                    break
            Test_tasks[index].priority = priority_temp
            # if index != -1:
            #     Test_tasks[index].priority = priority_temp
            #     # print(index)
            #     # table_print(Test_tasks)
            # else:
            #     break
        else:
            Test_tasks[i].priority = priority_temp

        # table_print(Test_tasks)
        print('\n', "current checked task ID", i+1)
        table_print(Test_tasks)

        safisty = schedulability_check(Test_tasks[i], Test_tasks)

        if safisty == 1:
            # Test_tasks[index].priority = priority_temp
            priority_temp -= 1
            print("!! schedulable update priority")
            table_print(Test_tasks)
            break

        else:
            print("The task is unschedulable with the priority level ")
            Test_tasks[i].priority = -1
            table_print(Test_tasks)

    return priority_temp

if __name__ == "__main__":

    print("Response time analysis")

    Test_tasks = []
    # task_id, deadline, period, execution_time_LO, execution_time_HI, priority,
    #                  importance, criticality
    task1 = Task(1, 25, 25, 5, 15, 3, 0, "HI")
    # task1 = Task(1, 30, 30, 5, 15, 3, 0, "HI")
    Test_tasks.append(task1)
    task2 = Task(2, 20, 20, 5, 0, 4, 3, "LO")
    Test_tasks.append(task2)
    task3 = Task(3, 8, 8, 2, 0, 1, 2, "LO")
    Test_tasks.append(task3)
    task4 = Task(4, 5, 5, 1, 0, 2, 4, "LO")
    Test_tasks.append(task4)
    task5 = Task(5, 60, 60, 1, 3, 2, 1, "HI")
    Test_tasks.append(task5)



    # table_print(Test_tasks)

    temp = []
    for i in Test_tasks:
        temp.append(i.importance)


    table_print(Test_tasks)
    Test_tasks = sorted(Test_tasks, key=lambda Task:Task.importance, reverse=True)
    table_print(Test_tasks)

# -------- Priority definition --------#
#     print('\n', "START TO DETERMINE THE PRIORITY")
    priority_level = len(Test_tasks)
    unschedulable = []
    for i in Test_tasks:
        i.priority = -1
        unschedulable.append(i.task_id)
    table_print(Test_tasks)

    priority_temp = priority_level

    conti  = 1

    while conti:
        print("*********************")
        num = 0
        # table_print(Test_tasks)
        for i in Test_tasks:
            if i.priority == -1:
                num +=1
        if num == 0:
            break
        else:
            print("The allocated priority level:", priority_temp)
            priority_temp = priority_recursive(priority_temp, Test_tasks)

    print('\n', "Final result:", '\n')
    Test_tasks = sorted(Test_tasks, key=attrgetter('task_id'))
    table_print(Test_tasks)














