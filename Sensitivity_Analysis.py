import numpy as np
import random
import math
from random import sample
from prettytable import PrettyTable
import os, sys

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


def response_time_calculation_LO(task, task_set, Dropped):
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
        response_time = recursive_LO(response_time, hp_tasks, task, Dropped)
        print("response time", response_time)
        if response_time != rep:
            rep = response_time
        else:
            k = 0
    return response_time


def recursive_LO(response_time, hp_tasks, task, Dropped):

    temp0 = 0
    temp1 = 0

    execution_time = task.execution_time_LO
    for j in hp_tasks:
        if j in Dropped[0]:
            index = Dropped[0].index(j)
            print("kk", Dropped[1][index], j.period, j.execution_time_LO)
            temp0 += (math.ceil(Dropped[1][index] / j.period)) * j.execution_time_LO
        else:
            print("mark", response_time, j.period, j.execution_time_LO)
            temp1 += math.ceil(response_time / j.period) * j.execution_time_LO

        temp = temp0 + temp1
        print("temp", temp)

    response_time = execution_time + temp0 + temp1

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
            if index != -1:
                Test_tasks[index].priority = priority_temp
                # print(index)
                # table_print(Test_tasks)
            else:
                break
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


def response_time_HI_SA(task, Test_tasks, Dropped):

    print("get LO response time")
    response_time_LO_task = response_time_calculation_LO(task, Test_tasks, Dropped)
    print("the LO MODE response time:", response_time_LO_task)

    response_time_set = []
    MC_candidate = []

    for i in range(response_time_LO_task):
        print("+++++++++++++++++++++++++++++++++++++")
        s = i
        print("tested mode change time point:", s)
        response_time_HI_task = response_time_calculation_HI_SA(task, Test_tasks, s, Dropped)
        print("-------------------------------------")
        print("convergent response time:", response_time_HI_task, '\n')
        response_time_set.append(response_time_HI_task)
        MC_candidate.append(s)

    print("+++++++++++++++++++++++++++++++++++++", '\n')

    worst_case = max(response_time_set)
    index = response_time_set.index(worst_case)
    # print("index", index)
    print("The mode change time point:", MC_candidate[index], "with worst case response time:", worst_case)

    response = 0
    for i in range(len(response_time_set)):
        if response_time_set[i] > task.deadline:
            response = response_time_set[i]
            time_pont = MC_candidate[i]
            break

    if response > task.deadline:
        print("we need to drop task with the lowest importance value at time point s ", time_pont)

        # RT = response_time_calculation_HI_SA(task, Test_tasks, response_time_LO_task, Dropped)
        # if response

        LO_task_set = []
        temp = []

        table_print(Dropped[0])

        test = []
        temp = []
        for j in Test_tasks:
            if j.criticality == "LO":
                test.append(j)

        if len(Dropped[0]) == len(test):
            sati_HI = 0
        else:
            for i in Test_tasks:
                LO_task_set=[]
                if i.criticality == "LO":
                    if i not in Dropped[0]:
                        print(i.task_id)
                        LO_task_set.append(i)
                        temp.append(i.importance)

            print(temp)
            table_print(LO_task_set)
            temp_index = temp.index(max(temp))
            print(temp_index)
            print("TTTTTTTTTTTTTTTTT", LO_task_set[temp_index].task_id)
            Dropped[0].append(LO_task_set[temp_index])
            Dropped[1].append(time_pont)


            sati_HI = 1
    else:
        sati_HI = 0

    return worst_case, sati_HI


def response_time_calculation_HI_SA(task, task_set, time_point, Dropped):

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
        response_time = recursive_HI_SA(response_time, hp_tasks, task, MC_time_point, Dropped)
        print("response time", response_time)
        if response_time != rep:
            rep = response_time
        else:
            k = 0

    return response_time

def recursive_HI_SA(response_time, hp_tasks, task, MC_time_point, Dropped):

    temp0 = 0
    temp1 = 0
    temp2 = 0
    execution_time = task.execution_time_HI

    for j in hp_tasks:
        if j.criticality == "LO":
            if j in Dropped[0]:
                index = Dropped[0].index(j)
                temp0 += (math.ceil(Dropped[1][index] / j.period)) * j.execution_time_LO
                # index = Dropped[0].index(j)
                # if Dropped[1][index] >= MC_time_point:
                #     temp0 += (math.ceil(Dropped[1][index] / j.period)) * j.execution_time_LO
                # else:
                #     temp0 += (math.floor(MC_time_point / j.period) + 1) * j.execution_time_LO
            else:
                temp1 += (math.floor(MC_time_point / j.period) + 1) * j.execution_time_LO
        else:
            temp2 += math.ceil(MC_time_point / j.period) * j.execution_time_LO + math.ceil(
                (response_time - MC_time_point) / j.period) * (j.execution_time_HI - j.execution_time_LO)

    temp = temp0 + temp1 + temp2
    print("temp", temp)

    response_time = execution_time + temp

    return response_time

def Sensitivity_Analysis_LO(Test_tasks, Dropped):

    sati = 1
    for i in Test_tasks:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        if i not in Dropped[0]:
            response_time_LO = response_time_calculation_LO(i, Test_tasks, Dropped)
            if response_time_LO > i.deadline:
                print("!!!!!!!!! unschedulable !!!!!!!!", i.task_id)
                LO_task_set = []
                temp = []
                for j in Test_tasks:
                    if j.criticality == "LO":
                        if j not in Dropped[0]:
                            LO_task_set.append(j)
                            temp.append(j.importance)

                temp_index = temp.index(max(temp))
                # table_print(LO_task_set)
                # print("RRRRRRRRRRRRR", LO_task_set[temp_index].task_id)

                Dropped[0].append(LO_task_set[temp_index])
                Dropped[1].append(0)
                print("we need to drop tasks once overrun happened")
                sati = 1
                break
            else:
                sati = 0
                print("continual", sati)

    return sati


if __name__ == "__main__":

    print("Response time analysis")

    Test_tasks = []
    # task_id, deadline, period, execution_time_LO, execution_time_HI, priority, importance, criticality
    task1 = Task(1, 24, 24, 5, 15, 3, 0, "HI")
    # task1 = Task(1, 30, 30, 5, 15, 3, 0, "HI")
    Test_tasks.append(task1)
    task2 = Task(2, 20, 20, 5, 0, 4, 3, "LO")
    Test_tasks.append(task2)
    task3 = Task(3, 8, 8, 2, 0, 1, 2, "LO")
    Test_tasks.append(task3)
    task4 = Task(4, 5, 5, 1, 0, 2, 1, "LO")
    Test_tasks.append(task4)
    # task5 = Task(5, 30, 30, 1, 5, 5, 1, "HI")
    # Test_tasks.append(task5)

    print("---------Tasks in the system------------")
    table_print(Test_tasks)

    print("************* Sensitivity Analysis *****************")

    # in crease the overrun of HI tasks each time by 10%

    Dropped_Task = []
    Dropped_Time = []

    Dropped = [Dropped_Task,
               Dropped_Time]

    LO_task_set = []
    temp = []
    for j in Test_tasks:
        if j.criticality == "LO":
            LO_task_set.append(j)

    overrun_con = 1
    num_check = 0

    while 1:
        print('\n', "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", '\n')

        if len(LO_task_set) == len(Dropped[0]):
            print("All droppable tasks have already been dropped")
            table_print(Dropped[0])
            print(Dropped[1])
            break

        for i in Test_tasks:
            if i.execution_time_LO == i.execution_time_HI:
                num_check +=1

        HI_task_set = []
        for i in Test_tasks:
            if i.criticality == "HI":
                HI_task_set.append(i)

        if num_check == len(HI_task_set):
            print("All droppable tasks have already been dropped"
                  " and all HI task can be executed no more than their HI bound ",'\n',
                  "(Not all Low tasks need to be dropped)")

            table_print(Dropped[0])
            print(Dropped[1])
            break

        for i in Test_tasks:
            if i.criticality == 'HI':
                # i.execution_time_LO = math.floor((1 + 0.1*overrun_con) * i.execution_time_LO)
                i.execution_time_LO += 1
                if i.execution_time_LO >= i.execution_time_HI:
                    i.execution_time_LO = i.execution_time_HI

        print("Current overrun:", Test_tasks[0].execution_time_LO, '\n')
        table_print(Dropped[0])


        print("---------- Sensitivity Analysis LO-------------")
        # start = 1
        # round = 0
        # while start:
        #     print("round", round)
        #     sati = Sensitivity_Analysis_LO(Test_tasks, Dropped)
        #     round +=1
        #     if sati == 0:
        #         print("The system become schedulable after task dropping")
        #         break
        #
        # print("After Sensitivity Analysis LO, the dropped task under overload", 0.1 * overrun_con)
        # table_print(Dropped[0])
        # print(Dropped[1])

        sati = Sensitivity_Analysis_LO(Test_tasks, Dropped)
        if sati ==1:
            print("One Low task should be dropped once overrun happens")
            table_print(Dropped[0])
        else:
            print("Continue to find out the discarded task")

        print("---------- Sensitivity Analysis HI -------------")

        table_print(HI_task_set)
        start = 1
        num_check1 = 0
        HI_count = 0
        while start:

            if len(LO_task_set) == len(Dropped[0]):
                print("All droppable tasks have already been dropped")
                table_print(Dropped[0])
                print(Dropped[1])
                break

            for i in Test_tasks:
                if i.execution_time_LO == i.execution_time_HI:
                    num_check1 += 1

            if num_check1 == len(HI_task_set):
                print("All droppable tasks have already been dropped"
                      " and all HI task can be executed no more than their HI bound ", '\n',
                      "(Not all Low tasks need to be dropped)")

                table_print(Dropped[0])
                print(Dropped[1])
                break

            print("restart")
            table_print(Dropped[0])
            print(Dropped[1])


            for task in HI_task_set:
                if HI_count > len(HI_task_set):
                    sati_HI = 1
                    break
                else:
                    print("#########################################", '\n')
                    table_print(Dropped[0])
                    print(HI_count)
                    print("+++++++++++++++ task", task.task_id, "+++++++++++++++++++")
                    response_timeMC, sati_HI = response_time_HI_SA(task, Test_tasks, Dropped)
                    if sati_HI == 1:
                        break
                    elif sati_HI == 0:
                        HI_count += 1
                        print("continual HI", HI_count)

            if sati_HI == 1:
               print("Increase the overrun")
               start = 0

        overrun_con += 1

    print('\n', "++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Final result:")
    table_print(Dropped[0])
    print("Dropping time point", Dropped[1])












