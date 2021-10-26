import numpy as np
import random
import math
from random import sample
from prettytable import PrettyTable
import os, sys
import copy


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Task:
    criticality = ''

    def __init__(self, task, deadline, period, execution_time_LO, execution_time_HI, priority,
                 importance, criticality):
        self.task = task
        self.deadline = deadline
        self.period = period
        self.execution_time_LO = execution_time_LO
        self.execution_time_HI = execution_time_HI
        self.priority = priority
        self.importance = importance  # under different mode the value of importance may be varied
        self.criticality = criticality  # task criticality definition


def table_print(tasks):
    table = PrettyTable(
        ['task', 'deadline', 'period', 'execution_time_LO', 'execution_time_HI',
         'priority', 'importance', 'criticality'])
    for i in tasks:
        table.add_row(
            [i.task, i.deadline, i.period, i.execution_time_LO, i.execution_time_HI,
             i.priority, i.importance, i.criticality])
    print(table)


def SA_response_time_calculation_LO(task, task_set, Dropped):
    # print("##########################################################")
    print("The analysed task:", task.task)
    print("the execution time of analysed task:", task.execution_time_LO)
    response_time = task.execution_time_LO
    hp_tasks = []
    for i in task_set:
        if i.priority < task.priority:
            hp_tasks.append(i)
    print('\n', "Tasks with higher priority:")
    table_print(hp_tasks)
    print("Dropped tasks:")
    table_print(Dropped[0])
    rep = response_time
    print('\n', "------------------------------", '\n')
    print("start recursive")
    print('\n', "------------------------------")
    k = 1
    while k:
        response_time = SA_recursive_LO(response_time, hp_tasks, task, Dropped)
        print("response time", response_time)
        if response_time > task.deadline:
            break
        if response_time != rep:
            rep = response_time
        else:
            k = 0
    return response_time


def SA_recursive_LO(response_time, hp_tasks, task, Dropped):
    temp0 = 0
    temp1 = 0

    execution_time = task.execution_time_LO
    for j in hp_tasks:
        if j in Dropped[0]:
            index = Dropped[0].index(j)
            if response_time >= Dropped[1][index]:
                time_point  = Dropped[1][index]
            else:
                time_point = response_time
            print("interference from dropped task", time_point, j.period, j.execution_time_LO)
            temp0 += (math.floor(time_point / j.period) + 1) * j.execution_time_LO
        else:
            print("interference from HI task with higher priority", response_time, j.period, j.execution_time_LO)
            temp1 += math.ceil(response_time / j.period) * j.execution_time_LO

        temp = temp0 + temp1
        print("temp", temp)

    response_time = execution_time + temp0 + temp1

    return response_time


def response_time_HI_SA(task, Test_tasks, Dropped, overrun):
    print("get LO response time")
    response_time_LO_task = SA_response_time_calculation_LO(task, Test_tasks, Dropped)
    print("the LO MODE response time:", response_time_LO_task)

    response_time_set = []
    MC_candidate = []

    for i in range(math.floor(response_time_LO_task)):
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

    if worst_case > task.deadline:

        for i in range(len(response_time_set)):
            if response_time_set[i] >= task.deadline:
                response = response_time_set[i]
                if response == task.deadline:
                    upper_bound = response
                else:
                    upper_bound = response_time_set[i - 1]
                time_pont = MC_candidate[i]
                break

        print("we need to drop task with the lowest importance value at time point s ", time_pont, '\n')
        print("the execution time of checked task is ", task.execution_time_LO)
        # print(response_time_set)
        print("Already dropped tasks")
        table_print(Dropped[0])

        test = []
        temp = []
        for j in Test_tasks:
            if j.criticality == "LO":
                test.append(j)

        if len(Dropped[0]) == len(test):
            sati_HI = 0
        else:
            LO_task_set = []
            for i in Test_tasks:
                if i.criticality == "LO":
                    if i not in Dropped[0]:
                        # print(i.task)
                        LO_task_set.append(i)
                        temp.append(i.importance)

            table_print(LO_task_set)
            temp_index = temp.index(max(temp))

            print("The latest dropped task", LO_task_set[temp_index].task)
            # print(task.execution_time_LO)
            # print(overrun)

            test = copy.deepcopy(task)
            original = task.execution_time_LO / (1 + overrun)
            test.execution_time_LO = original * (1 + (overrun-0.1))
            with HiddenPrints():
                test_rp = SA_response_time_calculation_LO(test, Test_tasks, Dropped)
            # print("@@@@@@@", time_pont, test_rp)
            print("updated execution_time_LO", test.execution_time_LO)
            Dropped[0].append(LO_task_set[temp_index])
            Dropped[1].append(upper_bound)  # upperbound
            Dropped[2].append(test_rp)  # lower bound
            Dropped[3].append(overrun)
            Dropped[4].append((test.execution_time_LO, task.execution_time_LO))
            Dropped[5].append(task.task)

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
        if response_time > task.deadline:
            break
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
                print("interference from dropped task:", j.task)
                index = Dropped[0].index(j)
                if response_time >= Dropped[1][index]:
                    time_point = Dropped[1][index]
                else:
                    time_point = response_time
                print("The dropped time point", time_point)
                temp0 += (math.floor(time_point / j.period) + 1) * j.execution_time_LO
            else:
                print("interference from un-dropped task", j.task)
                temp1 += (math.floor(MC_time_point / j.period) + 1) * j.execution_time_LO
        else:
            print("interference from HI task with higher priority", j.task)
            temp2 += math.ceil(MC_time_point / j.period) * j.execution_time_LO + math.ceil(
                (response_time - MC_time_point) / j.period) * (j.execution_time_HI - j.execution_time_LO)

    temp = temp0 + temp1 + temp2
    print("temp", temp)

    response_time = execution_time + temp

    return response_time


def Sensitivity_Analysis_LO(Test_tasks, Dropped, overrun):
    print('\n', "Sensitivity_Analysis_LO", '\n')
    sati = 1
    for i in Test_tasks:
        print("**********************************************")
        if i not in Dropped[0]:
            response_time_LO = SA_response_time_calculation_LO(i, Test_tasks, Dropped)
            if response_time_LO > i.deadline:
                print("!!!!!!!!! unschedulable !!!!!!!!", i.task)
                LO_task_set = []
                temp = []
                for j in Test_tasks:
                    if j.criticality == "LO":
                        if j not in Dropped[0]:
                            LO_task_set.append(j)
                            temp.append(j.importance)

                temp_index = temp.index(max(temp))
                # table_print(LO_task_set)
                # print("RRRRRRRRRRRRR", LO_task_set[temp_index].task)

                Dropped[0].append(LO_task_set[temp_index])
                Dropped[1].append(0)
                Dropped[2].append(0)
                Dropped[3].append(overrun)
                Dropped[4].append(0)
                Dropped[5].append(i.task)

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
    # task, deadline, period, execution_time_LO, execution_time_HI, priority, importance, criticality
    task1 = Task(1, 25, 25, 5, 15, 3, 0, "HI")
    # task1 = Task(1, 30, 30, 5, 15, 3, 0, "HI")
    Test_tasks.append(task1)
    task2 = Task(2, 20, 20, 5, 0, 4, 3, "LO")
    Test_tasks.append(task2)
    task3 = Task(3, 8, 8, 2, 0, 1, 2, "LO")
    Test_tasks.append(task3)
    task4 = Task(4, 5, 5, 1, 0, 2, 1, "LO")
    Test_tasks.append(task4)
    task5 = Task(5, 30, 30, 1, 5, 5, 1, "HI")
    Test_tasks.append(task5)

    print("---------Tasks in the system------------")
    table_print(Test_tasks)

    print("************* Sensitivity Analysis *****************")

    # in crease the overrun of HI tasks each time by 10%

    Dropped_Task = []
    Dropped_Time_upperbound = []
    lower_bound = []
    system_overrun = []
    ex_time = []
    monitor_task = []

    Dropped = [Dropped_Task,
               Dropped_Time_upperbound,
               lower_bound,
               system_overrun,
               ex_time,
               monitor_task]

    LO_task_set = []
    temp = []
    for j in Test_tasks:
        if j.criticality == "LO":
            LO_task_set.append(j)

    overrun_con = 1
    num_check = 0

    execution_time_LO_cp = []
    Name_cp = []
    for i in Test_tasks:
        execution_time_LO_cp.append(i.execution_time_LO)
        Name_cp.append(i.task)

    while 1:
        print('\n', "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", '\n')

        if len(LO_task_set) == len(Dropped[0]):
            print("All droppable tasks have already been dropped")
            table_print(Dropped[0])
            print(Dropped[1])
            break

        for i in Test_tasks:
            if i.execution_time_LO == i.execution_time_HI:
                num_check += 1

        HI_task_set = []
        for i in Test_tasks:
            if i.criticality == "HI":
                HI_task_set.append(i)

        if num_check == len(HI_task_set):
            print("All HI task can be executed no more than their HI bound ", '\n',
                  "(Not all Low tasks need to be dropped)")

            table_print(Dropped[0])
            print(Dropped[1])
            break

        overrun = 0.1 * overrun_con


        for i in range(len(Test_tasks)):

            if Test_tasks[i].criticality == 'HI':
                # i.execution_time_LO = math.floor((1 + overrun) * i.execution_time_LO)
                # i.execution_time_LO += 1
                # print("before", Test_tasks[i].execution_time_LO)
                # print(overrun)
                Test_tasks[i].execution_time_LO = (1 + overrun) * execution_time_LO_cp[i]
                if Test_tasks[i].execution_time_LO >= Test_tasks[i].execution_time_HI:
                    Test_tasks[i].execution_time_LO = Test_tasks[i].execution_time_HI
                # print("after", Test_tasks[i].execution_time_LO)

        print("Current overrun:", overrun, '\n')
        table_print(Test_tasks)
        # print("Current overrun:", Test_tasks[0].execution_time_LO, '\n')
        print("Already drooped tasks:")
        table_print(Dropped[0])

        print('\n', "---------- Sensitivity Analysis LO-------------")

        sati = Sensitivity_Analysis_LO(Test_tasks, Dropped, overrun)
        if sati == 1:
            print("One Low task should be dropped once overrun happens")
            table_print(Dropped[0])
        else:
            print("Continue to find out the discarded task")

        print('\n', "---------- Sensitivity Analysis HI -------------")
        print('\n', "Sensitivity_Analysis_LO", '\n')

        table_print(HI_task_set)
        start = 1
        num_check1 = 0

        while start:

            Increace_mark = 0
            HI_count = 0

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

            print(" ++++++++++++++++ Restart +++++++++++++++++++")
            table_print(Dropped[0])
            print(Dropped[1])

            for task in HI_task_set:
                if HI_count > len(HI_task_set):
                    sati_HI = 1
                    break
                else:
                    print("#########################################", '\n')
                    print("Already dropped tasks")
                    table_print(Dropped[0])
                    # print(HI_count)
                    print("+++++++++++++++ task", task.task, "+++++++++++++++++++")
                    response_timeMC, sati_HI = response_time_HI_SA(task, Test_tasks, Dropped, overrun)
                    if sati_HI == 1:
                        Increace_mark = 0
                        break
                    elif sati_HI == 0:
                        Increace_mark = 1
                        HI_count += 1
                        # start = 0
                        print(HI_count)
                        print("The dropping task test of HI task:", task.task, "is finished", '\n')

            if Increace_mark == 1 and HI_count == len(HI_task_set):
                print("Increase the overrun")
                start = 0

        overrun_con += 1

    print('\n', "++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Final result:")
    table_print(Dropped[0])
    print("Dropping time point:", '\n', Dropped[1])
    print("System overrun:", '\n', Dropped[3])
    print("monitored HI tasks:", '\n', Dropped[5])

    for i in range(len(Dropped[0])):
        if Dropped[2][i] != 0:
            print('\n', "if HI task", Dropped[5][i], " with LO_execution time", Dropped[4][i][0], "(", Dropped[4][i][1], ")" 
                  "can not finish its execution after", Dropped[2][i], ".",
                  "It is allowed to be executed continuously.", '\n'
                  , " However, if the response time of it attempts to be larger than", Dropped[1][i], '.',
                  "LO task", Dropped[0][i].task, "should be dropped directly")
        else:
            print('\n', "Once overrun", Dropped[3][i], "happens. LO Task", Dropped[0][i].task,
                  "need to be dropped directly")
