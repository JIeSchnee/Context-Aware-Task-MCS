import numpy as np
import random
import math
from random import sample
from prettytable import PrettyTable


class Task:
    criticality = ''

    def __init__(self, task_id, app_id, release_time, deadline, period, execution_time_LO, execution_time_HI, priority,
                 importance, criticality, context_aware):
        self.task_id = task_id
        self.app_id = app_id
        self.release_time = release_time
        self.deadline = deadline
        self.period = period
        self.execution_time_LO = execution_time_LO
        self.execution_time_HI = execution_time_HI
        self.priority = priority
        self.importance = importance  # under different mode the value of importance may be varied
        self.criticality = criticality  # task criticality definition
        self.context_aware = context_aware  # true 1, false 0


def execution_time_Generator(task_number, period, target_Uti):
    utilizations = uunifast(task_number, target_Uti)

    execution_times = np.multiply(np.array(utilizations), np.array(period))
    # print(generated_transmission_times)
    for i in range(len(execution_times)):
        if execution_times[i] < 1:
            execution_times[i] = 1
        elif execution_times[i] >= 1:
            execution_times[i] = math.floor(execution_times[i])

    return execution_times


def uunifast(stream_number, target_Uti):
    sum_utilization = target_Uti
    utilizations = []

    for i in range(1, stream_number):
        nextSumU = sum_utilization * random.uniform(0, 1) ** (1.0 / (stream_number - i))
        utilizations.append(sum_utilization - nextSumU)
        sum_utilization = nextSumU

    utilizations.append(sum_utilization)
    return utilizations


def task_definition(task_number, period, execution_time, criticality_factor, criticality_proportion):
    tasks = []
    HI_task_num = math.floor(task_number * criticality_proportion) - 1
    # context aware task will be selected from LO group
    index_init = range(task_number)
    HI_task_index = sample(index_init, HI_task_num)
    # print("the index of HI-criticality tasks in one task set:", HI_task_index)
    # context_aware_index = sample(HI_task_index, 1)
    # print("the index of context-aware task in one task set:", context_aware_index)

    for i in range(task_number):
        if i in HI_task_index:
            criticality = "HI"
            execution_time_HI = execution_time[i] * criticality_factor
        else:
            criticality = "LO"
            execution_time_HI = 0

        # if i in context_aware_index:
        #     context_aware = 1
        # else:
        #     context_aware = 0

        # release_time = random.randint(0, hyperperiod_calculation(period))
        release_time = 0

        task = Task(0, 0, release_time, period[i], period[i], execution_time[i], execution_time_HI, i, i,
                    criticality, 0)

        tasks.append(task)
    return tasks


def hyperperiod_calculation(period):
    hyperperiod = period[0]
    for i in range(1, len(period)):
        hyperperiod = hyperperiod * period[i] // math.gcd(hyperperiod, period[i])

    return hyperperiod


def table_print(tasks):
    table = PrettyTable(
        ['task_id', 'app_id', 'release_time', 'deadline', 'period', 'execution_time_LO', 'execution_time_HI',
         'priority',
         'importance', 'criticality', 'context-aware:'])
    for i in tasks:
        table.add_row(
            [i.task_id, i.app_id, i.release_time, i.deadline, i.period, i.execution_time_LO, i.execution_time_HI,
             i.priority, i.importance, i.criticality, i.context_aware])
    print(table)


def cmp(a):
    return a.deadline


def tasks_sort(app_tasks, app_id):
    for i in app_tasks:
        i.app_id = app_id
    app_tasks.sort(key=cmp)
    for i in range(len(app_tasks)):
        app_tasks[i].task_id = i

    return app_tasks


if __name__ == "__main__":
    print("Start Tasks Generation")
    print("--------------------------------")

    task_number = 15  # varied from 4 to 60, the task of one task set
    target_Uti = 0.5  # varied from 0.05 to 0.95

    period_set = [10, 50, 100, 300, 600, 900, 1800]
    period = []
    for i in range(task_number):
        period.append(random.choice(period_set))
    print("Generated Periods:", period)

    k = 1
    while k:
        execution_time = execution_time_Generator(task_number, period, target_Uti)

        uti = []
        for i in range(task_number):
            uti.append(execution_time[i] / period[i])
        actual_uti = sum(uti)
        if actual_uti <= target_Uti:
            k = 0

    print("Generated Execution Time:", execution_time)
    print("Actual Utilization:", actual_uti)
    offset = np.zeros(task_number, dtype=int)
    Hyperperiod = hyperperiod_calculation(period)
    print("Hyperperiod:", Hyperperiod)
    print("----------------------------------------- ")

    criticality_factor = 2  # the C(HI) = criticality_factor * C(LO), the factor may be varied from 1 to 5.5
    criticality_proportion = 0.5  # the probability that a generated task is of HI-criticality, the factor may be
    # varied from 0.05 to 0.95

    task_set = task_definition(task_number, period, execution_time, criticality_factor, criticality_proportion)
    # print("-------------------- all tasks in the system--------------------")
    # table_print(task_set)

    # ------- LO_tasks and HI_tasks segmentation and Context aware task selection-----------#

    LO_tasks = []
    HI_tasks = []
    for i in task_set:
        if i.criticality == "LO":
            LO_tasks.append(i)
        else:
            HI_tasks.append(i)

    # ---  Context aware task selection ----#
    LO_tasks = tasks_sort(LO_tasks, 0)
    LO_tasks[0].criticality = "HI"
    LO_tasks[0].context_aware = 1
    LO_tasks[0].execution_time_HI = LO_tasks[0].execution_time_LO * criticality_factor
    HI_tasks.append(LO_tasks[0])
    LO_tasks.remove(LO_tasks[0])

    HI_tasks = tasks_sort(HI_tasks, 0)
    for i in range(len(HI_tasks)):
        HI_tasks[i].app_id = 1680  # specific number
        HI_tasks[i].task_id = i

    print("-------------------- HI tasks in the system--------------------")
    table_print(HI_tasks)

    # --- Tasks dependency generator for LO_tasks--- #

    # step1: LO_application number 
    LO_app_num = 2

    # step2: tasks fragmentation
    LO_apps = {}
    tasks_perapp = math.ceil(len(LO_tasks) / LO_app_num)
    for i in range(LO_app_num):
        LO_apps[i] = sample(LO_tasks, tasks_perapp)
        for j in LO_apps[i]:
            LO_tasks.remove(j)
            if len(LO_tasks) < tasks_perapp:
                tasks_perapp = len(LO_tasks)
    LO_apps_tasks = []
    for i in range(len(LO_apps)):
        LO_apps[i] = tasks_sort(LO_apps[i], i)
        for j in LO_apps[i]:
            LO_apps_tasks.append(j)
    # you can check the tasks of LO_applications in different way
    print("-------------------- LO tasks in the system--------------------")
    table_print(LO_apps_tasks)

    # step3: Tasks dependency definition for LO_apps_tasks (currently assume stream group)


