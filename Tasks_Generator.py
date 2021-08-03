import numpy as np
import random
import math
from copy import deepcopy
from random import sample
import sys
from prettytable import PrettyTable


class Task:
    criticality = ''

    def __init__(self, task_id, job_id, release_time, deadline, period, execution_time_LO, execution_time_HI, priority,
                 importance, criticality, context_aware):
        self.task_id = task_id
        self.job_id = job_id
        self.release_time = release_time
        self.deadline = deadline
        self.period = period
        self.execution_time_LO = execution_time_LO
        self.execution_time_HI = execution_time_HI
        self.priority = priority
        self.importance = importance  # under different mode the value of importance may be varied
        self.criticality = criticality  # task criticality definition
        self.context_aware = context_aware  # true 1, false 0


def period_Generator(stream_number, generated_transmission_times):
    utilizations = uunifast(stream_number)
    # for j in range(len(utilizations)):
    #     print(utilizations[j])

    generated_period = []
    for j in range(stream_number):
        generated_period.append(math.ceil(generated_transmission_times[j] / utilizations[j]))

    # test the total utilization will not exist 1 after ceiling operation
    uti = []
    for i in range(stream_number):
        uti.append(generated_transmission_times[i] / generated_period[i])
    b = sum(uti)
    print(b)

    return generated_period


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
    HI_task_num = math.floor(task_number * criticality_proportion)
    index_init = range(task_number)
    HI_task_index = sample(index_init, HI_task_num)
    print("the index of HI-criticality tasks in one task set:", HI_task_index)
    context_aware_index = sample(HI_task_index, 1)
    print("the index of context-aware task in one task set:", context_aware_index)

    for i in range(task_number):
        if i in HI_task_index:
            criticality = "HI"
            execution_time_HI = execution_time[i] * criticality_factor
        else:
            criticality = "LO"
            execution_time_HI = 0

        if i in context_aware_index:
            context_aware = 1
        else:
            context_aware = 0

        release_time = random.randint(0, hyperperiod_calculation(period))

        task = Task(i, 0, release_time, period[i], period[i], execution_time[i], execution_time_HI, i, i,
                    criticality, context_aware)

        tasks.append(task)
    return tasks


def hyperperiod_calculation(period):
    hyperperiod = period[0]
    for i in range(1, len(period)):
        hyperperiod = hyperperiod * period[i] // math.gcd(hyperperiod, period[i])

    return hyperperiod


if __name__ == "__main__":
    print("Start Tasks Generation")
    print("--------------------------------")

    task_number = 10  # varied from 4 to 60, the task of one task set
    target_Uti = 0.5  # varied from 0.05 to 0.95

    period_set = [10, 50, 100, 200, 500, 1000]
    idxs = np.random.randint(0, len(period_set), size=task_number)

    period = []
    for i in idxs:
        period.append(period_set[i])
    print("Generated Periods:", period)

    k = 1
    execution_time = []
    actual_uti = 0
    while k:
        execution_time = execution_time_Generator(task_number, period, target_Uti)

        uti = []
        for i in range(task_number):
            uti.append(execution_time[i] / period[i])
        actual_uti = sum(uti)
        if actual_uti < 0.5:
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
    table = PrettyTable(
        ['task_id', 'job_id', 'release_time', 'deadline', 'period', 'execution_time_LO', 'execution_time_HI', 'priority',
         'importance', 'criticality', 'context-aware:'])
    for i in task_set:
        table.add_row(
            [i.task_id, i.job_id, i.release_time, i.deadline, i.period, i.execution_time_LO, i.execution_time_HI,
             i.priority, i.importance, i.criticality, i.context_aware])
    print(table)
