import numpy as np
import random
import math
from random import sample
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

        # release_time = random.randint(0, hyperperiod_calculation(period))
        release_time = 0

        task = Task(i, 0, release_time, period[i], period[i], execution_time[i], execution_time_HI, i, i,
                    criticality, context_aware)

        tasks.append(task)
    return tasks


def hyperperiod_calculation(period):
    hyperperiod = period[0]
    for i in range(1, len(period)):
        hyperperiod = hyperperiod * period[i] // math.gcd(hyperperiod, period[i])

    return hyperperiod


def table_print(tasks):
    table = PrettyTable(
        ['task_id', 'job_id', 'release_time', 'deadline', 'period', 'execution_time_LO', 'execution_time_HI',
         'priority',
         'importance', 'criticality', 'context-aware:'])
    for i in tasks:
        table.add_row(
            [i.task_id, i.job_id, i.release_time, i.deadline, i.period, i.execution_time_LO, i.execution_time_HI,
             i.priority, i.importance, i.criticality, i.context_aware])
    print(table)


if __name__ == "__main__":
    print("Start Tasks Generation")
    print("--------------------------------")

    task_number = 10  # varied from 4 to 60, the task of one task set
    target_Uti = 0.5  # varied from 0.05 to 0.95

    period_set = [10, 50, 100, 200, 500, 1000]
    period = []
    for i in range(task_number):
        period.append(random.choice(period_set))
    print("Generated Periods:", period)

    execution_time = execution_time_Generator(task_number, period, target_Uti)

    uti = []
    for i in range(task_number):
        uti.append(execution_time[i] / period[i])
    actual_uti = sum(uti)

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
    table_print(task_set)

    LO_tasks = []
    HI_tasks = []
    for i in task_set:
        if i.criticality == "LO":
            LO_tasks.append(i)
        else:
            HI_tasks.append(i)

    # table_print(LO_tasks)
    # table_print(HI_tasks)

    # --- tasks dependency generator for LO_tasks--- #

    LO_app1 = sample(LO_tasks, math.ceil(len(LO_tasks)/2))
    for i in LO_app1:
        LO_tasks.remove(i)
    LO_app2 = LO_tasks
    table_print(LO_app1)
    table_print(LO_app2)

