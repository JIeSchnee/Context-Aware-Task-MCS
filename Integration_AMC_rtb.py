import copy
import os
import random
import sys
from random import choice
import operator
from itertools import chain
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from prettytable import PrettyTable
from functools import reduce
from RTA_priority_definition import priority_recursive
from Sensitivity_Analysis_AMC_rtb import response_time_HI_SA, Sensitivity_Analysis_LO


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Task:
    # TODO: the properties will be extended to include more information for the scheduling problem
    # TODO: Pay attention to the order of conditions. When necessary, the TabularCPD.evidence order should be defined
    #  independently

    def __init__(self, task, cpd, criticality, deadline, period, execution_time_LO, execution_time_HI, priority,
                 importance):
        self.task = task  # task name
        self.cpd = cpd
        self.criticality = criticality
        self.deadline = deadline
        self.period = period
        self.execution_time_LO = execution_time_LO
        self.execution_time_HI = execution_time_HI
        self.priority = priority
        self.importance = importance


class APP:
    def __init__(self, app_name, taskset, keynode, All_tasks):
        self.app_name = app_name
        self.taskset = taskset
        self.keynode = keynode
        self.All_tasks = All_tasks


def load_task(task_idx, dag_base_folder="/home/jiezou/Documents/Context_aware MCS/dag-gen-rnd-master/data"):
    # << load DAG task <<
    dag_task_file = dag_base_folder + "Tau_{:d}.gpickle".format(task_idx)

    # task is saved as NetworkX gpickle format
    G = nx.read_gpickle(dag_task_file)

    # formulate the graph list
    G_dict = {}
    C_dict = {}
    V_array = []
    T = G.graph["T"]

    max_key = 0
    for u, v, weight in G.edges(data='label'):
        if u not in G_dict:
            G_dict[u] = [v]
        else:
            G_dict[u].append(v)

        if v > max_key:
            max_key = v

        if u not in V_array:
            V_array.append(u)
        if v not in V_array:
            V_array.append(v)

        C_dict[u] = weight
    C_dict[max_key] = 1

    G_dict[max_key] = []

    # formulate the c list (c[0] is c for v1!!)
    C_array = []
    for key in sorted(C_dict):
        C_array.append(C_dict[key])

    V_array.sort()
    W = sum(C_array)

    C = G.nodes[max_key]['C']
    # print(C)
    # >> end of load DAG task >>
    return G_dict, V_array, C_dict, C_array, T, W, C


def dictionary_definition():
    H = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    dict = {}
    for i in range(100):
        salt = ''
        for j in range(3):
            salt += random.choice(H)
        dict[i] = salt

    return dict


def parameters_initialisation(dict, dag_base_folder):
    Appset = []
    bias = 0
    network_tasks = []
    network_edges = []
    HI_group = []
    Execution_times = {}
    app_dependency_node = []
    period = {}
    uti = 0
    for app in range(3):
        print("==========================")
        G, V, C, _, T, W, sink = load_task(task_idx=app,
                                           dag_base_folder=dag_base_folder)
        # print("G: ", G)
        # print("V: ", V)
        # print("C: ", C)
        # print("W: ", W)
        # print("ET", T)
        sink_search = []

        for i in C.keys():
            sink_search.append(i)

        # print(sink_search[-1])
        C[sink_search[-1]] = sink

        for i in C.keys():
            # print(C[i])
            uti += C[i] / T

        print("Uti check", uti)

        edges = []
        for i in range(len(V)):
            for j in range(len(G[V[i]])):
                pair = (dict[V[i] + bias], dict[G[V[i]][j] + bias])
                edges.append(pair)

            V[i] += bias

        tempKey = {}
        temp_priod = {}
        for i in C.keys():
            y = i + bias
            tempKey[dict[y]] = C[i]
            temp_priod[dict[y]] = T
            # print(y)

        # print("faefafgaa", tempKey)

        bias += len(V)

        for i in tempKey:
            Execution_times[i] = tempKey[i]

        for i in temp_priod:
            period[i] = temp_priod[i]

        # Generate the relationship
        # print("V: ", V)
        app_dependency_node.append(dict[choice(V[0:-2])])

        # print("The tasks of current application", V)
        # print("The edges of current application", edges)

        for i in range(len(V)):
            V[i] = dict[V[i]]

        tempApp = APP(app, V, V[-1], V)
        Appset.append(tempApp)

        for task in V:
            network_tasks.append(task)

        for edge in edges:
            network_edges.append(edge)

        # define the criticality of each task.
        # the key node should be HI.
        # At least one parent nodes of HI task should be HI
        # V : all tasks of current application
        # edges : all edges of current application

        seed = V[-1]
        HI_group.append(seed)
        Hi_temp = [seed]
        HI_candidate = []
        edges_test = edges
        # print("Original edges", edges_test)
        tt = 1
        while tt:
            remove_temp = []
            # print("updated edges", edges_test)
            for i in range(len(edges_test)):
                if edges_test[i][1] == seed:
                    HI_candidate.append(edges_test[i][0])
                    remove_temp.append(edges_test[i])
            # print("remove", remove_temp)
            edges_test = [x for x in edges_test if x not in remove_temp]

            if HI_candidate:
                seed = choice(HI_candidate)
                # print("candidates", HI_candidate)
                # print("new seed", seed)
                HI_group.append(seed)
                Hi_temp.append(seed)
                HI_candidate = []
            else:
                print("Hi tasks in application", app, ":", Hi_temp)
                tt = 0
    print("===================================================")
    print("Hi tasks in the network ", HI_group)
    print("===================================================")

    # print("All tasks in the network", network_tasks)
    # print("All edges in the network", network_edges)

    print(app_dependency_node)
    for i in range(len(app_dependency_node)):
        if i + 1 < len(app_dependency_node):
            network_edges.append((app_dependency_node[i], app_dependency_node[i + 1]))
    # print("All edges in the network", network_edges)

    for i in Appset:
        Droppable_Tasks = []
        for x in i.taskset:
            if x in HI_group:
                # print("not droppable")
                continue
            else:
                Droppable_Tasks.append(x)

        i.taskset = copy.deepcopy(Droppable_Tasks)
        print("Application ID:", i.app_name, '\n', "The droppable tasks in the application", i.taskset, '\n',
              "The keynode of the application", i.keynode)

    return network_edges, network_tasks, Appset, HI_group, Execution_times, period


def value_generation(network_tasks):
    values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, len(network_tasks))),
                          columns=network_tasks)
    return values


def initialisation(network_edges, network_tasks, Appset, HI_group, values, Execution_times, period):
    model = BayesianNetwork(network_edges)
    # values = pd.DataFrame(np.random.randint(low=0, high=2, size=(100, len(network_tasks))),
    #                       columns=network_tasks)
    model.fit(values)
    model.get_cpds()
    dict = dictionary_definition()
    Tasks = []
    # print("fdffaf", period)

    for i in network_tasks:
        if i in HI_group:
            criticality = 'HI'
            for j in Execution_times.keys():
                if i == j:
                    execution_base = Execution_times[j]
                    execution_time_LO = execution_base
                    execution_time_HI = 2 * execution_base
        else:
            criticality = 'LO'
            for j in Execution_times.keys():
                if i == j:
                    execution_base = Execution_times[j]
                    execution_time_LO = execution_base
                    execution_time_HI = 0

        for l in period.keys():
            if i == l:
                T_period = period[l]
                T_deadline = period[l]

        Tasks.append(Task(model.get_cpds(i).variable, model.get_cpds(i), criticality, T_deadline, T_period,
                          execution_time_LO, execution_time_HI, -1, -1))

    return Tasks, model, Appset, HI_group


def table_Reconstruction(cpd, dropped_task):
    print("current treated table:", cpd.variable)
    print("Before modification", cpd.variables)
    # print(cpd.values.shape, '\n')
    print("current moved task", dropped_task)

    task = cpd.variable
    print("CHECK POINT: pay attention to the evidence order")
    evidence_task = list(cpd.get_evidence())
    evidence_task.reverse()
    print("evidence_task", evidence_task)
    dropped_index = evidence_task.index(dropped_task)
    evidence_task.pop(dropped_index)
    updated_cpd_evidence = evidence_task
    print("remained evidence tasks:", updated_cpd_evidence)

    extract_index = cpd.variables.index(dropped_task)
    # print("extract_index", extract_index)
    # print("the original cpd", cpd)
    # print(cpd.values)

    test = cpd.values
    updated_cpd_values = np.delete(test, 0, axis=extract_index)
    updated_cpd_values = np.reshape(updated_cpd_values, (2, -1))
    # print(updated_cpd_values)

    updated_cpd_evidence_card = []
    for i in range(len(updated_cpd_evidence)):
        updated_cpd_evidence_card.append(2)

    new_cpd = TabularCPD(variable=task, variable_card=2,
                         values=updated_cpd_values,
                         evidence=updated_cpd_evidence,
                         evidence_card=updated_cpd_evidence_card)
    # print("the original cpd", cpd)
    # print("the updated cpd", new_cpd)

    return new_cpd


def modified_Task(dropped_task, model):
    modified_task = []
    evidence = model.get_cpds(dropped_task).get_evidence()
    edges = model.edges
    for i in edges:
        if dropped_task in i:
            for j in i:
                if j != dropped_task and j not in evidence:
                    modified_task.append(j)

    return modified_task


def task_Drop(dropped_task, model, Tasks):
    # single task drop (not a set)
    modified_task = modified_Task(dropped_task, model)
    print("Current treated task:", dropped_task)
    print("Related tasks, whose cpd need to be modified:", modified_task, '\n')
    tau = []
    for i in range(len(Tasks)):
        tau.append(Tasks[i].task)
    # print(tau)

    for i in modified_task:
        print("^^^^^^^^^^^^^^^^^^^^^^^^", '\n')
        print("Start to update CPD:", i)
        # print(Tasks[tau.index(i)].cpd)
        Tasks[tau.index(i)].cpd = table_Reconstruction(Tasks[tau.index(i)].cpd, dropped_task)
        print("updated table", Tasks[tau.index(i)].cpd)

    return Tasks


def Task_Dropping_Test(dropped_task_set, Tasks1, model1, keynode):
    # print("--- Network Re-initialisation ---")
    # Tasks1 = parameters_initialisation()
    tau_assump1 = []
    for i in range(len(Tasks1)):
        tau_assump1.append(Tasks1[i].task)
    # print("----- Bayesian Network setup ------")
    # model1 = model_initialisation(Tasks1)

    # cpds = model1.get_cpds()
    # for cpd in cpds:
    #     print(f'CPT of {cpd.variable}:')
    #     print(cpd, '\n')

    print("START TO DROP TASKS UNDER ASSUMPTION")

    print("The dropped tasks:", dropped_task_set, '\n')

    print("Update the CPDs in of the network", '\n')
    print("********************************************")
    print(model1.nodes)
    for i in dropped_task_set:
        Tasks1 = task_Drop(i, model1, Tasks1)

    print("The original tasks in the Bayesian network:", model1.nodes)

    for i in dropped_task_set:
        model1.remove_node(i)
    # model1.remove_node('tau3')
    # model1.remove_node('tau2')
    updated_network_nodes = model1.nodes
    print("Updated tasks in the Bayesian network:", updated_network_nodes)
    # print("the edges", model1.edges)
    print("The CPDs should be attached to the updated network")

    # check the attached CPDs after network update

    for i in updated_network_nodes:
        # print(Tasks1[tau_assump1.index(i)].task)
        model1.add_cpds(Tasks1[tau_assump1.index(i)].cpd)

    # check the correctness of network and select the calculation method (e.g., VariableElimination method)
    model1.get_cpds()

    # cpds = model1.get_cpds()
    # for cpd in cpds:
    #     print(f'CPT of {cpd.variable}:')
    #     print(cpd, '\n')

    infer = VariableElimination(model1)

    print("Select the key nodes and calculate corresponding Marginal Probability", keynode)

    key_nodes = keynode
    # key_nodes = ['tau5', 'tau7']

    marginal_prob_set = []
    print("key_nodes", key_nodes)
    # cpds = model1.get_cpds()
    # for cpd in cpds:
    #     print(f'CPT of {cpd.variable}:')
    #     print(cpd, '\n')

    # print(dict)

    for i in key_nodes:
        # print(model1.get_cpds(i).get_evidence())
        marginal_prob = infer.query([i])
        marginal_prob_set.append(marginal_prob)
        # print(marginal_prob)

    return marginal_prob_set


def global_Expected_Utility(marginal_prob_set):
    # TODO: the definition can be improved with the consideration of the safety-related elements.

    EU_global = 1
    for i in marginal_prob_set:
        EU_global *= i.values[0]
    return EU_global


def Application_drop_test(App_name, dropped_task_set, EU_global_set, Dropped_APPs, App_drop_tasks, model,
                          Tasks, keynode):
    # model_copy = model.copy()
    # Tasks_copy = Tasks

    marginal_prob_set1 = Task_Dropping_Test(dropped_task_set, Tasks, model, keynode)

    EU_global_set.append(global_Expected_Utility(marginal_prob_set1))
    Dropped_APPs.append(App_name)
    App_drop_tasks.append(dropped_task_set)

    print("The global expected utility of assumption ( drop", App_name, "):", '\n',
          global_Expected_Utility(marginal_prob_set1), '\n')

    return EU_global_set, Dropped_APPs, App_drop_tasks


def remove_task(Tasks, dropped_task):
    for i in range(len(Tasks)):
        if Tasks[i].task == dropped_task:
            Tasks.remove(Tasks[i])
            break
    return Tasks


def remove_app(Appset, dropped_app):
    # print(dropped_app)
    for i in Appset:
        if i.app_name == dropped_app:
            Appset.remove(i)
    print("Remove APP succeed")
    return Appset


def model_task_copy(model_original, tasks_name_index, HI_group, Tasks_original):
    model_copy = copy.deepcopy(model_original)
    # model_copy = model_original.copy()
    temp_task = copy.deepcopy(Tasks_original)
    Tasks_copy = []
    for j in tasks_name_index:
        for i in temp_task:
            if j == i.task:
                i.cpd = model_copy.get_cpds(j)
                Tasks_copy.append(i)

    return model_copy, Tasks_copy


def Application_drop_and_update(Tasks_pre, model_pre, Appset, HI_group, keynode):
    EU_global_set = []
    Dropped_APPs = []
    App_drop_tasks = []
    tasks_name_index = []
    for j in Tasks_pre:
        tasks_name_index.append(j.task)
    # print("$$$$$", tasks_name_index)

    for i in range(len(Appset)):
        print("====================App round=========================", '\n')
        print("---Assumption:", "drop", Appset[i].app_name, '\n')

        model_copy, Tasks_copy = model_task_copy(model_pre, tasks_name_index, HI_group, Tasks_pre)

        dropped_task_set = Appset[i].taskset
        print("current node:", model_copy.nodes(), '\n', "length:", len(model_copy.nodes()))

        # cpds = model_copy.get_cpds()
        # for cpd in cpds:
        #     print(f'CPT of {cpd.variable}:')
        #     print(cpd, '\n')

        EU_global_set, Dropped_APPs, App_drop_tasks = Application_drop_test(Appset[i].app_name,
                                                                            dropped_task_set,
                                                                            EU_global_set, Dropped_APPs,
                                                                            App_drop_tasks,
                                                                            model_copy, Tasks_copy, keynode)
    print('\n', "#################################################", '\n')
    print("---Application discarding decision---", '\n')
    assumption_ID = EU_global_set.index(max(EU_global_set))
    EU_value = max(EU_global_set)
    print("Task dropping start from: ", Dropped_APPs[assumption_ID], '(', App_drop_tasks[assumption_ID], ')', '\n')
    print(EU_value)
    print("#################################################", '\n')
    print("-----Network update and look for the next dropped App--- ")
    model_copy, Tasks_copy = model_task_copy(model_pre, tasks_name_index, HI_group, Tasks_pre)
    for i in Appset:
        if i.app_name == Dropped_APPs[assumption_ID]:
            print(i.app_name, i.taskset)
            dropped_task_set = i.taskset
    marginal_prob_set = Task_Dropping_Test(dropped_task_set, Tasks_copy, model_copy, keynode)
    # print(model_copy.nodes)

    # for i in Tasks_copy:
    #     print(i.task)

    # print("fdfaffddddddddddddd", dropped_task_set)

    if dropped_task_set:
        for i in dropped_task_set:
            Tasks = remove_task(Tasks_copy, i)
    else:
        Tasks = Tasks_copy

    Appset = remove_app(Appset, Dropped_APPs[assumption_ID])

    print("dropped app and tasks:", Dropped_APPs[assumption_ID], dropped_task_set)
    print("remain App", len(Appset))
    print("rest nodes", model_copy.nodes)

    # for i in Tasks:
    #     print(i.task)

    return model_copy, Tasks, Appset, Dropped_APPs[assumption_ID], EU_value


def app_task_drop_test(app, Test_task_set, model_pre, tasks_name_index, HI_group, Appset, Keynode, Tasks_pre):
    EU_local_set = []
    Dropped_task = []

    # if Test_task_set:
    # print(model.nodes)
    # cpds = model.get_cpds()
    # for cpd in cpds:
    #     print(f'CPT of {cpd.variable}:')
    #     print(cpd, '\n')

    for i in range(len(Test_task_set)):
        # print(Test_task_set)
        print("The task set of corresponding app:", Test_task_set)
        print("=======================================================", '\n')
        print("*****Current tested task****", Test_task_set[i])

        model, Tasks = model_task_copy(model_pre, tasks_name_index, HI_group, Tasks_pre)
        print(model.nodes)

        dropped_task_set = []
        dropped_task_set.append(Test_task_set[i])
        marginal_prob_set = Task_Dropping_Test(dropped_task_set, Tasks, model, Keynode)
        print(Keynode)
        for j in marginal_prob_set:
            # print(j.variables)
            # print(j)
            if j.variables[0] == Keynode[0]:
                marginal_prob_key = j
        print("***** Marginal Distribution of Key node ****", '\n')
        print(marginal_prob_key)
        EU_local = 1
        for j in marginal_prob_set:
            EU_local *= j.values[0]
        print(EU_local)

        # local_Expected_Utility = marginal_prob_key.values[0]
        local_Expected_Utility = EU_local
        # TODO: the definition can be improved with the consideration of the safety-related elements.
        EU_local_set.append(local_Expected_Utility)
        Dropped_task.append(Test_task_set[i])
        # Test_task_set.remove(i)

    return EU_local_set, Dropped_task


def Task_drop_and_update(app, Tasks_pre, model_pre, Appset, HI_group, Test_task_set, Task_drop_order, keynode):
    tasks_name_index = []
    for j in Tasks_pre:
        tasks_name_index.append(j.task)
    print("Existing tasks:", tasks_name_index)

    print("Start Task Dropping Test", '\n')

    EU_local_set, Dropped_task = app_task_drop_test(app, Test_task_set, model_pre, tasks_name_index, HI_group, Appset,
                                                    keynode, Tasks_pre)

    print('\n', "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", '\n')
    print("Task discarding decision:")
    task_Drop_ID = EU_local_set.index(max(EU_local_set))
    print("Task dropping start from: ", app, Dropped_task[task_Drop_ID], '\n')
    print(EU_local_set)
    mrvalue = max(EU_local_set)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", '\n')
    print("-----Network update and look for the next dropped task--- ")
    model, Tasks = model_task_copy(model_pre, tasks_name_index, HI_group, Tasks_pre)
    Appset = Appset_original
    print(model.nodes)
    dropped_task_set = []
    dropped_task_set.append(Dropped_task[task_Drop_ID])
    marginal_prob_set = Task_Dropping_Test(dropped_task_set, Tasks, model, keynode)
    for i in dropped_task_set:
        Tasks = remove_task(Tasks, i)

    Test_task_set.remove(Dropped_task[task_Drop_ID])
    Task_drop_order.append(Dropped_task[task_Drop_ID])

    return model, Tasks, Appset, Task_drop_order, Test_task_set, mrvalue


def table_print(tasks):
    table = PrettyTable(
        ['task_id', 'deadline', 'period', 'execution_time_LO', 'execution_time_HI',
         'priority', 'importance', 'criticality'])
    for i in tasks:
        table.add_row(
            [i.task, i.deadline, i.period, i.execution_time_LO, i.execution_time_HI,
             i.priority, i.importance, i.criticality])
    print(table)


def Sensitivity_Analysis(Test_tasks):
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
    temp_test = copy.deepcopy(Test_tasks)
    for i in temp_test:
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

        print('\n', "---------- Sensitivity Analysis LO -------------")
        # count_LOSA = 0
        while 1:

            sati = Sensitivity_Analysis_LO(Test_tasks, Dropped, overrun)
            if sati == 1:
                print("One Low task should be dropped once overrun happens. Restart to check the schedulability of LO "
                      "mode")
                table_print(Dropped[0])
            else:
                # count_LOSA += 1
                print("==============================================", '\n')
                print(" In LO mode in current overrun is schedulable and start to check the shedulability of system MC")

                break
                # if count_LOSA == len(Test_tasks):
                #     break

        print('\n', "---------- Sensitivity Analysis HI (MC)-------------")

        table_print(HI_task_set)
        start = 1
        num_check1 = 0

        while start:
            print('\n', "++++++++++++++++ Restart +++++++++++++++++++")
            table_print(Dropped[0])
            print(Dropped[1])

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

                with HiddenPrints():
                    sati = Sensitivity_Analysis_LO(Test_tasks, Dropped, overrun)
                sati_HI = 1
                for h in Test_tasks:
                    with HiddenPrints():
                        response_timeMC, sati_HI = response_time_HI_SA(h, Test_tasks, Dropped, overrun)
                    if sati_HI == 1:
                        print("attention!!!!!")
                        break

                if sati == 1 and sati_HI == 0:
                    print("All droppable tasks have already been dropped and system is scheduled", '\n',
                          "(Not all Low tasks need to be dropped)")
                    table_print(Dropped[0])
                    print(Dropped[1])
                    break

            for task in HI_task_set:
                print("#########################################", '\n')
                print("current tested HI task", task.task)
                if HI_count > len(HI_task_set):
                    sati_HI = 1
                    break
                else:
                    print("Already dropped tasks")
                    table_print(Dropped[0])
                    # print(HI_count)
                    print("+++++++++++++++ task", task.task, "+++++++++++++++++++")
                    with HiddenPrints():
                        response_timeMC, sati_HI = response_time_HI_SA(task, Test_tasks, Dropped, overrun)
                    if sati_HI == 1:
                        Increace_mark = 0
                        break
                    elif sati_HI == 0:
                        Increace_mark = 1
                        HI_count += 1
                        # start = 0
                        print("The number of already tested HI tasks", HI_count)
                        print("The dropping task test of HI task:", task.task, "is finished", '\n')

            if Increace_mark == 1 and HI_count == len(HI_task_set):
                print("Increase the overrun")
                start = 0

        overrun_con += 1

    print('\n', "++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Final result:")
    return Dropped


def Importance_OPA(Tasks_original):
    Tasks_PR_OPA_IP = copy.deepcopy(Tasks_original)

    ####TODO: delete this part
    # for i in Tasks_PR_OPA_IP:
    #     i.importance = random.randint(0, 100)

    Tasks_PR_OPA_IP = sorted(Tasks_PR_OPA_IP, key=lambda Task: Task.importance, reverse=True)
    table_print(Tasks_PR_OPA_IP)

    priority_level = len(Tasks_PR_OPA_IP)
    unschedulable = []
    for i in Tasks_PR_OPA_IP:
        # i.priority = -1
        unschedulable.append(i.task)

    priority_temp = priority_level

    conti = 1

    while conti:
        print("*********************")
        num = 0
        # table_print(Test_tasks)
        for i in Tasks_PR_OPA_IP:
            if i.priority == -1:
                num += 1
        if num == 0:
            break
        else:
            print("The allocated priority level:", priority_temp)
            with HiddenPrints():
                priority_temp, count = priority_recursive(priority_temp, Tasks_PR_OPA_IP)
            if count == len(Tasks_PR_OPA_IP):
                print("System unscheduled")
                schedulability_mark = 0
                break
            else:
                schedulability_mark = 1

    return Tasks_PR_OPA_IP, schedulability_mark


def Standard_OPA(Tasks_original):
    Tasks_PR_OPA = copy.deepcopy(Tasks_original)

    # for i in Tasks_PR_OPA:
    #     i.importance = random.randint(0, 100)

    priority_level = len(Tasks_PR_OPA)
    unschedulable = []
    for i in Tasks_PR_OPA:
        # i.priority = -1
        unschedulable.append(i.task)

    priority_temp = priority_level

    conti = 1

    while conti:
        print("*********************")
        num = 0
        # table_print(Test_tasks)
        for i in Tasks_PR_OPA:
            if i.priority == -1:
                num += 1
        if num == 0:
            break
        else:
            print("The allocated priority level:", priority_temp)
            with HiddenPrints():
                priority_temp, count = priority_recursive(priority_temp, Tasks_PR_OPA)
            # print("ouikhjkh",count, len(Tasks_PR_OPA))

            if count == len(Tasks_PR_OPA):
                print("System unscheduled")
                break

    return Tasks_PR_OPA


def Degradation_order(Tasks_original, model_original, Appset_original, HI_group):
    Appset_backpack = []
    Appset = Appset_original
    keynode = []
    for t in Appset:
        keynode.append(t.keynode)
        Appset_backpack.append(t)
        # print("GGGGGGG", t.taskset)

    Appset_task = copy.deepcopy(Appset_original)

    for j in Tasks_original:
        tasks_name_index.append(j.task)
    # print("tasks_name_index", tasks_name_index)
    model, Tasks = model_task_copy(model_original, tasks_name_index, HI_group, Tasks_original)

    print("--------------------------------------------------------", '\n')
    print("--- Application discarding order ---", '\n')
    print("--------------------------------------------------------", '\n')
    EU_set = []
    while len(Appset) >= 1:
        model, Tasks, Appset, Dropped_APP, EU_value = Application_drop_and_update(Tasks, model, Appset, HI_group,
                                                                                  keynode)
        App_drop_order.append(Dropped_APP)
        EU_set.append(EU_value)

    for i in Appset:
        # print(i.app_name)
        App_drop_order.append(i.app_name)

    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')
    # print("Application discarding order:", App_drop_order)
    # print("The EU value variation", EU_set)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')

    print("--------------------------------------------------------", '\n')
    print("--- Determine Task dropping order ---", '\n')
    print("--------------------------------------------------------", '\n')

    model, Tasks = model_task_copy(model_original, tasks_name_index, HI_group, Tasks_original)
    # print(model.nodes)
    Task_drop_order = []
    print(App_drop_order)
    mgr_values = []
    for app in App_drop_order:
        temp_order = []
        print("================== App based Task Round =======================", '\n')
        print("Start to determine the task dropping order of application:", app)
        # with HiddenPrints():
        #     Appset = parameters_initialisation(dict)[2]
        Appset = Appset_task
        APP_check = copy.deepcopy(Appset)
        for t in Appset:
            # print("TTTTT", t.taskset)
            keynode.append(t.keynode)

        Test_task_set = []
        keynode = []
        for t in Appset:
            # print(t.app_name, t.taskset)
            keynode.append(t.keynode)
            if t.app_name == app:
                app_keynode = t.keynode
                Test_task_set = t.taskset

        print("Droppable tasks", Test_task_set, '\n', "Keynode:", app_keynode)

        # keynode.append(app_keynode)
        check = copy.deepcopy(Test_task_set)
        if Test_task_set == []:
            mgr_values.append(665)

        while Test_task_set:
            model, Tasks, Appset, temp_order, Test_task_set, marginal_value = Task_drop_and_update(app, Tasks, model,
                                                                                                   Appset,
                                                                                                   HI_group,
                                                                                                   Test_task_set,
                                                                                                   temp_order,
                                                                                                   keynode)
            mgr_values.append(marginal_value)

        Task_drop_order.append(temp_order)

    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')
    # print("Application discarding order:", App_drop_order)
    # print("The EU value variation", EU_set)
    # print("Application discarding order:", Task_drop_order)
    # print("The EU value variation", mgr_values)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')

    importance = []
    for i in Task_drop_order:
        for j in i:
            importance.append(j)
    # print(importance)
    # print(len(importance))
    # table_print(Tasks_original)
    for i in range(len(importance)):
        for task in Tasks_original:
            if task.task == importance[i]:
                task.importance = len(importance) - i
            elif task.importance == -1:
                task.importance = 0

    return App_drop_order, Task_drop_order, EU_set, mgr_values


def PurTask_Degradation_order(Tasks_original, model_original, Appset_original, HI_group):
    Appset_backpack = []
    keynode = []
    Appset = copy.deepcopy(Appset_original)
    for t in Appset:
        keynode.append(t.keynode)
        Appset_backpack.append(t)
        # print("GGGGGGG", t.taskset)

    Appset_task = copy.deepcopy(Appset_original)

    for j in Tasks_original:
        tasks_name_index.append(j.task)
    # print("tasks_name_index", tasks_name_index)
    model, Tasks = model_task_copy(model_original, tasks_name_index, HI_group, Tasks_original)

    mgr_values = []
    temp_order = []

    for t in Appset:
        # print("TTTTT", t.taskset)
        keynode.append(t.keynode)

    Test_task_set = []
    for t in Appset:
        for i in t.taskset:
            Test_task_set.append(i)

    print("Droppable tasks", Test_task_set)
    app = 78787879
    while Test_task_set:
        model, Tasks, Appset, temp_order, Test_task_set, marginal_value = Task_drop_and_update(app, Tasks, model,
                                                                                               Appset,
                                                                                               HI_group, Test_task_set,
                                                                                               temp_order,
                                                                                               keynode)
        mgr_values.append(marginal_value)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')
    print("Final result:")
    print("Task degradation order", temp_order, '\n')
    print(mgr_values)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')

    return temp_order, mgr_values


if __name__ == "__main__":

    path = "/home/jiezou/Documents/Context_aware MCS/dag-gen-rnd-master/data/data-multi-m4-u0.9/"
    O_system_uti = []
    O_survive = []
    O_Alan_remain = []
    O_tolerant_overrun = []
    O_final_EU = []
    O_Alan_tolerant_overrun = []
    O_Alan_final_EU = []
    O_Droppable_Tasks_set = []
    EU_difference_holistic = []
    D_size_holistic = []
    graph_size = []
    file_name = []
    O_Dropped_diff = []
    O_Dropped_diff_raw = []
    O_BBN_sur = []
    O_Alan_sur = []
    O_Drop_all = []

    for file in os.listdir(path):
        print("####### file number", file)

        test_round = 30
        hope_num = 30
        system_uti = []
        survive = []
        Alan_remain = []
        tolerant_overrun = []
        final_EU = []
        Alan_tolerant_overrun = []
        Alan_final_EU = []
        Droppable_Tasks_set = []
        EU_difference = []
        D_size = []
        Dropped_diff = []
        test_file = []
        BBN_sur = []
        Alan_sur = []
        Drop_all = []

        while test_round:
            print("%%%%%%%%%%  NEW TEST ROUND %%%%%%%%%%%%%%%,", test_round)
            # file = "1"
            dag_base_folder = path + file + "/"
            print("Current folder", dag_base_folder)
            App_drop_order = []
            App_drop_task_order = []
            tasks_name_index = []
            dict = dictionary_definition()
            Droppable_Tasks = []
            margnial_variation = []
            Alan_margnial_variation = []

            network_edges, network_tasks, Appset, HI_group, Execution_times, period = parameters_initialisation(dict,
                                                                                                                dag_base_folder)
            values = value_generation(network_tasks)
            Tasks_original, model_original, Appset_original, HI_group = initialisation(network_edges, network_tasks,
                                                                                       Appset,
                                                                                       HI_group, values,
                                                                                       Execution_times,
                                                                                       period)

            # table_print(Tasks_original)
            size = len(Tasks_original)
            mgr_model = copy.deepcopy(model_original)
            mgr_Tasks = copy.deepcopy(Tasks_original)

            ALan_mgr_model = copy.deepcopy(model_original)
            Alan_Taskset = copy.deepcopy(Tasks_original)
            Alan_App = copy.deepcopy(Appset)

            for i in Appset:
                Droppable_Tasks.append(i.taskset)
            Droppable_Tasks = reduce(operator.add, Droppable_Tasks)
            print("============================================")
            print("Droppable tasks:", Droppable_Tasks)

            uti = 0
            for i in Tasks_original:
                uti += float(i.execution_time_LO) / i.period

            print("System Utilisation", uti)

            system_uti.append(uti)

            # if bias <= 0.1:

            with HiddenPrints():
                App_drop_order, Task_drop_order, App_EU, Task_EU = Degradation_order(Tasks_original, model_original,
                                                                                     Appset_original, HI_group)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')
            print("Final result:")
            print("Application discarding order:", App_drop_order)
            print("The EU value variation", App_EU)
            print("Application discarding order:", Task_drop_order)
            print("The EU value variation", Task_EU)

            order_check = 0
            temp_V = App_EU[0]
            for i in App_EU:
                if i <= temp_V and Task_EU[0] >= App_EU[0]:
                    order_check = 1
                    temp_V = i
                else:
                    order_check = 0
                    print("CPTs regeneration", order_check)
                    break

            if order_check:
                print("+++++++++++++++++ Output the table of tasks with Importance definition ++++++++++++++++++++",
                      '\n')
                Droppable_Tasks_set.append(len(Droppable_Tasks))
                D_size.append(len(Droppable_Tasks))
                # print(Droppable_Tasks_set)
                importance = []
                for i in Task_drop_order:
                    for j in i:
                        importance.append(j)
                # print(importance)
                # print(len(importance))
                # table_print(Tasks_original)
                for i in range(len(importance)):
                    for task in Tasks_original:
                        if task.task == importance[i]:
                            task.importance = len(importance) - i
                        elif task.importance == -1:
                            task.importance = 0
                # table_print(Tasks_original)

                print('\n', "######### Priority definition ##########", '\n')

                with HiddenPrints():
                    Tasks_PR_OPA_IP, schedulability_mark = Importance_OPA(Tasks_original)  # sorted by importance
                    # Tasks_PR_OPA_IP = Standard_OPA(Tasks_original) # standard method without importance emphasizing
                table_print(Tasks_PR_OPA_IP)
                Tasks_IP_Alan = copy.deepcopy(Tasks_PR_OPA_IP)

                if schedulability_mark:
                    print('\n', "######### Sensitivity Analysis ##########", '\n')

                    Test_tasks = copy.deepcopy(Tasks_PR_OPA_IP)
                    # table_print(Test_tasks)
                    with HiddenPrints():
                        Dropped = Sensitivity_Analysis(Test_tasks)

                    print("The dropped task:")
                    table_print(Dropped[0])
                    print("System switch point:", '\n', Dropped[1])
                    # print("The interference bound of dropped task:", '\n', Dropped[2])
                    print("System overrun before update:", '\n', Dropped[3])
                    if Dropped[3]:
                        temp_overrun = Dropped[3][0]
                        for i in range(len(Dropped[3])):
                            if Dropped[3][i] < temp_overrun:
                                print("eytey", Dropped[3][i])
                                Dropped[3][i] = temp_overrun
                                temp_overrun = Dropped[3][i]
                                print("fasfdagfa", temp_overrun, Dropped[3][i])

                    print("System overrun:", '\n', Dropped[3])
                    # print("Unscheduled tasks:", '\n', Dropped[5])
                    #
                    # for i in range(len(Dropped[0])):
                    #     if Dropped[2][i] != 0:
                    #         print('\n', "if HI task", Dropped[5][i], " with LO_execution time", Dropped[4][i][0],
                    #               "can not finish its execution after", Dropped[2][i], ".", '\n',
                    #               "LO Task", Dropped[0][i].task, "need to be dropped.","\n"
                    #               " However, the system switch point can not later than", Dropped[1][i],
                    #               ", after the release of task with overrun(", Dropped[3][i], ") ")
                    #     else:
                    #         print('\n', "Once overrun", Dropped[3][i], "happens. LO Task", Dropped[0][i].task,
                    #               "need to be dropped directly")
                    #

                    milestone_set = copy.deepcopy(Dropped[3])
                    temp = 999
                    milestone = []
                    for i in milestone_set:
                        if i != temp:
                            milestone.append(i)
                            temp = i
                    # print("iojuiojoj", milestone)

                    drop_group = []
                    drop_app = []
                    for j in milestone:
                        temp = []
                        temp_app = []
                        # print("nhcnn", j)
                        for i in range(len(Dropped[0])):
                            if Dropped[3][i] == j:
                                # print("tdjkj", Dropped[0][i].task)
                                temp.append(Dropped[0][i].task)
                                for k in Alan_App:
                                    if Dropped[0][i].task in k.taskset:
                                        if k.app_name not in temp_app:
                                            temp_app.append(k.app_name)

                        drop_group.append(temp)
                        drop_app.append(temp_app)

                    marginal_maintenance = []
                    # print(drop_group)
                    # print(drop_app)

                    if Dropped[0]:
                        # print(Task_drop_order)
                        # print(Task_EU)
                        Task_search_set = []
                        for i in Task_drop_order:
                            if i:
                                for j in i:
                                    Task_search_set.append(j)
                            else:
                                Task_search_set.append(665)
                        # print("hjhuhihklhgkajk")
                        # print(Task_drop_order)
                        # print(Task_search_set)
                        # print(Task_EU)
                        # print(drop_group)

                        for i in drop_group:
                            temo_t = i[-1]
                            ind = Task_search_set.index(temo_t)
                            # print("index", ind)
                            marginal_maintenance.append(Task_EU[ind])

                        milestone_margin = [milestone,
                                            marginal_maintenance]

                    print("the overrun level", milestone)
                    print("Dropped tasks of corresponding overrun", drop_group)

                    print("probability:", marginal_maintenance)
                    if milestone:
                        tolerant_overrun.append(milestone[-1])
                        final_EU.append(marginal_maintenance[-1])

                    else:
                        tolerant_overrun.append(0)
                        final_EU.append(0)

                    print("The number of tasks, which are survived", len(Droppable_Tasks) - len(Dropped[0]))
                    # table_print(Dropped[0])
                    remain_bbn = copy.deepcopy(Droppable_Tasks)
                    temp_bbn = len(Droppable_Tasks)
                    remain_bbn_check = []

                    for i in drop_group:
                        if temp_bbn > 0:
                            temp_bbn -= len(i)
                        else:
                            temp_bbn = 0
                        remain_bbn_check.append(temp_bbn)

                    for j in Dropped[0]:
                        if j.task in Droppable_Tasks:
                            remain_bbn.remove(j.task)
                    print("The survived tasks", remain_bbn)
                    survive.append(len(remain_bbn))

                    print('\n', "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", '\n')
                    print("The method proposed by Alan's paper")

                    Alan_Droppable_Tasks = copy.deepcopy(Droppable_Tasks)
                    Alan_Dropped = copy.deepcopy(Dropped)
                    remained = copy.deepcopy(Alan_Droppable_Tasks)
                    temp_alan = len(Alan_Droppable_Tasks)
                    # print("Original tasks in the system for Alan's method", remained)

                    Alan_app_drop = []
                    remain_Alan_check = []
                    S = []
                    for i in drop_app:
                        S_t = []
                        for y in i:
                            S_t.append(y)
                        # print("huhiii0", S_t, S)
                        iner = []
                        for d in S_t:
                            if d not in S:
                                iner.append(d)
                        if iner:
                            # print("not", iner)
                            for j in iner:
                                for k in Alan_App:
                                    if j == k.app_name:
                                        temp_alan -= len(k.taskset)
                                        for tt in i:
                                            S.append(tt)
                        remain_Alan_check.append(temp_alan)

                    for i in Alan_Dropped[0]:
                        for j in Alan_App:
                            if i.task in j.taskset and i.task in remained:
                                # print(j.taskset)
                                Alan_app_drop.append(j.app_name)
                                for dp in j.taskset:
                                    remained.remove(dp)
                                break
                    print("The application drop order", Alan_app_drop)

                    marginal_App = []
                    check = []
                    for i in drop_app:
                        temp_app = i[-1]
                        if i[-1] not in check:
                            temp_EU = App_EU[App_drop_order.index(temp_app)]
                            marginal_App.append(temp_EU)
                            for j in i:
                                check.append(j)
                        else:
                            marginal_App.append(marginal_App[-1])

                    print("the overrun level", milestone)
                    print("Dropped Apps of corresponding overrun", drop_app)
                    print("probability:", marginal_App)
                    print("The number of tasks, which are survived", len(remained))
                    print("The finally remained tasks with Alan's method", remained)
                    Alan_remain.append(len(remained))
                    if milestone:
                        Alan_tolerant_overrun.append(milestone[-1])
                        Alan_final_EU.append(marginal_App[-1])
                    else:
                        Alan_tolerant_overrun.append(0)
                        Alan_final_EU.append(0)

                    test_round -= 1

                    # name_list = milestone
                    # list1 = marginal_maintenance  # task level
                    # list2 = marginal_App  # app level
                    #
                    # total_width, n = 0.8, 2
                    # width = total_width / n
                    # x = list(range(len(list1)))
                    # # print(x)
                    # # print(marginal_maintenance)
                    # # print(marginal_App)
                    # plt.bar(x, marginal_maintenance, width=width, label='BBN', tick_label=name_list)
                    # for i in range(len(x)):
                    #     x[i] = x[i] + width
                    # plt.bar(x, marginal_App, width=width, label='Alan', tick_label=name_list)
                    # plt.legend()
                    # plt.show()

                    temp_ED = 0
                    for i in range(len(marginal_maintenance)):
                        temp_ED += ((marginal_maintenance[i] - marginal_App[i]) / marginal_App[i]) * 100

                    if marginal_maintenance:
                        EU_difference.append(temp_ED / len(marginal_maintenance))
                    else:
                        EU_difference.append(0)



                    print("The remained tasks at each dropping point", remain_bbn_check, '\n', remain_Alan_check)

                    BBN_sur.append(remain_bbn_check)
                    Alan_sur.append(remain_Alan_check)
                    Drop_all.append(len(Alan_Droppable_Tasks))

                    for i in range(len(remain_bbn_check)):
                        Dropped_diff.append(
                            ((remain_bbn_check[i] - remain_Alan_check[i]) / len(Alan_Droppable_Tasks)) * 100)
                        # if remain_Alan_check[i]:
                        #     Dropped_diff.append(((remain_bbn_check[i] - remain_Alan_check[i])/remain_Alan_check[i]) * 100)
                        # else:
                        #     Dropped_diff.append((remain_bbn_check[i] - remain_Alan_check[i]) * 100)
                    test_file.append(file)
                else:
                    test_round = 0
                    if EU_difference:
                        if len(EU_difference) != hope_num:
                            test_round = hope_num - len(EU_difference)
                        else:
                            test_round = 0
                    else:
                        print("!!!!!!!!!!!!!!!!!! system unschedulable !!!!!!!!!!!!!!!!!!!!!")
                        test_file.append(555)



        for i in test_file:
            if i != 555:
                file_name.append(i)
                break

        count_555 = 0
        for i in test_file:
            if i == 555:
                count_555 += 1
        if count_555 == len(test_file):
            file_name.append(555)

        count_N5 = 0
        for i in file_name:
            if i != 555:
                count_N5 += 1

        # print("uiyoip", Dropped_diff)

        O_BBN_sur.append(BBN_sur)
        O_Alan_sur.append(Alan_sur)
        O_Drop_all.append(Drop_all)

        O_Dropped_diff.append(np.mean(Dropped_diff))
        O_Dropped_diff_raw.append(Dropped_diff)

        graph_size.append(size)
        D_size_holistic.append(D_size)
        EU_difference_holistic.append(EU_difference)
        O_system_uti.append(np.mean(system_uti))
        O_survive.append(sum(survive))
        O_Alan_remain.append(sum(Alan_remain))
        O_tolerant_overrun.append(np.mean(tolerant_overrun))
        O_final_EU.append(np.mean(final_EU))
        O_Alan_tolerant_overrun.append(np.mean(Alan_tolerant_overrun))
        O_Alan_final_EU.append(np.mean(Alan_final_EU))
        O_Droppable_Tasks_set.append(sum(Droppable_Tasks_set))

        if count_N5 == 10:
            break

    print("=============== Output comparison ================")
    print("The survived tasks based on BBN method", sum(O_survive))
    print("The survived tasks based on Alan's method", sum(O_Alan_remain))

    BBN_num = O_survive
    Alan_num = O_Alan_remain

    total_width, n = 0.8, 1
    width = total_width / n
    x = list(range(len(BBN_num)))

    plt.bar(x, O_Dropped_diff, width=width, tick_label=file_name)

    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, Alan_num, width=width, label='Alan_survived', tick_label=file_name)
    plt.xlabel('The file name')
    plt.ylabel('The difference of average proportion of survived tasks')
    # plt.legend()
    # plt.suptitle("System with Uti 1.0 with")
    plt.show()

    fig1, axes = plt.subplots()
    plt.boxplot(O_Dropped_diff_raw, meanline=True)
    plt.setp(axes, xticklabels=file_name)
    plt.xlabel('The file name')
    plt.ylabel('The distribution of proportion difference of survived tasks')
    # plt.title("System with Uti 1.0 with")
    plt.show()

    print("EU_deviation")
    for i in EU_difference_holistic:
        for j in range(len(i)):
            if i[j] is None:
                i[j] = 0
        print(i)

    name_list3 = []
    EU_mean = []
    for i in range(len(D_size_holistic)):
        name_list3.append(np.mean(D_size_holistic[i]))
        EU_mean.append(np.mean(EU_difference_holistic[i]))

    x = range(len(name_list3))
    plt.bar(x, EU_mean, tick_label=file_name)
    plt.xlabel('The file name')
    plt.ylabel('The average difference of EU value')
    # plt.title("System with Uti 1.0 with")
    plt.show()

    fig, axes = plt.subplots()
    plt.boxplot(EU_difference_holistic, meanline=True)
    plt.setp(axes, xticklabels=file_name)
    plt.xlabel('The file name')
    plt.ylabel('The distribution of EU value difference')
    # plt.title("System with Uti 1.0 with")
    plt.show()

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_BBN_sur_09_30.pickle', 'wb') as handle:
        pickle.dump(O_BBN_sur, handle, protocol=2)
    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Alan_sur_09_30.pickle', 'wb') as handle:
        pickle.dump(O_Alan_sur, handle, protocol=2)
    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Drop_all_09_30.pickle', 'wb') as handle:
        pickle.dump(O_Drop_all, handle, protocol=2)

    with open('/home/jiezou/Documents/Context_aware MCS/L_results/file_name_09_30.pickle', 'wb') as handle:
        pickle.dump(file_name, handle, protocol=2)
    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Dropped_diff_09_30.pickle', 'wb') as handle:
        pickle.dump(O_Dropped_diff, handle, protocol=2)
    with open('/home/jiezou/Documents/Context_aware MCS/L_results/O_Dropped_diff_raw_09_30.pickle', 'wb') as handle:
        pickle.dump(O_Dropped_diff_raw, handle, protocol=2)
    with open('/home/jiezou/Documents/Context_aware MCS/L_results/EU_mean_09_30.pickle', 'wb') as handle:
        pickle.dump(EU_mean, handle, protocol=2)
    with open('/home/jiezou/Documents/Context_aware MCS/L_results/EU_difference_holistic_09_30.pickle', 'wb') as handle:
        pickle.dump(EU_difference_holistic, handle, protocol=2)
