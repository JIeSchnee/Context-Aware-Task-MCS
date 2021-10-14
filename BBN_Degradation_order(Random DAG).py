import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import TreeSearch
from pgmpy.inference import CausalInference
from pgmpy.factors.discrete import JointProbabilityDistribution
import networkx as nx
import random
from random import choice


class Task:
    # TODO: the properties will be extended to include more information for the scheduling problem
    # TODO: Pay attention to the order of conditions. When necessary, the TabularCPD.evidence order should be defined
    #  independently

    def __init__(self, task, cpd, criticality):
        self.task = task
        self.cpd = cpd
        # self.evidence = evidence_order
        self.criticality = criticality
        # self.app = app
        # self.second = app_2


class APP:
    def __init__(self, app_name, taskset, keynode):
        self.app_name = app_name
        self.taskset = taskset
        self.keynode = keynode


def load_task(task_idx, dag_base_folder="/home/jiezou/Documents/Context_aware MCS/dag-gen-rnd/data"):
    # << load DAG task <<
    dag_task_file = dag_base_folder + "Tau_{:d}.gpickle".format(task_idx)

    # task is saved as NetworkX gpickle format
    G = nx.read_gpickle(dag_task_file)

    # formulate the graph list
    G_dict = {}
    C_dict = {}
    V_array = []
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

    # >> end of load DAG task >>
    return G_dict, V_array, C_dict, C_array, W


def dictionary_definition():

    H = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    dict = {}
    for i in range(100):
        salt = ''
        for j in range(3):
            salt += random.choice(H)
        dict[i] = salt

    return dict


def parameters_initialisation(dict):

    Appset = []
    bias = 0
    network_tasks = []
    network_edges = []
    HI_group = []

    for app in range(3):
        print("==========================")
        G, V, C, _, W = load_task(task_idx=app, dag_base_folder="/home/jiezou/Documents/Context_aware MCS/dag-gen-rnd/data/data-multi-m4-u2.0/0/")
        # print("G: ", G)
        # print("V: ", V)
        # print("C: ", C)
        # print("W: ", W)

        edges = []
        for i in range(len(V)):
            for j in range(len(G[V[i]])):
                pair = (dict[V[i] + bias], dict[G[V[i]][j] + bias])
                edges.append(pair)
            V[i] += bias
        bias += len(V)

        # print("The tasks of current application", V)
        # print("The edges of current application", edges)

        for i in range(len(V)):
            V[i] = dict[V[i]]

        tempApp = APP(app, V, V[-1])
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
            remove_temp=[]
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

    # network_tasks = [4, 5, 6, 7, 9]
    # network_edges = [(4, 7), (5, 7), (6, 7), (7, 9)]
    # model = BayesianNetwork(network_edges)
    # values = pd.DataFrame(np.random.randint(low=0, high=2, size=(100, len(network_tasks))),
    #                       columns= network_tasks)
    # model.fit(values)
    # model.get_cpds()

    # Tasks = []
    # for i in network_tasks:
    #     if i in HI_group:
    #         criticality = 'HI'
    #     else:
    #         criticality = 'LO'
    #     Tasks.append(Task(model.get_cpds(i).variable, model.get_cpds(i), criticality))

    # print(network_tasks)
    # print(model.nodes)

    for i in Appset:
        i.taskset = [x for x in i.taskset if x not in HI_group]
        print("Application ID:", i.app_name,'\n', "The droppable tasks in the application", i.taskset, '\n',
              "The keynode of the application", i.keynode)


    # return Tasks, Appset, model, HI_group
    return network_edges, network_tasks, Appset, HI_group


def value_generation(network_tasks):
    values = pd.DataFrame(np.random.randint(low=0, high=2, size=(100, len(network_tasks))),
                          columns=network_tasks)
    return values


def initialisation(network_edges, network_tasks, Appset, HI_group, values):

    model = BayesianNetwork(network_edges)
    # values = pd.DataFrame(np.random.randint(low=0, high=2, size=(100, len(network_tasks))),
    #                       columns=network_tasks)
    model.fit(values)
    model.get_cpds()

    Tasks = []
    for i in network_tasks:
        if i in HI_group:
            criticality = 'HI'
        else:
            criticality = 'LO'
        Tasks.append(Task(model.get_cpds(i).variable, model.get_cpds(i), criticality))

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
        print(marginal_prob)

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
    print(dropped_app)
    for i in Appset:
        if i.app_name == dropped_app:
            Appset.remove(i)
    print("Remove APP succeed")
    return Appset


def model_task_copy(model_original, tasks_name_index, HI_group):
    model_copy = model_original.copy()
    Tasks_copy = []
    for j in tasks_name_index:
        if j in HI_group:
            temp_task = Task(j, model_copy.get_cpds(j), 'HI')
        else:
            temp_task = Task(j, model_copy.get_cpds(j), 'LO')
        Tasks_copy.append(temp_task)

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

        model_copy, Tasks_copy = model_task_copy(model_pre, tasks_name_index, HI_group)

        dropped_task_set = Appset[i].taskset
        print(model_copy.nodes())

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
    print("Task dropping start from: ", Dropped_APPs[assumption_ID], '(', App_drop_tasks[assumption_ID], ')', '\n')
    print("#################################################", '\n')
    print("-----Network update and look for the next dropped App--- ")
    model_copy, Tasks_copy = model_task_copy(model_pre, tasks_name_index, HI_group)
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
    print("rest nodes", model_copy.nodes)

    # for i in Tasks:
    #     print(i.task)

    return model_copy, Tasks, Appset, Dropped_APPs[assumption_ID]


def app_task_drop_test(app, Test_task_set, model_pre, tasks_name_index, HI_group, Appset, Keynode):
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


        model, Tasks = model_task_copy(model_pre, tasks_name_index, HI_group)
        print(model.nodes)

        dropped_task_set = []
        dropped_task_set.append(Test_task_set[i])
        marginal_prob_set = Task_Dropping_Test(dropped_task_set, Tasks, model, Keynode)
        print(Keynode)
        for j in marginal_prob_set:
            print(j.variables)
            print(j)
            if j.variables[0] == Keynode[0]:
                marginal_prob_key = j
        print("***** Marginal Distribution of Key node ****", '\n')
        print(marginal_prob_key)

        local_Expected_Utility = marginal_prob_key.values[0]
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
                                                    keynode)

    print('\n', "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", '\n')
    print("Task discarding decision:")
    task_Drop_ID = EU_local_set.index(max(EU_local_set))
    print("Task dropping start from: ", app, Dropped_task[task_Drop_ID], '\n')
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", '\n')
    print("-----Network update and look for the next dropped task--- ")
    model, Tasks = model_task_copy(model_pre, tasks_name_index, HI_group)
    Appset = Appset_original
    print(model.nodes)
    dropped_task_set = []
    dropped_task_set.append(Dropped_task[task_Drop_ID])
    marginal_prob_set = Task_Dropping_Test(dropped_task_set, Tasks, model, keynode)
    for i in dropped_task_set:
        Tasks = remove_task(Tasks, i)

    Test_task_set.remove(Dropped_task[task_Drop_ID])
    Task_drop_order.append(Dropped_task[task_Drop_ID])

    return model, Tasks, Appset, Task_drop_order, Test_task_set


if __name__ == "__main__":

    App_drop_order = []
    App_drop_task_order = []
    tasks_name_index = []
    dict = dictionary_definition()
    network_edges, network_tasks, Appset, HI_group = parameters_initialisation(dict)
    values = value_generation(network_tasks)
    Tasks_original, model_original, Appset_original, HI_group = initialisation(network_edges, network_tasks, Appset, HI_group, values)

    Appset_backpack = []
    Appset = Appset_original
    keynode = []
    for t in Appset:
        keynode.append(t.keynode)
        Appset_backpack.append(t)
        # print("GGGGGGG", t.taskset)

    for j in Tasks_original:
        tasks_name_index.append(j.task)
    # print("tasks_name_index", tasks_name_index)
    model, Tasks = model_task_copy(model_original, tasks_name_index, HI_group)

    print("--------------------------------------------------------", '\n')
    print("--- Application discarding order ---", '\n')
    print("--------------------------------------------------------", '\n')

    while len(Appset) > 1:

        model, Tasks, Appset, Dropped_APP = Application_drop_and_update(Tasks, model, Appset, HI_group, keynode)
        App_drop_order.append(Dropped_APP)

        # for i in Tasks:
        #     print(i.task)

    for i in Appset:
        # print(i.app_name)
        App_drop_order.append(i.app_name)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')
    print("Application discarding order:", App_drop_order)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')

    print("--------------------------------------------------------", '\n')
    print("--- Determine Task dropping order ---", '\n')
    print("--------------------------------------------------------", '\n')
    # Appset = []
    model, Tasks = model_task_copy(model_original, tasks_name_index, HI_group)
    # print(model.nodes)
    Task_drop_order = []
    print(App_drop_order)

    for app in App_drop_order:
        temp_order = []
        print("================== App based Task Round =======================", '\n')
        print("Start to determine the task dropping order of application:", app)
        Appset = []
        for h in Appset_backpack:
            print("uuuuu", h.taskset)
            Appset.append(h)

        for t in Appset:
            print("TTTTT", t.taskset)
            keynode.append(t.keynode)
            # print("TTTTT", t.taskset)

        Test_task_set = []
        for t in Appset:
            # print(t.app_name, t.taskset)
            if t.app_name == app:
                app_keynode = t.keynode
                Test_task_set = t.taskset

        print("Droppable tasks", Test_task_set, '\n', "Keynode:", app_keynode)
        keynode = []
        keynode.append(app_keynode)


        while Test_task_set:
            model, Tasks, Appset, temp_order, Test_task_set = Task_drop_and_update(app, Tasks, model, Appset,
                                                                                   HI_group, Test_task_set, temp_order,
                                                                                   keynode)


        Task_drop_order.append(temp_order)
        # for tt in temp_order:
        #     Task_drop_order.append(tt)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')
    print("Final result:")
    print("Application discarding order:", App_drop_order)
    print("Task degradation order", Task_drop_order, '\n')
    print("++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')


