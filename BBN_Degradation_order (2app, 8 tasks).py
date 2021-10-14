import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import TreeSearch
from pgmpy.inference import CausalInference
from pgmpy.factors.discrete import JointProbabilityDistribution
import networkx as nx


# ----------------------------------------------------------------------------------------------------------------- #
#   Note here: If we only set tau2 as 'wrong' without graph reset,  the behavior ia regarded as the observation and #
#   and the state of tau1 will be automatically set as 'wrong' (MAP). Therefore, the marginal distribution of tau7  #
#   will also be modified and that is unexpected.                                                                   #
# ----------------------------------------------------------------------------------------------------------------- #

#### ----- Generate Bayesian Network randomly ------####

# model = BayesianNetwork([('A', 'B'), ('B', 'C'),
#                          ('A', 'D'), ('D', 'C'), ('E', 'D')])
# values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
#                       columns=['A', 'B', 'C', 'D', 'E'])
# model.fit(values)
# model.get_cpds()
# print(model.edges())
# print(model.get_cpds('A'))
# print(model.get_cpds('B'))
# print(model.get_cpds('C'))
# print(model.get_cpds('D'))

# model = BayesianNetwork([('A', 'D'), ('B', 'D'),
#                          ('C', 'D'), ('E', 'D')])
# values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
#                       columns=['A', 'B', 'C', 'D', 'E'])
# model.fit(values)
# model.get_cpds()
#
# V = np.reshape(model.get_cpds('D').values, (2, -1))
# print(V)

# model.remove_nodes_from(['A', 'B'])
# model.get_cpds()
# print(model.get_cpds('C'))
# print(model.get_cpds('D'))

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


def parameters_initialisation():

    for i in range(3):
        G, V, C, _, W = load_task(task_idx=i, dag_base_folder="/home/jiezou/Documents/Context_aware MCS/dag-gen-rnd/data/data-multi-m4-u2.0/0/")
        print("G: ", G)
        print("V: ", V)
        print("C: ", C)
        print("W: ", W)
        edges = []
        for i in V:
            for j in range(len(G[i])):
                pair = (i, G[i][j])
                edges.append(pair)
        print("The edges of network", edges)


    test2 = TabularCPD(variable=2, variable_card=2,
                          values=[[0.85, 0.01],
                                  [0.15, 0.99]],
                          evidence=[1],
                          evidence_card=[2],
                          )
    test1 = TabularCPD(variable=1, variable_card=2,
                          values=[[0.3],
                                  [0.7]],
                          )

    model = BayesianNetwork([(1, 2)])
    model.add_cpds(test2, test1)
    model.get_cpds()
    cpds = model.get_cpds()
    for cpd in cpds:
        print(f'CPT of {cpd.variable}:')
        print(cpd, '\n')




    model = BayesianNetwork([('A', 'D'), ('B', 'D'),
                             ('C', 'D'), ('E', 'D')])
    values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
                          columns=['A', 'B', 'C', 'D', 'E'])
    model.fit(values)
    model.get_cpds()

    V = np.reshape(model.get_cpds('D').values, (2, -1))
    # print(V)

    print("----- Parameters initialisation ------")

    Taskset = []

    print("----- Define the conditional probabilities tables and Task definition ------")

    Context_state = 'Abnormal'  # or Normal, the CPT of the related variables are depend on the context_state
    if Context_state == 'Abnormal':
        cpd_tau1 = TabularCPD(variable='tau1', variable_card=2,
                              values=[[0.3],
                                      [0.7]],
                              )

        cpd_tau4 = TabularCPD(variable='tau4', variable_card=2,
                              values=[[0.85],
                                      [0.15]],
                              )
    elif Context_state == 'Normal':
        cpd_tau1 = TabularCPD(variable='tau1', variable_card=2,
                              values=[[0.85],
                                      [0.15]],
                              )

        cpd_tau4 = TabularCPD(variable='tau4', variable_card=2,
                              values=[[0.75],
                                      [0.25]],
                              )

    Taskset.append(Task(cpd_tau1.variable, cpd_tau1, 'LO'))
    Taskset.append(Task(cpd_tau4.variable, cpd_tau4, 'HI'))

    cpd_tau3 = TabularCPD(variable='tau3', variable_card=2,
                          values=[[0.7],
                                  [0.3]],
                          )
    Taskset.append(Task(cpd_tau3.variable, cpd_tau3, 'LO'))

    cpd_tau6 = TabularCPD(variable='tau6', variable_card=2,
                          values=[[0.8],
                                  [0.2]],
                          )
    Taskset.append(Task(cpd_tau6.variable, cpd_tau6, 'HI'))

    cpd_tau2 = TabularCPD(variable='tau2', variable_card=2,
                          values=[[0.85, 0.01],
                                  [0.15, 0.99]],
                          evidence=['tau1'],
                          evidence_card=[2],
                          )
    Taskset.append(Task(cpd_tau2.variable, cpd_tau2, 'LO'))

    cpd_tau7 = TabularCPD(variable='tau7', variable_card=2,
                          values=[[0.9, 0.3, 0.85, 0.05],
                                  [0.1, 0.7, 0.15, 0.95]],
                          evidence=['tau1', 'tau6'],
                          evidence_card=[2, 2],
                          )
    Taskset.append(Task(cpd_tau7.variable, cpd_tau7, 'HI'))

    cpd_tau8 = TabularCPD(variable='tau8', variable_card=2,
                          values=[[0.7],
                                  [0.3]],
                          )

    Taskset.append(Task(cpd_tau8.variable, cpd_tau8, 'LO'))

    # cpd_tau5 = TabularCPD(variable='tau5', variable_card=2,
    #                       values=[[0.9, 0.85, 0.8, 0.75, 0.4, 0.35, 0.1, 0.05],
    #                               [0.1, 0.15, 0.2, 0.25, 0.6, 0.65, 0.9, 0.95]],
    #                       evidence=['tau4', 'tau3', 'tau2', "tau8"],
    #                       evidence_card=[2, 2, 2, 2],
    #                       )
    cpd_tau5 = TabularCPD(variable='tau5', variable_card=2,
                          values=V,
                          evidence=['tau4', 'tau3', 'tau2', "tau8"],
                          evidence_card=[2, 2, 2, 2],
                          )

    Taskset.append(Task(cpd_tau5.variable, cpd_tau5, 'HI'))

    return Taskset


def Appset_initialisation():
    Appset = []
    app1 = ['tau3', 'tau2', 'tau8']
    APP1 = APP('App1', app1, 'tau5')
    Appset.append(APP1)
    # app2 = ['tau1', 'tau2']
    app2 = ['tau1']
    APP2 = APP('App2', app2, 'tau7')
    Appset.append(APP2)
    return Appset


def model_initialisation(Tasks):
    model = BayesianNetwork([('tau1', 'tau7'), ('tau6', 'tau7'), ('tau1', 'tau2'), ('tau2', 'tau5'), ('tau3', 'tau5'),
                             ('tau4', 'tau5'), ('tau8', 'tau5')])

    tau = []
    for i in range(len(Tasks)):
        tau.append(Tasks[i].task)
    # print(tau)

    for i in tau:
        # print(Tasks1[tau_assump1.index(i)].cpd)
        model.add_cpds(Tasks[tau.index(i)].cpd)

    print("----- Check the correctness of the model and check the correctness of CPDs ------")
    model.get_cpds()

    return model


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


def Task_Dropping_Test(dropped_task_set, Tasks1, model1):
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

    for i in dropped_task_set:
        Tasks1 = task_Drop(i, model1, Tasks1)

    # dropped_task = 'tau3'
    # Tasks1 = task_Drop(dropped_task, model1, Tasks1)
    # print("")
    # # print(Tasks1[tau_assump1.index('tau5')].cpd)
    # dropped_task2 = 'tau2'
    # Tasks1 = task_Drop(dropped_task2, model1, Tasks1)

    print("The original tasks in the Bayesian network:", model1.nodes)
    # model = BayesianNetwork([('tau1', 'tau7'), ('tau6', 'tau7'), ('tau1', 'tau2'), ('tau2', 'tau5'), ('tau3', 'tau5'),
    #                          ('tau4', 'tau5')])

    Key_node_candidates = []
    LO_num = 0
    HI_num = 0
    for i in model1.nodes:
        evidence_num = len((model1.get_cpds(i).get_evidence()))
        # print(evidence_num)
        if evidence_num >= 2:
            for j in model1.get_cpds(i).get_evidence():
                if Tasks1[tau_assump1.index(j)].criticality == 'LO':
                    LO_num += 1
                elif Tasks1[tau_assump1.index(j)].criticality == 'HI':
                    HI_num += 1
            if LO_num >= 1 and HI_num >= 1:
                Key_node_candidates.append(i)
    print("Key nodes candidates:", Key_node_candidates)

    for i in dropped_task_set:
        model1.remove_node(i)
    # model1.remove_node('tau3')
    # model1.remove_node('tau2')
    updated_network_nodes = model1.nodes
    print("Updated tasks in the Bayesian network:", updated_network_nodes)
    print("The CPDs should be attached to the updated network")

    # check the attached CPDs after network update
    for i in updated_network_nodes:
        # print(Tasks1[tau_assump1.index(i)].cpd)
        model1.add_cpds(Tasks1[tau_assump1.index(i)].cpd)

    # check the correctness of network and select the calculation method (e.g., VariableElimination method)
    model1.get_cpds()

    # cpds = model1.get_cpds()
    # for cpd in cpds:
    #     print(f'CPT of {cpd.variable}:')
    #     print(cpd, '\n')

    infer = VariableElimination(model1)

    print("Select the key nodes and calculate corresponding Marginal Probability")
    key_nodes = Key_node_candidates
    # key_nodes = ['tau5', 'tau7']

    marginal_prob_set = []
    for i in key_nodes:
        marginal_prob = infer.query([i])
        marginal_prob_set.append(marginal_prob)
        print(marginal_prob)

    return marginal_prob_set


def initialisation():
    Tasks = parameters_initialisation()

    tau_assump = []
    for i in range(len(Tasks)):
        tau_assump.append(Tasks[i].task)
    print("----- Bayesian Network setup ------")
    model = model_initialisation(Tasks)

    # check the correctness of network reinitialisation

    # cpds = model.get_cpds()
    # for cpd in cpds:
    #     print(f'CPT of {cpd.variable}:')
    #     print(cpd, '\n')

    return Tasks, model


def global_Expected_Utility(marginal_prob_set):
    # TODO: the definition can be improved with the consideration of the safety-related elements.

    EU_global = 1
    for i in marginal_prob_set:
        EU_global *= i.values[0]
    return EU_global


def Application_drop_test(App_name, dropped_task_set, EU_global_set, Dropped_APPs, App_drop_tasks, model,
                          Tasks):
    # model_copy = model.copy()
    # Tasks_copy = Tasks

    marginal_prob_set1 = Task_Dropping_Test(dropped_task_set, Tasks, model)

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


def Application_drop_and_update(Tasks_pre, model_pre, Appset, HI_group):
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
                                                                            model_copy, Tasks_copy)
    print('\n', "#################################################", '\n')
    print("---Application discarding decision---", '\n')
    assumption_ID = EU_global_set.index(max(EU_global_set))
    print("Task dropping start from: ", Dropped_APPs[assumption_ID], '(', App_drop_tasks[assumption_ID], ')', '\n')
    print("#################################################", '\n')
    print("-----Network update and look for the next dropped App--- ")
    model_copy, Tasks_copy = model_task_copy(model_original, tasks_name_index, HI_group)
    for i in Appset:
        if i.app_name == Dropped_APPs[assumption_ID]:
            print(i.app_name, i.taskset)
            dropped_task_set = i.taskset
    marginal_prob_set = Task_Dropping_Test(dropped_task_set, Tasks_copy, model_copy)
    # print(model_copy.nodes)

    for i in dropped_task_set:
        Tasks = remove_task(Tasks_copy, i)

    Appset = remove_app(Appset, Dropped_APPs[assumption_ID])

    print("dropped app and tasks:", Dropped_APPs[assumption_ID], dropped_task_set)
    print("rest nodes", model_copy.nodes)

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
        marginal_prob_set = Task_Dropping_Test(dropped_task_set, Tasks, model)

        for j in marginal_prob_set:
            # print(j.variables)
            # print(j)
            if j.variables[0] == Keynode:
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
    Appset = Appset_initialisation
    print(model.nodes)
    dropped_task_set = []
    dropped_task_set.append(Dropped_task[task_Drop_ID])
    marginal_prob_set = Task_Dropping_Test(dropped_task_set, Tasks, model)
    for i in dropped_task_set:
        Tasks = remove_task(Tasks, i)

    Test_task_set.remove(Dropped_task[task_Drop_ID])
    Task_drop_order.append(Dropped_task[task_Drop_ID])

    return model, Tasks, Appset, Task_drop_order, Test_task_set


if __name__ == "__main__":
    App_drop_order = []
    App_drop_task_order = []
    tasks_name_index = []
    [Tasks_original, model_original] = initialisation()
    cpds = model_original.get_cpds()
    for cpd in cpds:
        print(f'CPT of {cpd.variable}:')
        print(cpd, '\n')
    HI_group = ['tau4', 'tau5', 'tau6', 'tau7']
    Appset = Appset_initialisation()
    for j in Tasks_original:
        tasks_name_index.append(j.task)
    print("tasks_name_index", tasks_name_index)
    model, Tasks = model_task_copy(model_original, tasks_name_index, HI_group)

    print("--------------------------------------------------------", '\n')
    print("--- Application discarding order ---", '\n')
    print("--------------------------------------------------------", '\n')

    while len(Appset) > 1:
        model, Tasks, Appset, Dropped_APP = Application_drop_and_update(Tasks, model, Appset, HI_group)
        App_drop_order.append(Dropped_APP)

    for i in Appset:
        # print(i.app_name)
        App_drop_order.append(i.app_name)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')
    print("Application discarding order:", App_drop_order)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')

    print("--------------------------------------------------------", '\n')
    print("--- Determine Task dropping order ---", '\n')
    print("--------------------------------------------------------", '\n')
    Appset = Appset_initialisation()

    model, Tasks = model_task_copy(model_original, tasks_name_index, HI_group)
    print(model.nodes)
    Task_drop_order = []

    for app in App_drop_order:
        temp_order = []
        print("================== App based Task Round =======================", '\n')
        print("Start to determine the task dropping order of application:", app)
        Appset = Appset_initialisation()
        Test_task_set = []
        for t in Appset:
            if t.app_name == app:
                app_keynode = t.keynode
                Test_task_set = t.taskset

        while Test_task_set:
            model, Tasks, Appset, temp_order, Test_task_set = Task_drop_and_update(app, Tasks, model, Appset,
                                                                                   HI_group, Test_task_set, temp_order,
                                                                                   app_keynode)

        Task_drop_order.append(temp_order)
        # for tt in temp_order:
        #     Task_drop_order.append(tt)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')
    print("Final result:")
    print("Application discarding order:", App_drop_order)
    print("Task degradation order", Task_drop_order, '\n')
    print("++++++++++++++++++++++++++++++++++++++++++++++++++", '\n')

