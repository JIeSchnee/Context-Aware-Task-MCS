import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.inference import CausalInference
from pgmpy.factors.discrete import JointProbabilityDistribution

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


# model.remove_nodes_from(['A', 'B'])
# model.get_cpds()
# print(model.get_cpds('C'))
# print(model.get_cpds('D'))


# ------------------ Test based on specified structure ----------------#


# cpd_tau8 = TabularCPD(variable='tau8', variable_card=2,
#                       values=[[0.8, 0.1],
#                               [0.2, 0.9]],
#                       evidence=['tau7'],
#                       evidence_card=[2],
#                       state_names={'tau8': ['correct', 'wrong'],
#                                    'tau7': ['correct', 'wrong']})

# tau is defined for the index searching

def parameters_initialisation():
    print("----- Parameters initialisation ------")

    class Task:
        # TODO: the properties will be extended to include more information for the scheduling problem
        # TODO: Pay attention to the order of conditions. When necessary, the TabularCPD.evidence order should be defined
        #  independently

        def __init__(self, task, cpd, criticality):
            self.task = task
            self.cpd = cpd
            # self.evidence = evidence_order
            self.criticality = criticality

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

    cpd_tau5 = TabularCPD(variable='tau5', variable_card=2,
                          values=[[0.9, 0.85, 0.8, 0.75, 0.4, 0.35, 0.1, 0.05],
                                  [0.1, 0.15, 0.2, 0.25, 0.6, 0.65, 0.9, 0.95]],
                          evidence=['tau4', 'tau3', 'tau2'],
                          evidence_card=[2, 2, 2],
                          )
    Taskset.append(Task(cpd_tau5.variable, cpd_tau5, 'HI'))

    return Taskset


def model_initialisation(Tasks):
    model = BayesianNetwork([('tau1', 'tau7'), ('tau6', 'tau7'), ('tau1', 'tau2'), ('tau2', 'tau5'), ('tau3', 'tau5'),
                             ('tau4', 'tau5')])

    tau = []
    for i in range(len(Tasks)):
        tau.append(Tasks[i].task)
    # print(tau)

    model.add_cpds(Tasks[tau.index('tau4')].cpd, Tasks[tau.index('tau3')].cpd,
                   Tasks[tau.index('tau2')].cpd,
                   Tasks[tau.index('tau5')].cpd, Tasks[tau.index('tau1')].cpd,
                   Tasks[tau.index('tau6')].cpd,
                   Tasks[tau.index('tau7')].cpd)

    print("----- Check the correctness of the model and check the correctness of CPDs ------")
    model.get_cpds()

    return model


def table_Reconstruction(cpd, dropped_task):
    # print(cpd.variables, '\n')
    # print(cpd.values.shape, '\n')

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
    updated_cpd_values = np.reshape(updated_cpd_values, (2,-1))
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
        print("Start to update CPD:", i)
        Tasks[tau.index(i)].cpd = table_Reconstruction(Tasks[tau.index(i)].cpd, dropped_task)
        print(Tasks[tau.index(i)].cpd)

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
        print("********************************************")

    # dropped_task = 'tau3'
    # Tasks1 = task_Drop(dropped_task, model1, Tasks1)
    # print("")
    # # print(Tasks1[tau_assump1.index('tau5')].cpd)
    # dropped_task2 = 'tau2'
    # Tasks1 = task_Drop(dropped_task2, model1, Tasks1)

    print("The original tasks in the Bayesian network:", model1.nodes)
    # model = BayesianNetwork([('tau1', 'tau7'), ('tau6', 'tau7'), ('tau1', 'tau2'), ('tau2', 'tau5'), ('tau3', 'tau5'),
    #                          ('tau4', 'tau5')])

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
    infer = VariableElimination(model1)

    print("Select the key nodes and calculate corresponding Marginal Probability")
    key_nodes = ['tau5', 'tau7']

    marginal_prob_set = []
    for i in key_nodes:
        marginal_prob = infer.query([i])
        marginal_prob_set.append(marginal_prob)
        print(marginal_prob)
    return marginal_prob_set


def reinitialisation():

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
    EU_global = 1
    for i in marginal_prob_set:
        EU_global *= i.values[0]
    return EU_global


if __name__ == "__main__":

    print("---which group first ?---", '\n')
    print("#################################################")
    EU_global_set = []
    Dropped_APPs = []
    print("---Assumption 1: drop app2 (tau1 and tau2)", '\n')

    [Tasks, model] = reinitialisation()
    dropped_task_set = ['tau1', 'tau2']
    marginal_prob_set1 = Task_Dropping_Test(dropped_task_set, Tasks, model)

    EU_global_set.append(global_Expected_Utility(marginal_prob_set1))
    Dropped_APPs.append('App2')

    print("The global expected utility of assumption 1:", '\n', global_Expected_Utility(marginal_prob_set1), '\n')
    print("#################################################", '\n')

    print("---Assumption 2: drop app1 (tau3 and tau2)")

    [Tasks, model] = reinitialisation()

    dropped_task_set = ['tau3', 'tau2']
    marginal_prob_set2 = Task_Dropping_Test(dropped_task_set, Tasks, model)

    EU_global_set.append(global_Expected_Utility(marginal_prob_set2))
    Dropped_APPs.append('App1')

    print("The global expected utility of assumption 2:", '\n', global_Expected_Utility(marginal_prob_set2), '\n')

    print("---Application discarding decision---", '\n')
    assumption_ID = EU_global_set.index(max(EU_global_set))
    print("Task dropping start from: ", Dropped_APPs[assumption_ID], '\n')

    print("--------------------------------------------------------")
