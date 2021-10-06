import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.inference import CausalInference
from pgmpy.factors.discrete import JointProbabilityDistribution

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
print("----- Define the model structure of Bayesian Network ------")


class Task:

    def __init__(self, task, cpd):
        self.task = task
        self.cpd = cpd
Tasks = []

print("----- Define the conditional probabilities tables ------")

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

Tasks.append(Task(cpd_tau1.variable, cpd_tau1))
Tasks.append(Task(cpd_tau4.variable, cpd_tau4))

cpd_tau3 = TabularCPD(variable='tau3', variable_card=2,
                      values=[[0.7],
                              [0.3]],
                      )
Tasks.append(Task(cpd_tau3.variable, cpd_tau3))

cpd_tau6 = TabularCPD(variable='tau6', variable_card=2,
                      values=[[0.8],
                              [0.2]],
                      )
Tasks.append(Task(cpd_tau6.variable, cpd_tau6))

cpd_tau2 = TabularCPD(variable='tau2', variable_card=2,
                      values=[[0.85, 0.01],
                              [0.15, 0.99]],
                      evidence=['tau1'],
                      evidence_card=[2],
                      )
Tasks.append(Task(cpd_tau2.variable, cpd_tau2))

cpd_tau7 = TabularCPD(variable='tau7', variable_card=2,
                      values=[[0.9, 0.3, 0.85, 0.05],
                              [0.1, 0.7, 0.15, 0.95]],
                      evidence=['tau1', 'tau6'],
                      evidence_card=[2, 2],
                      )
Tasks.append(Task(cpd_tau7.variable, cpd_tau7))


cpd_tau5 = TabularCPD(variable='tau5', variable_card=2,
                      values=[[0.9, 0.85, 0.8, 0.75, 0.4, 0.35, 0.1, 0.05],
                              [0.1, 0.15, 0.2, 0.25, 0.6, 0.65, 0.9, 0.95]],
                      evidence= ['tau4', 'tau3', 'tau2'],
                      evidence_card=[2, 2, 2],
                      )

# print("%%%%%%%%%%%%%%%%%%%%%%%% ", cpd_tau5.get_evidence())
Tasks.append(Task(cpd_tau5.variable, cpd_tau5))
tau = []
for i in range(len(Tasks)):
    tau.append(Tasks[i].task)
print(tau)
# cpd_tau8 = TabularCPD(variable='tau8', variable_card=2,
#                       values=[[0.8, 0.1],
#                               [0.2, 0.9]],
#                       evidence=['tau7'],
#                       evidence_card=[2],
#                       state_names={'tau8': ['correct', 'wrong'],
#                                    'tau7': ['correct', 'wrong']})


print("----- Check the correctness of the model ------")
model = BayesianNetwork([('tau1', 'tau7'), ('tau6', 'tau7'), ('tau1', 'tau2'), ('tau2', 'tau5'), ('tau3', 'tau5'),
                         ('tau4', 'tau5')])
model.add_cpds(cpd_tau4, cpd_tau3, cpd_tau2, cpd_tau5, cpd_tau1, cpd_tau6, cpd_tau7)
model.get_cpds()
cpds = model.get_cpds()
for cpd in cpds:
    print(f'CPT of {cpd.variable}:')
    print(cpd, '\n')

print("---which group first ?---")

print("---Assumption 1: drop app2 (tau1 and tau2)")


def table_Reconstruction(cpd, dropped_task):
    # print(cpd.variables, '\n')
    # print(cpd.values.shape, '\n')

    task = cpd.variable
    print("pay attention to the order")
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

    # print(len(updated_cpd_values.shape))
    # if len(updated_cpd_values.shape) > 2:
    #     updated_cpd_values = np.squeeze(updated_cpd_values)
    #     print(len(updated_cpd_values.shape))
    # print(updated_cpd_values)

    # updated_cpd_values = cpd.values[:, extract_index, :]
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


def modified_Task(dropped_task):
    modified_task = []
    evidence = model.get_cpds(dropped_task).get_evidence()
    edges = model.edges
    for i in edges:
        if dropped_task in i:
            for j in i:
                if j != dropped_task and j not in evidence:
                    modified_task.append(j)

    return modified_task


dropped_task = 'tau1'
modified_task = modified_Task(dropped_task)
print("the tasks, whose cpd need to be modified:", modified_task)


updated_cpds = []
for i in modified_task:
    print(i)
    Tasks[tau.index(i)].cpd = table_Reconstruction(model.get_cpds(i), dropped_task)
    print(Tasks[tau.index(i)].cpd)

# print(Tasks[tau.index('tau7')].cpd)
# print(Tasks[tau.index('tau2')].cpd)

# model = BayesianNetwork([('tau6', 'tau7'), ('tau2', 'tau5'), ('tau3', 'tau5'),
#                          ('tau4', 'tau5')])
# model.add_cpds(cpd_tau4, cpd_tau3, cpd_tau2, cpd_tau5, cpd_tau1, cpd_tau6, cpd_tau7)

# model.remove_node('tau1')
# print(model.nodes)
# model.add_cpds(Tasks[tau.index('tau4')].cpd, Tasks[tau.index('tau3')].cpd, Tasks[tau.index('tau2')].cpd,
#                Tasks[tau.index('tau5')].cpd, Tasks[tau.index('tau6')].cpd, Tasks[tau.index('tau7')].cpd)
# model.get_cpds()
# infer = VariableElimination(model)
#
# tau5_dist = infer.query(['tau5'], evidence={'tau2': 1})
# tau7_dist = infer.query(['tau7'], evidence={'tau2': 1})
# print(tau5_dist, tau7_dist)

# # tau5_correct = tau5_dist.values[0]
# # tau7_correct = tau7_dist.values[0]
# # EU_global = tau5_correct * tau7_correct

# # print(tau5_correct, tau7_correct,  EU_global)


print("---Assumption 2: drop app1 (tau3 and tau2)")
print("")
# ----------------------------------------------------------------------------------------------------------------- #
#   Note here: If we only set tau2 as 'wrong' without graph reset,  the behavior ia regarded as the observation and #
#   and the state of tau1 will be automatically set as 'wrong' (MAP). Therefore, the marginal distribution of tau7  #
#   will also be modified and that is unexpected.                                                                   #
# # ----------------------------------------------------------------------------------------------------------------- #
# #
# infer = VariableElimination(model)
# tau5_dist1 = infer.query(['tau5'], evidence={'tau3': 1, 'tau2': 1})
# tau7_dist1 = infer.query(['tau7'], evidence={'tau3': 1, 'tau2': 1})
# print(tau5_dist1, tau7_dist1)
#
# #
# # tau1_MAP = infer.map_query(variables=['tau1'], evidence={'tau2': 1})
# # print(tau1_MAP)
# # #
#
# print("*************  Test of network Reconstruction  ***************")
# print("")
#
# model.remove_edge('tau1', 'tau2')
# cpd_tau2 = TabularCPD(variable='tau2', variable_card=2,
#                       values=[[0],
#                               [1]])
# model.add_cpds(cpd_tau4, cpd_tau3, cpd_tau2, cpd_tau5, cpd_tau1, cpd_tau6, cpd_tau7)
# model.get_cpds()
# infer = VariableElimination(model)
# tau5_dist1 = infer.query(['tau5'], evidence={'tau3': 1, 'tau2': 1})
# tau7_dist1 = infer.query(['tau7'], evidence={'tau3': 1, 'tau2': 1})
# print(tau5_dist1, tau7_dist1)
#
#
# # print(EU_global1)
# tau5_correct1 = tau5_dist1.values[0]
# tau7_correct1 = tau7_dist1.values[0]
# EU_global1 = tau5_correct1 * tau7_correct1
# print(tau5_correct1, tau7_correct1,  EU_global1)


# if EU_global > EU_global1:
#     print("drop app2 first")
# else:
#     print("drop app1 first")
# # print(tau5_dist * tau7_dist, tau5_dist1 * tau7_dist1)
#
#
# # print(model.get_cpds('tau1'))
# # print(model.get_cpds('tau6'))
# # print(model.get_cpds('tau7'))
# # model.remove_nodes_from(['tau2'])
# # model.get_cpds()
# # infer = VariableElimination(model)
# # # print(model.get_cpds('tau1'))
# # tau5_dist2 = infer.query(['tau5'])
# # tau7_dist2 = infer.query(['tau7'])
# #
# # print(tau7_dist2)
# #
#
# #
# # tau5_dist2 = infer.query(['tau5'], evidence={'tau2': 'wrong'})
# # tau7_dist2 = infer.query(['tau7'], evidence={'tau2': 'wrong'})
# # print(tau5_dist2, tau7_dist2)
#
