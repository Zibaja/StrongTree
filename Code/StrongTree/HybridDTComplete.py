import numpy as np
from Tree import Tree
import gurobipy as gp
from gurobipy import GRB
from Hybrid_utils import *

import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math



class FlowOCT:
    def __init__(self, X,y, features, tree, lambdaa, eta, alpha,bb_prediction, time_limit):
        self.X = X #X is datafarme
        self.y = y.astype(str)
        self.datapoints = X.index
        self.num_datapoints = self.X.shape[0]
        self.class_labels = np.unique(self.y)
        self.labels = np.append(self.class_labels, 'bb')
        self.cat_features = features
        self.tree = tree
        self.lambdaa = lambdaa
        self.eta = eta
        self.alpha = alpha
        self.black_box_pred = np.asarray(bb_prediction).astype(str) 
        
        # parameters
        self.m = {}
        for i in self.datapoints:
            self.m[i] = 1


        # Decision Variables
        self.b = 0 #branching variables
        self.p = 0 #leaf variables
        self.beta = 0 #label assignment variables
        self.zeta = 0 # z[i,n,tk] is the flow from node n to sink nodes
        self.z = 0 # z[i,n] is the incoming flow to node n for datapoint i
        self.d = 0 #variables for outgoing flow of deferred nodes

        # Gurobi model
        self.model = gp.Model('FlowOCT')
        '''
        To compare all approaches in a fair setting we limit the solver to use only one thread to merely evaluate 
        the strength of the formulation.
        '''
        self.model.params.Threads = 1
        self.model.params.TimeLimit = time_limit

        '''
        The following variables are used for the Benders problem to keep track of the times we call the callback.
        They are not used for this formulation.
        '''
        self.model._total_callback_time_integer = 0
        self.model._total_callback_time_integer_success = 0

        self.model._total_callback_time_general = 0
        self.model._total_callback_time_general_success = 0

        self.model._callback_counter_integer = 0
        self.model._callback_counter_integer_success = 0

        self.model._callback_counter_general = 0
        self.model._callback_counter_general_success = 0



    ###########################################################
    # Create the MIP formulation
    ###########################################################
    def create_primal_problem(self):
        '''
        This function create and return a gurobi model formulating the FlowOCT problem
        :return:  gurobi model object with the FlowOCT formulation
        '''
        #one-hot encoding for predictions of blackbox
        bb_labels = self.class_labels
        bb_pred = {k: np.zeros(self.X.shape[0]) for k in bb_labels}

        for k in bb_labels:
            bb_pred[k][np.where(self.black_box_pred == k)] = 1


        ############################### define variables
        # b[n,f] ==1 iff at node n we branch on feature f
        self.b = self.model.addVars(self.tree.Nodes, self.cat_features, vtype=GRB.BINARY, name='b')
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self.p = self.model.addVars(self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name='p')
        
        #For classification beta[n,k]=1 iff at node n we predict class k #Change by Ziba to binary instead of continuos
        self.beta = self.model.addVars(self.tree.Nodes + self.tree.Leaves, self.labels, vtype=GRB.BINARY, lb=0,
                                       name='beta')
        # zeta[i,n] is the amount of flow through the edge connecting node n to sink node t_k for datapoint i
        self.zeta = self.model.addVars(self.datapoints, self.tree.Nodes + self.tree.Leaves,self.class_labels, vtype=GRB.CONTINUOUS, lb=0,
                                       name='zeta')
        # z[i,n] is the incoming flow to node n for datapoint i
        self.z = self.model.addVars(self.datapoints, self.tree.Nodes + self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                    name='z')
        
        self.d = self.model.addVars(self.datapoints, self.tree.Nodes + self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                    name='d')

        ############################### define constraints

        # z[i,n] = z[i,l(n)] + z[i,r(n)] + sum(zeta[i,n,k], k)    forall i, n in Nodes
        for n in self.tree.Nodes:
            n_left = int(self.tree.get_left_children(n))
            n_right = int(self.tree.get_right_children(n))
            self.model.addConstrs(
                (self.z[i, n] == self.z[i, n_left] + self.z[i, n_right] + gp.quicksum(self.zeta[i, n, k] for k in self.class_labels)) for i in self.datapoints)

        # z[i,n] = sum(z[i,n,k] , k) for i,n in terminal nodes
        for n in self.tree.Leaves:
            self.model.addConstrs(gp.quicksum(self.zeta[i, n, k] for k in self.class_labels) == self.z[i, n] for i in self.datapoints)

        #z[i,1]==1 forall i 
        self.model.addConstrs(self.z[i,1]==1 for i in self.datapoints)


        # z[i,l(n)] <= m[i] * sum(b[n,f], f if x[i,f]=0)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_left_children(n))] <= self.m[i] * gp.quicksum(
                self.b[n, f] for f in self.cat_features if self.X.at[i, f] == 0)) for n in self.tree.Nodes)

        # z[i,r(n)] <= m[i] * sum(b[n,f], f if x[i,f]=1)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_right_children(n))] <= self.m[i] * gp.quicksum(
                self.b[n, f] for f in self.cat_features if self.X.at[i, f] == 1)) for n in self.tree.Nodes)

        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self.model.addConstrs(
            (gp.quicksum(self.b[n, f] for f in self.cat_features) + self.p[n] + gp.quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Nodes)

        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self.model.addConstrs(
            (self.p[n] + gp.quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Leaves)

        # sum(sum(b[n,f], f), n) <= branching_limit
        # self.model.addConstr(
        #     (quicksum(
        #         quicksum(self.b[n, f] for f in self.cat_features) for n in self.tree.Nodes)) <= self.branching_limit)

        # loss reduction:
        # sum(beta[n,k], k in labels) = p[n]

        # zeta[i,n,k] <= beta[n,k] for all n, i, k
        for n in self.tree.Nodes + self.tree.Leaves:
            for k in self.class_labels:
                # self.model.addConstrs(self.zeta[i, n, k] <= self.beta[n, k] for i in self.datapoints)
                # # zeta[i,n,k] <= pred_bb[i,k] * beta[n,'bb'] for all n, i, k for bb nodes
                # self.model.addConstrs(self.zeta[i,n,k]<= bb_pred[k][i] * self.beta[n,'bb'] for i in self.datapoints)
                
                self.model.addConstrs(self.zeta[i, n, k] <= self.beta[n, k] + bb_pred[k][i] * self.beta[n,'bb'] 
                                       for i in self.datapoints)
                
        
        # sum(beta[n,k]==p_n for all n)
        self.model.addConstrs(
            (gp.quicksum(self.beta[n, k] for k in self.labels) == self.p[n]) for n in
            self.tree.Nodes + self.tree.Leaves)
       
        #min_transparency constraint
        self.model.addConstr(1- ((gp.quicksum(self.d[i,n] for i in self.datapoints
                                              for n in (self.tree.Nodes + self.tree.Leaves)))/self.num_datapoints) >= self.alpha)

        #set of constraints related to d variables
        for n in self.tree.Nodes + self.tree.Leaves:
            self.model.addConstrs(self.d[i,n] <= gp.quicksum(self.zeta[i,n,k] for k in self.class_labels) for i in self.datapoints)
            self.model.addConstrs(self.d[i,n] <= self.beta[n,'bb'] for i in self.datapoints)
            self.model.addConstrs(self.d[i,n] >= gp.quicksum(
                self.zeta[i,n,k] for k in self.class_labels)-(1-self.beta[n,'bb']) for i in self.datapoints)
            


        # define objective function


        obj = ((1/self.num_datapoints)* gp.quicksum(
            self.zeta[i,n,self.y[i]] for i in self.datapoints for n in self.tree.Nodes + self.tree.Leaves )) - (self.lambdaa * (
                gp.quicksum(self.b[n,f] for n in self.tree.Nodes for f in self.cat_features))) - ((self.eta/self.num_datapoints) * (
                    gp.quicksum(self.d[i,n] for i in self.datapoints for n in self.tree.Nodes + self.tree.Leaves  )))


        self.model.setObjective(obj, GRB.MAXIMIZE)

        return self
    
    def _record_incumbent_callback(self, model, where):
        if where == GRB.Callback.MIPSOL:
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            best_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)

            if abs(obj) > 1e-10:
                gap = abs(best_bound - obj) / abs(obj)
            else:
                gap = None

            zeta_sol = model.cbGetSolution(self.zeta)
            d_sol = model.cbGetSolution(self.d)

            train_acc = sum(
                zeta_sol[i, n, self.y[i]]
                for i in self.datapoints
                for n in self.tree.Nodes + self.tree.Leaves
            ) / self.num_datapoints

            train_transparency = 1 - (
                sum(
                    d_sol[i, n]
                    for i in self.datapoints
                    for n in self.tree.Nodes + self.tree.Leaves
                ) / self.num_datapoints
            )

            model._callback_history.append({
                "time": runtime,
                "obj": obj,
                "best_bound": best_bound,
                "gap": gap,
                "train_acc": train_acc,
                "train_transparency": train_transparency,
            })

    
    # def solve (self):
    #     self.model.update()
    #     #self.model.write('testmodel.lp') #just for debug
    #     self.model.optimize()
    #     model_info =  {'Status':self.model.getAttr("Status"), "obj_value": self.model.getAttr("ObjVal"), "MIPGap":self.model.getAttr("MIPGap") * 100,
    #      "NodeCount": self.model.getAttr("NodeCount"), "total_callback_time_integer":self.model._total_callback_time_integer,
    #       "total_callback_time_integer_success":self.model._total_callback_time_integer_success,"callback_counter_integer":self.model._callback_counter_integer,
    #         "callback_counter_integer_success":self.model._callback_counter_integer_success}
    #     return model_info

    def solve(self):
        self.model.update()

        self.model._callback_history = []

        self.model.optimize(self._record_incumbent_callback)

        model_info = {
            "Status": self.model.Status,
            "obj_value": self.model.ObjVal if self.model.SolCount > 0 else None,
            "MIPGap": self.model.MIPGap * 100 if self.model.SolCount > 0 else None,
            "NodeCount": self.model.NodeCount,
            "Runtime": self.model.Runtime,
            "SolCount": self.model.SolCount,
            "callback_history": self.model._callback_history,
            "total_callback_time_integer": self.model._total_callback_time_integer,
            "total_callback_time_integer_success": self.model._total_callback_time_integer_success,
            "callback_counter_integer": self.model._callback_counter_integer,
            "callback_counter_integer_success": self.model._callback_counter_integer_success,
        }

        return model_info
       


class BendersOCT:
    def __init__(self, X,y, features, tree, lambdaa, eta,bb_prediction, time_limit):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            features (_type_): _description_
            tree (_type_): _description_
            lambdaa (_type_): _description_
            eta (_type_): _description_
            alpha (_type_): _description_
            bb_prediction (np.array): this is used for training and is indeed the bb predictions vector 
            time_limit (_type_): _description_
        """
        self.X = X #X is datafarme
        self.y = y.astype(str)
        self.datapoints = X.index
        self.num_datapoints = self.X.shape[0]
        self.class_labels = np.unique(self.y)
        self.labels = np.append(self.class_labels, 'bb')
        self.cat_features = features
        self.tree = tree
        self.lambdaa = lambdaa
        self.eta = eta
        self.black_box_pred = np.asarray(bb_prediction).astype(str) 
        
        # parameters
        self.m = {}
        for i in self.datapoints:
            self.m[i] = 1

      

        # Decision Variables
        self.g = 0
        self.b = 0
        self.p = 0
        self.beta = 0

        
        # Gurobi model
        self.model = gp.Model('BendersOCT')
        # The cuts we add in the callback function would be treated as lazy constraints
        self.model.params.LazyConstraints = 1
        '''
        To compare all approaches in a fair setting we limit the solver to use only one thread to merely evaluate 
        the strength of the formulation.
        '''
        self.model.params.Threads = 1
        self.model.params.TimeLimit = time_limit

        '''
        The following variables are used for the Benders problem to keep track of the times we call the callback.
        
        - counter_integer tracks number of times we call the callback from an integer node in the branch-&-bound tree
            - time_integer tracks the associated time spent in the callback for these calls
        - counter_general tracks number of times we call the callback from a non-integer node in the branch-&-bound tree
            - time_general tracks the associated time spent in the callback for these calls
        
        the ones ending with success are related to success calls. By success we mean ending up adding a lazy constraint 
        to the model
        
        
        '''
        self.model._total_callback_time_integer = 0
        self.model._total_callback_time_integer_success = 0

        self.model._total_callback_time_general = 0
        self.model._total_callback_time_general_success = 0

        self.model._callback_counter_integer = 0
        self.model._callback_counter_integer_success = 0

        self.model._callback_counter_general = 0
        self.model._callback_counter_general_success = 0

        # We also pass the following information to the model as we need them in the callback
        self.model._master = self

    ###########################################################
    # Create the master problem
    ###########################################################
    def create_master_problem(self):
        '''
        This function create and return a gurobi model formulating the BendersOCT problem
        :return:  gurobi model object with the BendersOCT formulation
        '''
     
        ############################### define variables
        # g[i] is the objective value for the subproblem[i]
        self.g = self.model.addVars(self.datapoints, vtype=GRB.CONTINUOUS,lb=-self.eta, ub=1, name='g')
        # b[n,f] ==1 iff at node n we branch on feature f
        self.b = self.model.addVars(self.tree.Nodes, self.cat_features, vtype=GRB.BINARY, name='b')
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self.p = self.model.addVars(self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name='p')
        
        #For classification beta[n,k]=1 iff at node n we predict class k #Change by Ziba to binary instead of continuos
        self.beta = self.model.addVars(self.tree.Nodes + self.tree.Leaves, self.labels, vtype=GRB.BINARY, lb=0,
                                       name='beta')
     

        # we need these in the callback to have access to the value of the decision variables
        self.model._vars_g = self.g
        self.model._vars_b = self.b
        self.model._vars_p = self.p
        self.model._vars_beta = self.beta

        # define constraints
        ###############################################################################
        
        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self.model.addConstrs(
            (gp.quicksum(self.b[n, f] for f in self.cat_features) + self.p[n] + gp.quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Nodes)


        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self.model.addConstrs(
            (self.p[n] + gp.quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Leaves)
        
        # sum(beta[n,k]==p_n for all n)
        self.model.addConstrs(
            (gp.quicksum(self.beta[n, k] for k in self.labels) == self.p[n]) for n in
            self.tree.Nodes + self.tree.Leaves)


        # define objective function
        obj = gp.LinExpr(0)
        for i in self.datapoints:
            obj.add((1/self.num_datapoints) * (self.g[i]))

        for n in self.tree.Nodes:
            for f in self.cat_features:
                obj.add(-1 * self.lambdaa * self.b[n, f])

        self.model.setObjective(obj, GRB.MAXIMIZE)

        return self ##added by Ziba

#I first need to find out all arcs capaciti values as well as master decision varaibles to construct the expresion
# then , find the L values on all enabled arcs which is exactly solving the SP
# having solved SP, I can obtain gamma values for DSP
#then construct the cut



    def flow_graph_construction(self, b, beta, i): #master hamoon BendersOCT 
        
        all_arcs = {}
        all_arcs[(0,1)] = {'capacity':1, 'cost': 0, 'exp':1} 
        left_features = [
            f for f in self.cat_features
            if self.X.at[i, f] == 0
        ]

        right_features = [
            f for f in self.cat_features
            if self.X.at[i, f] == 1
        ]
        for n in self.tree.Nodes:
            all_arcs[(n,int(self.tree.get_left_children(n)))] = {'capacity':float(sum([b[n,f] for f in left_features]))
                                                                    , 'cost': 0, 'exp': gp.quicksum(self.b[n,f] for f in left_features)}
            all_arcs[(n,int(self.tree.get_right_children(n)))] = {'capacity':float(sum([b[n,f] for f in right_features])), 
                                                                    'cost': 0, 'exp': gp.quicksum(self.b[n,f] for f in right_features) }
            for k in self.class_labels:
                sink = ("sink", k)
                all_arcs[(n,sink, 'DT')] = {'capacity':float(beta[n,k]) , 'cost':float(self.y[i]==k), 'exp': self.beta[n,k]}
                all_arcs[(n,sink, 'BB')] = {'capacity':float(beta[n,'bb']*int(self.black_box_pred[i]==str(k) )) , 'cost':float(self.y[i]==k)-self.eta, 'exp': self.beta[n,'bb']*int(self.black_box_pred[i]==str(k) )}

        for n in self.tree.Leaves:
            for k in self.class_labels:
                sink = ("sink", k)
                all_arcs[(n,sink, 'DT')] = {'capacity':float(beta[n,k]) , 'cost':float(self.y[i]==k), 'exp':self.beta[n,k] }
                all_arcs[(n,sink, 'BB')] = {'capacity':float(beta[n,'bb']*int(self.black_box_pred[i]==str(k) ) ), 'cost':float(self.y[i]==k)-self.eta, 'exp': self.beta[n,'bb']*int(self.black_box_pred[i]==str(k) )}

        return all_arcs

    def compute_L_values(self, all_arcs, root=1, tol=1e-8):
        """
        Compute the longest-path value L[n] for every tree node.

        Only arcs with positive incumbent capacity are enabled in the
        current subproblem.

        Parameters
        ----------
        all_arcs : dict
            Template arcs with capacity, cost, and master expression.
        root : int
            Root tree node.
        tol : float
            Numerical tolerance.

        Returns
        -------
        L : dict
            Longest-path value from every node to a prediction sink.
        best_arc : dict
            Best outgoing arc selected from each node.
        subproblem_value : float
            Longest-path value starting from the root.
        """

        def arc_origin(arc):
            return arc[0]

        def arc_destination(arc):
            return arc[1]

        def is_sink(node):
            return (
                isinstance(node, tuple)
                and len(node) == 2
                and node[0] == "sink"
            )

        outgoing = defaultdict(list)
        all_nodes = set()
        sinks = set()

        for arc in all_arcs:
            origin = arc_origin(arc)
            destination = arc_destination(arc)

            outgoing[origin].append(arc)

            all_nodes.add(origin)
            all_nodes.add(destination)

            # IMPORTANT: the sink is the destination of a prediction arc.
            if is_sink(destination):
                sinks.add(destination)

        # Every class sink has continuation value zero.
        L = {sink: 0.0 for sink in sinks}

        best_arc = {}
        visiting = set()

        def dp(node):
            if node in L:
                return L[node]

            if node in visiting:
                raise RuntimeError(
                    "The subproblem graph contains a cycle. "
                    "The dynamic program requires a DAG."
                )

            visiting.add(node)

            best_value = None
            selected_arc = None

            for arc in outgoing.get(node, []):
                capacity = float(all_arcs[arc]["capacity"])

                # Only positive-capacity arcs are available in the
                # current fixed-master subproblem.
                if capacity <= tol:
                    continue

                destination = arc_destination(arc)
                destination_value = dp(destination)

                if destination_value is None:
                    continue

                candidate_value = (
                    float(all_arcs[arc]["cost"])
                    + destination_value
                )

                if (
                    best_value is None
                    or candidate_value > best_value + tol
                ):
                    best_value = candidate_value
                    selected_arc = arc

            visiting.remove(node)

            if best_value is None:
                L[node] = None
            else:
                L[node] = float(best_value)
                best_arc[node] = selected_arc

            return L[node]

        # Compute values for all tree nodes, including currently unreachable
        # nodes, because their potentials are needed for the dual cut.
        for node in all_nodes:
            if not is_sink(node):
                dp(node)

        if L.get(root) is None:
            positive_root_arcs = [
                arc
                for arc in outgoing.get(root, [])
                if all_arcs[arc]["capacity"] > tol
            ]

            raise RuntimeError(
                f"Datapoint {root=} has no enabled path to a sink.\n"
                f"Positive-capacity arcs leaving the root: "
                f"{positive_root_arcs}"
            )

        subproblem_value = float(L[root])

        # Potentials of nodes having no active path can be selected freely.
        # Setting them to zero is valid because gamma will compensate for
        # their zero-capacity arcs.
        for node in all_nodes:
            if L.get(node) is None:
                L[node] = 0.0

        return L, best_arc, subproblem_value


    # def compute_L_values(self,all_arcs,root=1,tol=1e-8):
    #     """
    #     Compute the longest-path DP value L[n] for every template node.

    #     L[n] is the maximum reward obtainable starting at node n,
    #     using arcs having positive capacity under the current master
    #     solution.

    #     For nodes having no enabled path to a sink, L[n] is set to 0.
    #     Such nodes must not be reachable from the root in a feasible
    #     incumbent solution.

    #     Returns
    #     -------
    #     L : dict
    #         DP value for every node and sink.

    #     best_arc : dict
    #         best_arc[n] is the arc selected by the DP at node n.

    #     subproblem_value : float
    #         L[root].
    #     """
    #     def arc_head(arc):
    #         return arc[0]
    #     def arc_tail(arc):
    #         return arc[1]

    #     def is_sink(node):
    #         return (
    #             isinstance(node, tuple)
    #             and len(node) == 2
    #             and node[0] == "sink"
    #         )

    #     # Full outgoing adjacency list
    #     outgoing = defaultdict(list)

    #     all_nodes = set()
    #     sinks = set()

    #     for arc in all_arcs:
    #         head = arc_head(arc)
    #         tail = arc_tail(arc)
            

    #         outgoing[head].append(arc)

    #         all_nodes.add(head)
    #         all_nodes.add(tail)

    #         if is_sink(head):
    #             sinks.add(tail) ####

    #     # Sink values
    #     L = {sink: 0.0 for sink in sinks}

    #     best_arc = {}
    #     visiting = set()

    #     def dp(node):
    #         """
    #         Recursive longest-path computation on the DAG.
    #         """

    #         if node in L:
    #             return L[node]

    #         if node in visiting:
    #             raise RuntimeError(
    #                 "The subproblem graph contains a cycle. "
    #                 "The DP requires an acyclic network."
    #             )

    #         visiting.add(node)

    #         best_value = None
    #         selected_arc = None

    #         for arc in outgoing.get(node, []):

    #             capacity = float(all_arcs[arc]["capacity"])

    #             # Current longest-path problem uses only enabled arcs
    #             if capacity <= tol:
    #                 continue

    #             tail = arc_tail(arc)
    #             tail_value = dp(tail)

    #             # None means that this enabled arc does not eventually
    #             # reach a prediction sink.
    #             if tail_value is None:
    #                 continue

    #             candidate_value = (
    #                 float(all_arcs[arc]["cost"])
    #                 + tail_value
    #             )

    #             if (
    #                 best_value is None
    #                 or candidate_value > best_value + tol
    #             ):
    #                 best_value = candidate_value
    #                 selected_arc = arc

    #         visiting.remove(node)

    #         if best_value is None:
    #             # This node has no enabled route to a sink.
    #             L[node] = None
    #         else:
    #             L[node] = float(best_value)
    #             best_arc[node] = selected_arc

    #         return L[node]

    #     # Compute the DP value for every node, including currently
    #     # unreachable subtrees.
    #     for node in all_nodes:
    #         if not is_sink(node):
    #             dp(node)

    #     # The root must have an enabled path to a prediction sink.
    #     if L.get(root) is None:
    #         raise RuntimeError(
    #             f"Datapoint subproblem is infeasible: node {root} "
    #             "has no enabled path to a prediction sink."
    #         )

    #     subproblem_value = float(L[root])

    #     # Potentials for completely inactive nodes can be chosen freely.
    #     # Set them to zero. Their zero-capacity arcs will be handled by
    #     # positive gamma values if needed.
    #     for node in all_nodes:
    #         if L.get(node) is None:
    #             L[node] = 0.0

    #     return L, best_arc, subproblem_value

    def compute_dual_variables(self,all_arcs,L,root=1,tol=1e-8):
        """
        Recover a dual-feasible solution using pi[n] = L[n].

        For an arc (origin, destination),

            gamma[a] =
                max(0, cost[a] - L[origin] + L[destination]).
        """

        def arc_origin(arc):
            return arc[0]

        def arc_destination(arc):
            return arc[1]

        def is_sink(node):
            return (
                isinstance(node, tuple)
                and len(node) == 2
                and node[0] == "sink"
            )

        pi = {
            node: float(value)
            for node, value in L.items()
            if node != 0 and not is_sink(node)
        }

        rho = float(L[root])
        gamma = {}

        for arc, arc_data in all_arcs.items():
            if arc == (0, 1):
                gamma[arc] = 0.0
                continue

            origin = arc_origin(arc)
            destination = arc_destination(arc)

            cost = float(arc_data["cost"])
            origin_value = float(L[origin])
            destination_value = float(L[destination])

            gamma_value = max(
                0.0,
                cost - origin_value + destination_value
            )

            if gamma_value <= tol:
                gamma_value = 0.0

            gamma[arc] = gamma_value

        return pi, rho, gamma

    # def compute_dual_variables(self,all_arcs,L,root=1,tol=1e-8):
    #     """
    #     Recover a dual-feasible solution from the DP values.

    #     Dual convention:
    #         pi[n] = L[n]

    #     For an internal arc a=(n,m):
    #         gamma[a] = max(0, c[a] - L[n] + L[m])

    #     For a sink arc a=(n,t_k), L[t_k]=0:
    #         gamma[a] = max(0, c[a] - L[n])

    #     For the source equality z[s,1] = 1:
    #         rho = L[1]

    #     Returns
    #     -------
    #     pi : dict
    #         Node-potential dual variables.

    #     rho : float
    #         Free dual variable associated with z[s,1] = 1.

    #     gamma : dict
    #         Capacity dual variables for all template arcs.
    #     """
    #     def arc_head(arc):
    #         return arc[0]
    #     def arc_tail(arc):
    #         return arc[1]

    #     def is_sink(node):
    #         return (
    #             isinstance(node, tuple)
    #             and len(node) == 2
    #             and node[0] == "sink"
    #         )

    #     # Flow-conservation dual variables
    #     pi = {
    #         node: float(value)
    #         for node, value in L.items()
    #         if not is_sink(node) and node != 0
    #     }

    #     # Dual variable of z[s,1] = 1
    #     rho = float(L[root])

    #     gamma = {}

    #     for arc, arc_data in all_arcs.items():

    #         # The source arc is fixed by z[s,1] = 1.
    #         # Its upper bound is redundant, so select gamma_s1 = 0.
    #         if arc == (0, 1):
    #             gamma[arc] = 0.0
    #             continue
    #         head = arc_head(arc)
    #         tail = arc_tail(arc)
            

    #         cost = float(arc_data["cost"])
            
    #         head_L = float(L[head]) # sink values are already zero
    #         tail_L = float(L[tail])
              

    #         gamma_value = max(
    #             0.0,
    #             cost - head_L + tail_L
    #         )

    #         if gamma_value <= tol:
    #             gamma_value = 0.0

    #         gamma[arc] = gamma_value

    #     return pi, rho, gamma


    def build_benders_cut(self,all_arcs,rho,gamma,tol=1e-8):
        """
        Construct the right-hand side of the Benders optimality cut:

            g[i] <= rho + sum_a gamma[a] * u_a(master)

        The master capacity expression u_a(master) is stored in:
            all_arcs[a]["exp"]

        Returns
        -------
        cut_rhs : gurobipy.LinExpr
            Right-hand side of the Benders cut.
        """

        cut_rhs = gp.LinExpr()
        cut_rhs.addConstant(float(rho))

        for arc, gamma_value in gamma.items():

            # Source upper bound is redundant and gamma is fixed to zero
            if arc == (0, 1):
                continue

            if gamma_value <= tol:
                continue

            cut_rhs += (
                float(gamma_value)
                * all_arcs[arc]["exp"]
            )

        return cut_rhs


    def solve_subproblem_and_generate_cut(self,b,beta,i,root=1,tol=1e-8,check_duality=True):
        """
        Solve the datapoint-i subproblem using longest-path DP,
        recover the dual variables, and construct the Benders cut.

        Returns
        -------
        result : dict
            {
                "all_arcs": ...,
                "L": ...,
                "best_arc": ...,
                "pi": ...,
                "rho": ...,
                "gamma": ...,
                "subproblem_value": ...,
                "dual_value": ...,
                "cut_rhs": ...
            }
        """

        # 1. Construct full template graph and incumbent capacities
        all_arcs = self.flow_graph_construction(b=b,beta=beta,i=i)

        # 2. Solve longest-path DP
        L, best_arc, subproblem_value = self.compute_L_values(
            all_arcs=all_arcs,
            root=root,
            tol=tol
        )

        # 3. Recover dual variables
        pi, rho, gamma = self.compute_dual_variables(
            all_arcs=all_arcs,
            L=L,
            root=root,
            tol=tol
        )

        # 4. Evaluate dual objective at the current incumbent
        dual_value = float(rho)

        for arc, gamma_value in gamma.items():

            if arc == (0, 1):
                continue

            dual_value += (
                float(all_arcs[arc]["capacity"])
                * float(gamma_value)
            )

        # Strong-duality check
        if check_duality:
            if abs(subproblem_value - dual_value) > 1e-6:
                raise RuntimeError(
                    "The recovered dual solution does not satisfy "
                    "strong duality.\n"
                    f"Datapoint: {i}\n"
                    f"Primal DP value: {subproblem_value}\n"
                    f"Dual value: {dual_value}\n"
                    f"Difference: {subproblem_value - dual_value}"
                )

        # 5. Build the Benders cut RHS
        cut_rhs = self.build_benders_cut(
            all_arcs=all_arcs,
            rho=rho,
            gamma=gamma,
            tol=tol
        )

        return {
            "all_arcs": all_arcs,
            "L": L,
            "best_arc": best_arc,
            "pi": pi,
            "rho": rho,
            "gamma": gamma,
            "subproblem_value": subproblem_value,
            "dual_value": dual_value,
            "cut_rhs": cut_rhs
        }
    
    def extract_optimal_path(self,best_arc,root=1):
        """
        Recover the optimal root-to-sink path selected by the DP.
        """

        path = []
        current_node = root

        while current_node in best_arc:
            arc = best_arc[current_node]
            path.append(arc)
            current_node = arc[1]

        return path
    #related to the optimal path by subproblem
    # path = self.extract_optimal_path(
    # result["best_arc"],
    # root=1)

    # print("Subproblem value:", result["subproblem_value"])
    # print("Dual value:", result["dual_value"])
    # print("Optimal path:", path)
    # print("L values:", result["L"])

    # positive_gamma = {
    #     arc: value
    #     for arc, value in result["gamma"].items()
    #     if value > 1e-8
    # }

    # print("Positive gamma values:", positive_gamma)


    def mycallback(self, model, where):
        """
        Add one lazy Benders optimality cut for every violated datapoint
        subproblem at an integer master solution.
        """

        if where != GRB.Callback.MIPSOL:
            return

        func_start_time = time.time()
        model._callback_counter_integer += 1

        local_eps = 1e-6

        try:
            g_sol = model.cbGetSolution(model._vars_g)
            b_sol = model.cbGetSolution(model._vars_b)
            beta_sol = model.cbGetSolution(model._vars_beta)

            number_of_added_cuts = 0
            maximum_violation = 0.0

            # Every datapoint must be checked, even when g[i] <= 0.
            for i in model._master.datapoints:
                if True: #g_sol[i]+self.eta >= 1e-10:
                    result = model._master.solve_subproblem_and_generate_cut(
                        b=b_sol,
                        beta=beta_sol,
                        i=i
                    )

                    subproblem_value = result["subproblem_value"]
                    violation = float(g_sol[i]) - subproblem_value

                    maximum_violation = max(maximum_violation, violation)

                    if violation > local_eps:
                        model.cbLazy(
                            model._vars_g[i] <= result["cut_rhs"]
                        )
                        number_of_added_cuts += 1

            func_time = time.time() - func_start_time
            model._total_callback_time_integer += func_time

            if number_of_added_cuts > 0:
                model._callback_counter_integer_success += 1
                model._total_callback_time_integer_success += func_time

            # Useful during debugging.
            print(
                f"[Benders callback] "
                f"incumbent={model._callback_counter_integer}, "
                f"cuts={number_of_added_cuts}, "
                f"max_violation={maximum_violation:.6f}, "
                f"time={func_time:.4f}s",
                flush=True
            )

        except Exception as exc:
            # Do not allow Gurobi to silently ignore a callback failure and
            # continue solving an invalid relaxed master.
            model._callback_exception = exc

            print(
                "\nBenders callback failed:\n"
                f"{type(exc).__name__}: {exc}\n",
                flush=True
            )

            model.terminate()

    # def mycallback(self, model, where):
    #     '''
    #     This function is called by gurobi at every node through the branch-&-bound tree while we solve the model.
    #     Using the argument "where" we can see where the callback has been called. We are specifically interested at nodes
    #     where we get an integer solution for the master problem.
    #     When we get an integer solution for b and p, for every datapoint we solve the subproblem which is a minimum cut and
    #     check if g[i] <= value of subproblem[i]. If this is violated we add the corresponding benders constraint as lazy
    #     constraint to the master problem and proceed. Whenever we have no violated constraint! It means that we have found
    #     the optimal solution.
    #     :param model: the gurobi model we are solving.
    #     :param where: the node where the callback function is called from
    #     :return:
    #     '''
    #     data_train = model._master.X
        

    #     local_eps = 0.0001
    #     if where == GRB.Callback.MIPSOL:
    #         func_start_time = time.time()
    #         model._callback_counter_integer += 1
    #         # we need the value of b,w and g
    #         g = model.cbGetSolution(model._vars_g)
    #         b = model.cbGetSolution(model._vars_b)
    #         p = model.cbGetSolution(model._vars_p)
    #         beta = model.cbGetSolution(model._vars_beta)

    #         added_cut = 0
    #         # We only want indices that g_i is one!
    #         for i in data_train.index:
    #             g_threshold = 0 #for classification, I cannot define a threshold here 
    #             if g[i] > g_threshold:
    #                 result = model._master.solve_subproblem_and_generate_cut(b=b,beta=beta,i=i)
    #                 if ((result["subproblem_value"] + local_eps) < g[i]):  
    #                     added_cut = 1
    #                     model.cbLazy(model._master.g[i] <= result["cut_rhs"])

    #         func_end_time = time.time()
    #         func_time = func_end_time - func_start_time
    #         # print(model._callback_counter)
    #         model._total_callback_time_integer += func_time
    #         if added_cut == 1:
    #             model._callback_counter_integer_success += 1
    #             model._total_callback_time_integer_success += func_time

    def solve(self):
        start_time = time.time()

        self.model.update()
        self.model._callback_exception = None

        self.model.optimize(self.mycallback)

        if self.model._callback_exception is not None:
            raise RuntimeError(
                "The Benders optimization was terminated because the "
                "callback failed."
            ) from self.model._callback_exception

        solving_time = time.time() - start_time

        model_info = {
            "Status": self.model.Status,
            "total_solving_time": solving_time,
            "obj_value": (self.model.ObjVal if self.model.SolCount > 0 else None),
            "MIPGap": (self.model.MIPGap * 100 if self.model.SolCount > 0 else None),
            "NodeCount": self.model.NodeCount,
            "Runtime": self.model.Runtime,
            "SolCount": self.model.SolCount,
            "callback_history": None,
            "total_callback_time_integer":self.model._total_callback_time_integer,
            "total_callback_time_integer_success":self.model._total_callback_time_integer_success,
            "callback_counter_integer":self.model._callback_counter_integer,
            "callback_counter_integer_success":self.model._callback_counter_integer_success,
        }

        return model_info
    
    # def solve(self):
    #     start_time = time.time()
    #     self.model.update()
    #     self.model.optimize(self.mycallback)
    #     end_time = time.time()
    #     solving_time = end_time - start_time

    #     model_info = {
    #         "Status": self.model.Status,
    #         "total_solving_time": solving_time,
    #         "obj_value": self.model.ObjVal if self.model.SolCount > 0 else None,
    #         "MIPGap": self.model.MIPGap * 100 if self.model.SolCount > 0 else None,
    #         "NodeCount": self.model.NodeCount,
    #         "Runtime": self.model.Runtime,
    #         "SolCount": self.model.SolCount,
    #         "callback_history": None,
    #         "total_callback_time_integer": self.model._total_callback_time_integer,
    #         "total_callback_time_integer_success": self.model._total_callback_time_integer_success,
    #         "callback_counter_integer": self.model._callback_counter_integer,
    #         "callback_counter_integer_success": self.model._callback_counter_integer_success,
    #     }


    #     return model_info




    


class HybridDT():
    def __init__(self, black_box_classifier=None,depth = None, lambdaa = None, eta = None,min_transp = None , estimator = 'FlowOCT', verbosity = ['hybrid'], random_state=42, bb_pretrained=False ):
        self.depth = depth
        self.lambdaa = lambdaa
        self.min_transp = min_transp
        self.eta = eta
        self.bb_pretrained=bb_pretrained
        self.verbosity = verbosity
        self.estimator = estimator
        self.tree = Tree(depth)
        ## attributes related to the trained model
        self.b = None
        self.beta = None
        self.p = None
        self.lables = None
        self.features = None

        np.random.seed(random_state)

        # Creation of the black-box part of the Hybrid model
        if black_box_classifier is None:
            if self.bb_pretrained:
                raise ValueError("Parameters indicate that the black-box is pretrained but it is not provided!")
            print("Unspecified black_box_classifier parameter, using sklearn RandomForestClassifier() for black-box part of the model.")
            from sklearn.ensemble import RandomForestClassifier
            black_box_classifier = RandomForestClassifier()
        self.black_box_part = black_box_classifier

        # If parameters indicate that BB is pretrained, verify it now
        if self.bb_pretrained:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.black_box_part)
        
        
  
        self.is_fitted = False
        if "hybrid" in self.verbosity:
            print("Hybrid model created!")

    
    def fit(self, X, y, features, time_limit):
        # 1) (if not pretrained) Fit the black-box part of the Hybrid model
        if not self.bb_pretrained:
            if "hybrid" in self.verbosity:
                print("Training the BB part on the entire dataset")
            self.black_box_part.fit(X, y)
        else:
            if "hybrid" in self.verbosity:
                print("Not retraining BB.")
        bb_prediction = self.black_box_part.predict(X)
        bb_acc = np.mean(self.black_box_part.predict(X) == y)
        print("black box accuracy = ", bb_acc)
        # 2) Fit the interpretable part of the model using the whole primal model
        if self.estimator == 'FlowOCT':
            if "hybrid" in self.verbosity:
                print("Fitting the Decision Tree using FlowOCT...")

            myestimator = FlowOCT(X,y, features, self.tree , self.lambdaa , self.eta, self.min_transp,bb_prediction, time_limit)
            model_info = myestimator.create_primal_problem().solve()
            
            print(model_info, flush=True)

        # 2) Fit the interpretable part of the model using the Benders Decomposition 
        if self.estimator == 'BendersOCT':
            print("Fitting the Decision Tree using BendersOCT...")
            myestimator = BendersOCT(X,y, features, self.tree , self.lambdaa , self.eta,bb_prediction, time_limit)
            model_info = myestimator.create_master_problem().solve()
            print(model_info, flush=True)


        #return prediction and accuracy, pred_types
        ##########################################################
        # Preparing the output after model is fitted
        ##########################################################
        self.status = model_info['Status']
        self.optgap = model_info['MIPGap']
        self.callback_history = model_info["callback_history"]
        self.b = myestimator.model.getAttr("X", myestimator.b) # remember primal_problem.b, is a tuple dict from gurobi
        self.beta = myestimator.model.getAttr("X", myestimator.beta) #also getAttr outputs tupledict
        self.p = myestimator.model.getAttr("X", myestimator.p)
        self.is_fitted = True
        self.labels = np.append(np.unique(y), 'bb')
        self.features = features
        self.black_box_train_acc = bb_acc #uncomment if you want, I save it as I probably need standalone bb performance
        
        return self
    
    def tree_to_string(self):
        return print_tree(self)
        

    def check_is_fitted(self):
        if self.is_fitted and self.b and self.beta and self.p:
            return True
        else:
            return False


    def predict(self, X): #probably the same for benders
        if not self.check_is_fitted() :
            raise ValueError("Model is not trained yet")
        bb_prediction = self.black_box_part.predict(X)
        predicted_labels = np.array([get_predicted_value(self, X, i) for i in X.index])
        pred = (np.where(predicted_labels=='bb', bb_prediction,predicted_labels )).astype(int)
        pred_type = np.where(predicted_labels=='bb', 0,1 )
        return pred, pred_type
    

    def plot_tree(self, filename=None, show=True):
        
        G = nx.DiGraph()
        labels = {}

        def is_leaf(n):
            return self.p[n] > 0.5

        def is_branch(n):
            return any(self.b[n, f] > 0.5 for f in self.features)

        def get_branch_feature(n):
            for f in self.features:
                if self.b[n, f] > 0.5:
                    return str(f)
            return None

        def get_leaf_label(n):
            best_label = max(self.labels, key=lambda k: self.beta[n, k])
            return str(best_label)

        def add_node_recursive(n):
            # Skip pruned nodes
            if not is_leaf(n) and not is_branch(n):
                return

            G.add_node(n)

            if is_leaf(n):
                labels[n] = f"leaf\n{get_leaf_label(n)}"
                return

            labels[n] = get_branch_feature(n)

            left = int(self.tree.get_left_children(n))
            right = int(self.tree.get_right_children(n))

            if is_leaf(left) or is_branch(left):
                G.add_edge(n, left, label="0")
                add_node_recursive(left)

            if is_leaf(right) or is_branch(right):
                G.add_edge(n, right, label="1")
                add_node_recursive(right)

        add_node_recursive(1)

        # Manual tree layout
        pos = {}

        def assign_positions(n, x=0, y=0, dx=1.0):
            if n not in G.nodes:
                return

            pos[n] = (x, y)

            children = list(G.successors(n))
            if len(children) == 0:
                return

            if len(children) == 1:
                assign_positions(children[0], x, y - 1, dx / 2)
            else:
                assign_positions(children[0], x - dx, y - 1, dx / 2)
                assign_positions(children[1], x + dx, y - 1, dx / 2)

        assign_positions(1, x=0, y=0, dx=2.0)

        plt.figure(figsize=(10, 6))

        nx.draw(
            G,
            pos,
            labels=labels,
            with_labels=True,
            node_size=2500,
            font_size=10,
            arrows=False,
            node_shape="s"
        )

        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=10
        )

        plt.axis("off")
      

        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return G