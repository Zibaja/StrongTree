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
    """
    Benders decomposition for the Hybrid Decision Tree model.

    Datapoints with identical:
        - feature vectors,
        - true labels,
        - black-box predictions

    are grouped together because they induce identical subproblems.

    One master variable g[r] is created for each unique group r.
    Its objective coefficient is:

        group_size[r] / number_of_datapoints.
    """


    def __init__(self,X,y,features,tree,lambdaa,eta,bb_prediction,time_limit):

        self.X = X
        self.y = y.astype(str)

        self.datapoints = list(X.index)
        self.num_datapoints = X.shape[0]

        self.class_labels = np.unique(self.y)
        self.labels = np.append(self.class_labels, "bb")

        self.cat_features = list(features)
        self.tree = tree

        self.lambdaa = float(lambdaa)
        self.eta = float(eta)
        
        self.black_box_pred = np.asarray(bb_prediction).astype(str)

        if len(self.black_box_pred) != self.num_datapoints:
            raise ValueError("The number of black-box predictions does not match " "the number of training datapoints.")

        # Safe mapping for arbitrary dataframe indices.
        self.black_box_pred_by_i = {
            i: pred
            for i, pred in zip(
                self.datapoints,
                self.black_box_pred,)}

        # ---------------------------------------------------------
        # Build groups of identical subproblems
        # ---------------------------------------------------------
        self._build_subproblem_groups()
        #self.validate_subproblem_groups() #to validate subgroups 

        # ---------------------------------------------------------
        # Decision variables
        # ---------------------------------------------------------
        self.g = None
        self.b = None
        self.p = None
        self.beta = None

        # ---------------------------------------------------------
        # Gurobi model
        # ---------------------------------------------------------
        self.model = gp.Model("GroupedBendersOCT")

        self.model.Params.LazyConstraints = 1
        self.model.Params.Threads = 1
        self.model.Params.TimeLimit = time_limit

        # Callback statistics
        self.model._total_callback_time_integer = 0.0
        self.model._total_callback_time_integer_success = 0.0

        self.model._total_callback_time_general = 0.0
        self.model._total_callback_time_general_success = 0.0

        self.model._callback_counter_integer = 0
        self.model._callback_counter_integer_success = 0

        self.model._callback_counter_general = 0
        self.model._callback_counter_general_success = 0

        self.model._callback_exception = None

        # Give the callback access to this estimator.
        self.model._master = self

    # =============================================================
    # Datapoint grouping
    # =============================================================

    def _build_subproblem_groups(self):
        """
        Build groups of datapoints inducing identical subproblems.

        Datapoints i and j are grouped when:

            X[i, :] == X[j, :]
            y[i] == y[j]
            BB_prediction[i] == BB_prediction[j]

        Attributes created
        ------------------
        group_ids : list[int]
            Integer IDs of unique groups.

        group_members : dict[int, list]
            Datapoint indices belonging to each group.

        group_representative : dict[int, index]
            Representative datapoint for each group.

        group_size : dict[int, int]
            Number of datapoints in each group.

        datapoint_to_group : dict[index, int]
            Group containing each datapoint.

        group_signature : dict[int, tuple]
            Underlying feature/label/BB signature.
        """

        signature_to_members = defaultdict(list)

        for i in self.datapoints:
            feature_pattern = tuple(self.X.at[i, f] for f in self.cat_features)

            signature = (feature_pattern,str(self.y[i]),str(self.black_box_pred_by_i[i]),)

            signature_to_members[signature].append(i)

        self.group_ids = list(range(len(signature_to_members)))

        self.group_members = {}
        self.group_representative = {}
        self.group_size = {}
        self.group_signature = {}
        self.datapoint_to_group = {}

        for group_id, (signature, members) in enumerate(signature_to_members.items()):
            self.group_members[group_id] = list(members)
            self.group_representative[group_id] = members[0]
            self.group_size[group_id] = len(members)
            self.group_signature[group_id] = signature

            for i in members:
                self.datapoint_to_group[i] = group_id

        self.num_unique_subproblems = len(self.group_ids)

        group_sizes = np.asarray(list(self.group_size.values()),dtype=float,)

        print("\nGrouped Benders preprocessing")
        print(f"Original datapoints: "f"{self.num_datapoints}")
        print(f"Unique subproblems: "f"{self.num_unique_subproblems}")
        print(f"Reduction: "f"{self.num_datapoints - self.num_unique_subproblems}")
        print(f"Average group size: "f"{group_sizes.mean():.2f}")
        print(f"Largest group size: "f"{int(group_sizes.max())}")
        print()


    def validate_subproblem_groups(self):
        """To validate if we create the subgroups corerctly 
        """
        for group_id in self.group_ids:
            members = self.group_members[group_id]
            representative = members[0]

            rep_x = tuple(self.X.at[representative, f]for f in self.cat_features)

            rep_y = str(self.y[representative])

            rep_bb = str(self.black_box_pred_by_i[representative])

            for i in members[1:]:
                current_x = tuple(self.X.at[i, f]for f in self.cat_features)

                current_y = str(self.y[i])

                current_bb = str(self.black_box_pred_by_i[i])

                if (
                    current_x != rep_x
                    or current_y != rep_y
                    or current_bb != rep_bb
                ):
                    raise RuntimeError(
                        "Invalid subproblem grouping for "
                        f"group {group_id}.")

        print("All grouped subproblems were validated.")

    # =============================================================
    # Master problem
    # =============================================================

    def create_master_problem(self):
        """
        Create the grouped Benders master problem.

        One variable g[r] is used for each unique datapoint group r. This is the key difference with HybridDTComplete 
        """

        # ---------------------------------------------------------
        # Subproblem-value variables
        # ---------------------------------------------------------
        # Minimum possible reward:
        #   incorrect BB prediction -> -eta
        #
        # Maximum possible reward:
        #   correct prediction -> 1
        # g[r] is the objective value for the subproblem accosiated with group r
        self.g = self.model.addVars(self.group_ids,vtype=GRB.CONTINUOUS,lb=-self.eta,ub=1.0,name="g",)

        # ---------------------------------------------------------
        # Tree-design variables
        # ---------------------------------------------------------
        # b[n,f] ==1 iff at node n we branch on feature f
        self.b = self.model.addVars(self.tree.Nodes,self.cat_features,vtype=GRB.BINARY,name="b",)
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self.p = self.model.addVars(self.tree.Nodes + self.tree.Leaves,vtype=GRB.BINARY,name="p",)
        #For classification beta[n,k]=1 iff at node n we predict class k #Change by Ziba to binary instead of continuos
        self.beta = self.model.addVars(self.tree.Nodes + self.tree.Leaves,self.labels,vtype=GRB.BINARY,name="beta",)

        # We need to pass these variables to use in callback
        self.model._vars_g = self.g
        self.model._vars_b = self.b
        self.model._vars_p = self.p
        self.model._vars_beta = self.beta

        # ---------------------------------------------------------
        # Constraints
        # ---------------------------------------------------------
      
             
        # sum(b[n,f], f) + p[n] + sum(p[m], m in A(n)) = 1   forall n in Nodes
        self.model.addConstrs(
            (gp.quicksum(self.b[n, f] for f in self.cat_features) + self.p[n] + gp.quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in
            self.tree.Nodes)
        
        # p[n] + sum(p[m], m in A(n)) = 1   forall n in Leaves
        self.model.addConstrs(
            (self.p[n] + gp.quicksum(
                self.p[m] for m in self.tree.get_ancestors(n)) == 1) for n in self.tree.Leaves)
        

        # sum(beta[n,k]==p_n for all n)
        self.model.addConstrs(
            (gp.quicksum(self.beta[n, k] for k in self.labels) == self.p[n]) for n in
            self.tree.Nodes + self.tree.Leaves)

        # ---------------------------------------------------------
        # Objective : this is different from initial objective with sum g[i], here we have weighted group contribution:
        # ---------------------------------------------------------
        objective = gp.LinExpr()

        # Weighted group contribution:
        #
        # (1/N) sum_r |I_r| g[r]
        for group_id in self.group_ids:
            group_weight = (self.group_size[group_id]/ self.num_datapoints)

            objective += (
                group_weight
                * self.g[group_id])

        # Tree-complexity penalty
        objective -= self.lambdaa * gp.quicksum(
            self.b[n, f]
            for n in self.tree.Nodes
            for f in self.cat_features)

        self.model.setObjective(
            objective,
            GRB.MAXIMIZE)

        return self



    def flow_graph_construction(self,b,beta,group_id,):
        """
        Construct the full template network for one group. This includes capacity , costs of each arc , given the tree decision variabels

        Since all members of a group have identical:
            X,
            y,
            black-box prediction,

        one representative datapoint is sufficient.
        """

        if group_id not in self.group_representative:
            raise KeyError(
                f"Unknown subproblem group: {group_id}")

        i = self.group_representative[group_id]

        true_label = str(self.y[i])
        bb_label = str(self.black_box_pred_by_i[i])

        all_arcs = {}

        # Fixed source-to-root arc.
        all_arcs[(0, 1)] = {
            "capacity": 1.0,
            "cost": 0.0,
            "exp": 1.0,
        }

        left_features = [f for f in self.cat_features if self.X.at[i, f] == 0]

        right_features = [f for f in self.cat_features if self.X.at[i, f] == 1]
         
        # ---------------------------------------------------------
        # Internal tree nodes
        # ---------------------------------------------------------
        for n in self.tree.Nodes:
            left_child = int(self.tree.get_left_children(n))

            right_child = int(self.tree.get_right_children(n))

            # Left routing arc
            all_arcs[(n, left_child)] = {
                "capacity": float(sum(b[n, f] for f in left_features)),
                "cost": 0.0,
                "exp": gp.quicksum(self.b[n, f] for f in left_features),}

            # Right routing arc
            all_arcs[(n, right_child)] = {
                "capacity": float(sum(b[n, f] for f in right_features)),
                "cost": 0.0,
                "exp": gp.quicksum(self.b[n, f] for f in right_features),}

            # Prediction arcs
            for k in self.class_labels:
                k = str(k)
                sink = ("sink", k)

                bb_matches_k = int(bb_label == k) #Pred_k_i

                # DT prediction
                all_arcs[(n, sink, "DT")] = {
                    "capacity": float(beta[n, k]),
                    "cost": float(true_label == k),
                    "exp": self.beta[n, k],}

                # Black-box prediction
                all_arcs[(n, sink, "BB")] = {
                    "capacity": float(beta[n, "bb"]* bb_matches_k),
                    "cost": (float(true_label == k)- self.eta),
                    "exp": (self.beta[n, "bb"]* bb_matches_k),
                }

        # ---------------------------------------------------------
        # Terminal-depth nodes
        # ---------------------------------------------------------
        for n in self.tree.Leaves:
            for k in self.class_labels:
                k = str(k)
                sink = ("sink", k)

                bb_matches_k = int(bb_label == k)

                all_arcs[(n, sink, "DT")] = {
                    "capacity": float(beta[n, k]),
                    "cost": float(true_label == k),
                    "exp": self.beta[n, k],}

                all_arcs[(n, sink, "BB")] = {
                    "capacity": float(beta[n, "bb"]* bb_matches_k),
                    "cost": (float(true_label == k)- self.eta),
                    "exp": ( self.beta[n, "bb"]* bb_matches_k),}
                

        return all_arcs

    # =============================================================
    # Longest-path dynamic program
    # =============================================================

    def compute_L_values(self,all_arcs,root=1,tol=1e-8,):
        """
        Compute longest-path values for the fixed-master subproblem.

        Only arcs having positive incumbent capacity are enabled.
        """

        def arc_origin(arc):
            return arc[0]

        def arc_destination(arc):
            return arc[1]

        def is_sink(node):
            return (
                isinstance(node, tuple)
                and len(node) == 2
                and node[0] == "sink")

        outgoing = defaultdict(list)
        all_nodes = set()
        sinks = set()

        for arc in all_arcs:
            origin = arc_origin(arc)
            destination = arc_destination(arc)

            outgoing[origin].append(arc)

            all_nodes.add(origin)
            all_nodes.add(destination)

            if is_sink(destination):
                sinks.add(destination)

        # Sink continuation values are zero.
        L = {sink: 0.0 for sink in sinks}

        best_arc = {}
        visiting = set()

        def dp(node):
            if node in L:
                return L[node]

            if node in visiting:
                raise RuntimeError(
                    "Cycle detected in the subproblem graph. "
                    "The longest-path DP requires a DAG.")

            visiting.add(node)

            best_value = None
            selected_arc = None

            for arc in outgoing.get(node, []):
                capacity = float(all_arcs[arc]["capacity"])

                if capacity <= tol:
                    continue

                destination = arc_destination(arc)
                destination_value = dp(destination)

                if destination_value is None:
                    continue

                candidate_value = (float(all_arcs[arc]["cost"])+ destination_value)

                if (best_value is None or candidate_value > best_value + tol):
                    best_value = candidate_value
                    selected_arc = arc

            visiting.remove(node)

            if best_value is None:
                L[node] = None
            else:
                L[node] = float(best_value)
                best_arc[node] = selected_arc

            return L[node]

        # Compute values for the complete template graph using DP
        for node in all_nodes:
            if not is_sink(node):
                dp(node)

        if L.get(root) is None:
            enabled_root_arcs = [arc for arc in outgoing.get(root, []) if (float(all_arcs[arc]["capacity"])> tol)]

            raise RuntimeError(
                "Grouped datapoint subproblem is infeasible: "
                f"root {root} has no enabled path to a sink. "
                f"Enabled root arcs: {enabled_root_arcs}")

        subproblem_value = float(L[root])

        # Potentials for inactive nodes.
        for node in all_nodes:
            if L.get(node) is None:
                L[node] = 0.0

        return (L,best_arc,subproblem_value)

    # =============================================================
    # Dual recovery after solving the DP and obtaining L values
    # =============================================================

    def compute_dual_variables(self, all_arcs, L, root=1, tol=1e-8):
       
        """
        Recover a dual-feasible solution using:

            pi[n] = L[n]

        and:

            gamma[a] =
                max(
                    0,
                    cost[a]
                    - L[origin]
                    + L[destination]
                ).
        """

        def arc_origin(arc):
            return arc[0]

        def arc_destination(arc):
            return arc[1]

        def is_sink(node):
            return (
                isinstance(node, tuple)
                and len(node) == 2
                and node[0] == "sink" )
        #these are the dual variables
        pi = {node: float(value) for node, value in L.items() if (node != 0 and not is_sink(node))}

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

            gamma_value = max(0.0,cost- origin_value+ destination_value)

            if gamma_value <= tol:
                gamma_value = 0.0

            gamma[arc] = gamma_value

        return pi, rho, gamma

    def generate_strong_path_cut(
    self,
    all_arcs,
    best_arc,
    subproblem_value,
    root=1,
    tol=1e-8,
):
        """
        Generate a path-based optimal dual Benders cut.

        Potentials are selected as:

            pi[n] = subproblem_value
                    if n lies on the current optimal path,

            pi[n] = 1
                    otherwise.

        This places coefficients on:

            1. alternative routing arcs leaving the current path;
            2. improving prediction arcs at nodes on the current path.

        Entire inactive subtrees are represented by their entering
        routing arcs.

        Parameters
        ----------
        all_arcs : dict
            Full template graph. Each arc contains:
                capacity
                cost
                exp

        best_arc : dict
            DP-selected outgoing arc for each node on the current path.

        subproblem_value : float
            Current optimal subproblem value Q_i.

        root : int
            Root tree node.

        tol : float
            Numerical tolerance.

        Returns
        -------
        dict
            rho, pi, gamma, cut_rhs, path_nodes and dual_value.
        """

        def is_sink(node):
            return (
                isinstance(node, tuple)
                and len(node) == 2
                and node[0] == "sink"
            )

        q_value = float(subproblem_value)

        # A correct DT prediction already achieves the maximum value.
        if q_value >= 1.0 - tol:
            cut_rhs = gp.LinExpr()
            cut_rhs.addConstant(1.0)

            return {
                "rho": 1.0,
                "pi": {},
                "gamma": {},
                "cut_rhs": cut_rhs,
                "path_nodes": set(),
                "dual_value": 1.0,
                "is_redundant": True,
            }

        # ---------------------------------------------------------
        # Extract the current root-to-sink optimal path
        # ---------------------------------------------------------
        path_nodes = set()
        path_arcs = []

        current_node = root

        while current_node in best_arc:
            path_nodes.add(current_node)

            arc = best_arc[current_node]
            path_arcs.append(arc)

            next_node = arc[1]

            if is_sink(next_node):
                break

            current_node = next_node

        if root not in path_nodes:
            raise RuntimeError(
                "Could not recover the current optimal path "
                "from best_arc."
            )

        # ---------------------------------------------------------
        # Collect all tree nodes appearing in the template graph
        # ---------------------------------------------------------
        tree_nodes = set()

        for arc in all_arcs:
            origin = arc[0]
            destination = arc[1]

            if origin != 0 and not is_sink(origin):
                tree_nodes.add(origin)

            if destination != 0 and not is_sink(destination):
                tree_nodes.add(destination)

        # ---------------------------------------------------------
        # Path-based dual potentials
        # ---------------------------------------------------------
        pi = {
            node: (
                q_value
                if node in path_nodes
                else 1.0
            )
            for node in tree_nodes
        }

        rho = q_value
        gamma = {}

        # ---------------------------------------------------------
        # Recover capacity dual variables
        # ---------------------------------------------------------
        for arc, arc_data in all_arcs.items():
            if arc == (0, 1):
                gamma[arc] = 0.0
                continue

            origin = arc[0]
            destination = arc[1]
            cost = float(arc_data["cost"])

            if is_sink(destination):
                # Prediction arc
                gamma_value = max(
                    0.0,
                    cost - pi[origin],
                )
            else:
                # Routing arc
                gamma_value = max(
                    0.0,
                    cost
                    - pi[origin]
                    + pi[destination],
                )

            if gamma_value <= tol:
                gamma_value = 0.0

            gamma[arc] = gamma_value

        # ---------------------------------------------------------
        # Evaluate the dual objective at the incumbent
        # ---------------------------------------------------------
        dual_value = rho

        for arc, gamma_value in gamma.items():
            if arc == (0, 1):
                continue

            dual_value += (
                gamma_value
                * float(all_arcs[arc]["capacity"])
            )

        if abs(dual_value - q_value) > 1e-6:
            positive_incumbent_terms = {
                arc: {
                    "gamma": gamma[arc],
                    "capacity": all_arcs[arc]["capacity"],
                }
                for arc in gamma
                if (
                    arc != (0, 1)
                    and gamma[arc] > tol
                    and float(all_arcs[arc]["capacity"]) > tol
                )
            }

            raise RuntimeError(
                "Path-based dual is not optimal at the current "
                "master solution.\n"
                f"Subproblem value: {q_value}\n"
                f"Dual value: {dual_value}\n"
                f"Positive incumbent terms: "
                f"{positive_incumbent_terms}"
            )

        # ---------------------------------------------------------
        # Build the Benders cut
        # ---------------------------------------------------------
        cut_rhs = gp.LinExpr()
        cut_rhs.addConstant(rho)

        for arc, gamma_value in gamma.items():
            if arc == (0, 1):
                continue

            if gamma_value <= tol:
                continue

            cut_rhs += (
                gamma_value
                * all_arcs[arc]["exp"]
            )

        return {
            "rho": rho,
            "pi": pi,
            "gamma": gamma,
            "cut_rhs": cut_rhs,
            "path_nodes": path_nodes,
            "path_arcs": path_arcs,
            "dual_value": dual_value,
            "is_redundant": False,
        }
    # =============================================================
    # Benders cut construction
    # =============================================================

    def build_benders_cut(self, all_arcs, rho, gamma, tol=1e-8):
        """
        Build:

            g[r] <= rho + sum_a gamma[a] u_a(b, beta).
        """

        cut_rhs = gp.LinExpr()
        cut_rhs.addConstant(float(rho))

        for arc, gamma_value in gamma.items():
            if arc == (0, 1):
                continue

            if gamma_value <= tol:
                continue

            master_expression = all_arcs[arc]["exp"]

            cut_rhs += (float(gamma_value)* master_expression)

        return cut_rhs

    # =============================================================
    # Solve one grouped subproblem
    # =============================================================

    def solve_subproblem_and_generate_cut(self,b,beta,group_id,root=1,tol=1e-8,check_duality=True):
        """
        Solve one unique group subproblem, recover its dual, and
        construct the corresponding Benders cut.
        """
        #first, compute the flow graph to have capacities and costs for each data subgroup 
        all_arcs = self.flow_graph_construction(b=b,beta=beta,group_id=group_id,)
        #then, solve the longest path sub problem using DP
        (L,best_arc,subproblem_value,) = self.compute_L_values(all_arcs=all_arcs,root=root,tol=tol)
      
        strong_cut_result = self.generate_strong_path_cut(
            all_arcs=all_arcs,
            best_arc=best_arc,
            subproblem_value=subproblem_value,
            root=root,
            tol=tol,
        )

        pi = strong_cut_result["pi"]
        rho = strong_cut_result["rho"]
        gamma = strong_cut_result["gamma"]
        cut_rhs = strong_cut_result["cut_rhs"]
        dual_value = strong_cut_result["dual_value"]
   
        if check_duality:
            duality_difference = abs(subproblem_value- dual_value) #checking strong duality condition

            if duality_difference > 1e-6:
                representative = (self.group_representative[group_id])

                raise RuntimeError(
                    "Strong duality failed for grouped "
                    "subproblem.\n"
                    f"Group: {group_id}\n"
                    f"Representative: {representative}\n"
                    f"Group size: "
                    f"{self.group_size[group_id]}\n"
                    f"Primal value: "
                    f"{subproblem_value}\n"
                    f"Dual value: {dual_value}\n"
                    f"Difference: "
                    f"{duality_difference}"
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
            "cut_rhs": cut_rhs,
            "path_nodes": strong_cut_result["path_nodes"],
            "is_redundant": strong_cut_result["is_redundant"],
        }

    # =============================================================
    # Solve one grouped subproblem
    # =============================================================

    def solve_subproblem_and_generate_cut_with_dual(self,b,beta,group_id,root=1,tol=1e-8,check_duality=True):
        """
        Solve one unique group subproblem, recover its dual, and
        construct the corresponding Benders cut.
        """
        #first, compute the flow graph to have capacities and costs for each data subgroup 
        all_arcs = self.flow_graph_construction(b=b,beta=beta,group_id=group_id,)
        #then, solve the longest path sub problem using DP
        (L,best_arc,subproblem_value,) = self.compute_L_values(all_arcs=all_arcs,root=root,tol=tol)
        #having all the L values, compute dual variables
        (pi,rho,gamma,) = self.compute_dual_variables(all_arcs=all_arcs,L=L,root=root,tol=tol,) #uncomment


        # Dual objective at current incumbent.
        dual_value = float(rho)

        for arc, gamma_value in gamma.items():
            if arc == (0, 1):
                continue

            dual_value += (float(all_arcs[arc]["capacity"])* float(gamma_value))

        if check_duality:
            duality_difference = abs(subproblem_value- dual_value) #checking strong duality condition

            if duality_difference > 1e-6:
                representative = (self.group_representative[group_id])

                raise RuntimeError(
                    "Strong duality failed for grouped "
                    "subproblem.\n"
                    f"Group: {group_id}\n"
                    f"Representative: {representative}\n"
                    f"Group size: "
                    f"{self.group_size[group_id]}\n"
                    f"Primal value: "
                    f"{subproblem_value}\n"
                    f"Dual value: {dual_value}\n"
                    f"Difference: "
                    f"{duality_difference}"
                )
        #having dual variables, generate a cut for each sungroup [r]
        cut_rhs = self.build_benders_cut(all_arcs=all_arcs,rho=rho,gamma=gamma,tol=tol) #uncomment

        return {
            "group_id": group_id,
            "representative":self.group_representative[group_id],
            "group_size":self.group_size[group_id],
            "all_arcs": all_arcs,
            "L": L,
            "best_arc": best_arc,
            "pi": pi,
            "rho": rho,
            "gamma": gamma,
            "subproblem_value":
                subproblem_value,
            "dual_value": dual_value,
            "cut_rhs": cut_rhs,
        }
       
    # =============================================================
    # Optional path extraction
    # =============================================================

    def extract_optimal_path(self,best_arc,root=1,):
        path = []
        current_node = root

        while current_node in best_arc:
            arc = best_arc[current_node]
            path.append(arc)
            current_node = arc[1]

        return path

    # =============================================================
    # Grouped Benders callback
    # =============================================================

    def mycallback(self, model, where):
        """
        Separate one subproblem per unique group.

        Since there is one g[group_id] variable, at most one cut is
        required for each unique group at an incumbent.
        """

        if where != GRB.Callback.MIPSOL:
            return

        callback_start = time.time()
        model._callback_counter_integer += 1

        violation_tolerance = 1e-6

        try:
            g_sol = model.cbGetSolution(model._vars_g)

            b_sol = model.cbGetSolution(model._vars_b)

            beta_sol = model.cbGetSolution(model._vars_beta)

            number_of_added_cuts = 0
            maximum_violation = 0.0
            total_weight_of_violated_groups = 0
            #just remember model._master is the BendersOCT object
            # One DP solve and at most one cut per group.
            for group_id in model._master.group_ids:
                result = (model._master.solve_subproblem_and_generate_cut(b=b_sol,beta=beta_sol,group_id=group_id))

                subproblem_value = float(result["subproblem_value"])

                violation = (float(g_sol[group_id])- subproblem_value)

                maximum_violation = max(maximum_violation,violation)
                #add the cust if there is a  violation
                if violation > violation_tolerance:
                    model.cbLazy(model._vars_g[group_id]<= result["cut_rhs"])

                    number_of_added_cuts += 1

                    total_weight_of_violated_groups += (model._master.group_size[group_id])

            callback_time = (time.time() - callback_start)

            model._total_callback_time_integer += (callback_time)

            if number_of_added_cuts > 0:
                model._callback_counter_integer_success += 1

                model._total_callback_time_integer_success += (callback_time)

            print(
                "[Grouped Benders] "
                f"incumbent="
                f"{model._callback_counter_integer}, "
                f"SPs="
                f"{model._master.num_unique_subproblems}, "
                f"cuts={number_of_added_cuts}, "
                f"covered_datapoints="
                f"{total_weight_of_violated_groups}, "
                f"max_violation="
                f"{maximum_violation:.8f}, "
                f"time={callback_time:.4f}s",
                flush=True,
            )

        except Exception as exc:
            model._callback_exception = exc

            print(
                "\nGrouped Benders callback failed:\n"
                f"{type(exc).__name__}: {exc}\n",
                flush=True,
            )

            model.terminate()

    # =============================================================
    # Solve
    # =============================================================

    def solve(self):
        start_time = time.time()

        self.model.update()
        self.model._callback_exception = None

        self.model.optimize(self.mycallback)

        if self.model._callback_exception is not None:
            raise RuntimeError(
                "Benders optimization terminated because "
                "the grouped callback failed.") from self.model._callback_exception

        total_solving_time = (time.time() - start_time)

        model_info = {
            "Status": self.model.Status,

            "total_solving_time":total_solving_time,

            "obj_value": (
                self.model.ObjVal if self.model.SolCount > 0 else None),

            "MIPGap": (self.model.MIPGap * 100 if self.model.SolCount > 0 else None),

            "NodeCount":self.model.NodeCount,

            "Runtime":self.model.Runtime,

            "SolCount":self.model.SolCount,

            "callback_history":None,

            "num_original_datapoints":self.num_datapoints,

            "num_unique_subproblems":self.num_unique_subproblems,

            "subproblem_reduction_ratio": (1.0- self.num_unique_subproblems/ self.num_datapoints),

            "total_callback_time_integer":self.model._total_callback_time_integer,

            "total_callback_time_integer_success":self.model._total_callback_time_integer_success,

            "callback_counter_integer":self.model._callback_counter_integer,

            "callback_counter_integer_success":self.model._callback_counter_integer_success,
        }

        return model_info
    

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
            
            self.num_unique_subproblems = (myestimator.num_unique_subproblems)

            self.subproblem_group_sizes = dict(myestimator.group_size)

            self.datapoint_to_subproblem_group = dict(myestimator.datapoint_to_group)


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