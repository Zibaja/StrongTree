import numpy as np
from Tree import Tree

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # CP-SAT can be used without a Gurobi installation.
    gp = None
    GRB = None

from Hybrid_utils import *

import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from fractions import Fraction
import math

from ortools.sat.python import cp_model


def _validate_n_threads(n_threads):
    """Return a valid positive solver thread count."""
    if (
        isinstance(n_threads, (bool, np.bool_))
        or not isinstance(n_threads, (int, np.integer))
        or n_threads < 1
    ):
        raise ValueError("n_threads must be a positive integer.")
    return int(n_threads)


def _add_map_domain(model, variable, literals, offset=0):
    """Call AddMapDomain across old and new OR-Tools Python APIs."""
    add_map_domain = getattr(model, "add_map_domain", None)
    if add_map_domain is None:
        add_map_domain = getattr(model, "AddMapDomain", None)
    if add_map_domain is None:
        raise AttributeError(
            "This OR-Tools version does not provide add_map_domain or "
            "AddMapDomain."
        )
    return add_map_domain(variable, literals, offset)



class FlowOCT:
    def __init__(self, X,y, features, tree, lambdaa, eta, alpha,bb_prediction, time_limit, n_threads=1):
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
        self.n_threads = _validate_n_threads(n_threads)
        
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
        self.model.params.Threads = self.n_threads
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


    def __init__(self,X,y,features,tree,lambdaa,eta,bb_prediction,time_limit,n_threads=1):

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
        self.n_threads = _validate_n_threads(n_threads)

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

        self.model.Params.PreCrush = 1 #comment if you onlu want cuts in integer nodes
        self.model.Params.LazyConstraints = 1
        self.model.Params.Threads = self.n_threads
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
        #having all the L values, compute dual variables
        (pi,rho,gamma,) = self.compute_dual_variables(all_arcs=all_arcs,L=L,root=root,tol=tol,)

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
        cut_rhs = self.build_benders_cut(all_arcs=all_arcs,rho=rho,gamma=gamma,tol=tol)

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
    

    def check_fractional_capacity_consistency(
        self,
        all_arcs,
        group_id,
        tol=1e-6,
    ):
        """
        Check basic capacity identities implied by the fractional master
        solution.

        This does not replace solving the flow problem, but it catches
        inconsistent capacity construction and callback indexing.
        """

        def is_sink(node):
            return (
                isinstance(node, tuple)
                and len(node) == 2
                and node[0] == "sink"
            )

        issues = []

        all_tree_nodes = (
            self.tree.Nodes + self.tree.Leaves
        )

        for n in all_tree_nodes:
            outgoing_capacity = 0.0

            for arc, arc_data in all_arcs.items():
                if arc == (0, 1):
                    continue

                if arc[0] == n:
                    outgoing_capacity += float(
                        arc_data["capacity"]
                    )

            if outgoing_capacity < -tol:
                issues.append(
                    f"Node {n}: negative outgoing capacity "
                    f"{outgoing_capacity}"
                )

            if outgoing_capacity > 1.0 + tol:
                issues.append(
                    f"Node {n}: outgoing capacity exceeds 1: "
                    f"{outgoing_capacity}"
                )

        root_outgoing_capacity = sum(
            float(arc_data["capacity"])
            for arc, arc_data in all_arcs.items()
            if arc != (0, 1) and arc[0] == 1
        )

        if abs(root_outgoing_capacity - 1.0) > tol:
            issues.append(
                "Root outgoing capacity is not one: "
                f"{root_outgoing_capacity}"
            )

        if issues:
            representative = self.group_representative[group_id]

            raise RuntimeError(
                "Invalid fractional capacity construction.\n"
                f"Group: {group_id}\n"
                f"Representative: {representative}\n"
                + "\n".join(issues)
            )
        
    def check_fractional_flow_feasibility(
    self,
    all_arcs,
    group_id,
    root=1,
):
        """
        Solve a zero-objective flow LP to verify that one unit can travel
        from the root to prediction sinks under fractional capacities.
        """

        def origin(arc):
            return arc[0]

        def destination(arc):
            return arc[1]

        def is_sink(node):
            return (
                isinstance(node, tuple)
                and len(node) == 2
                and node[0] == "sink"
            )

        arcs = [
            arc
            for arc in all_arcs
            if arc != (0, 1)
        ]

        feasibility_model = gp.Model(
            f"flow_feasibility_{group_id}"
        )

        feasibility_model.Params.OutputFlag = 0
        feasibility_model.Params.Threads = self.n_threads

        z = {
            arc: feasibility_model.addVar(
                lb=0.0,
                ub=max(
                    0.0,
                    float(all_arcs[arc]["capacity"]),
                ),
                name=f"z_{arc_number}",
            )
            for arc_number, arc in enumerate(arcs)
        }

        feasibility_model.update()

        tree_nodes = (
            self.tree.Nodes + self.tree.Leaves
        )

        for n in tree_nodes:
            incoming = gp.quicksum(
                z[arc]
                for arc in arcs
                if destination(arc) == n
            )

            outgoing = gp.quicksum(
                z[arc]
                for arc in arcs
                if origin(arc) == n
            )

            rhs = 1.0 if n == root else 0.0

            feasibility_model.addConstr(
                outgoing - incoming == rhs
            )

        feasibility_model.setObjective(
            0.0,
            GRB.MINIMIZE,
        )

        feasibility_model.optimize()

        if feasibility_model.Status != GRB.OPTIMAL:
            representative = (
                self.group_representative[group_id]
            )

            raise RuntimeError(
                "The fractional master solution produces an "
                "infeasible flow subproblem.\n"
                f"Group: {group_id}\n"
                f"Representative: {representative}\n"
                "This requires Benders feasibility cuts, not only "
                "optimality cuts."
            )
    
    def solve_fractional_subproblem_and_generate_cut(
        self,
        b_values,
        beta_values,
        group_id,
        root=1,
        tol=1e-8,
    ):
        """
        Solve the explicit dual of the fractional max-cost-flow
        subproblem and generate a globally valid Benders cut.
        """

        all_arcs = self.flow_graph_construction(
            b=b_values,
            beta=beta_values,
            group_id=group_id,
        )

        self.check_fractional_capacity_consistency(
            all_arcs=all_arcs,
            group_id=group_id,
        )

        self.check_fractional_flow_feasibility(
            all_arcs=all_arcs,
            group_id=group_id,
        )

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

        network_arcs = [
            arc
            for arc in all_arcs
            if arc != (0, 1)
        ]

        tree_nodes = list(
            self.tree.Nodes + self.tree.Leaves
        )

        dual_model = gp.Model(
            f"fractional_dual_group_{group_id}"
        )

        dual_model.Params.OutputFlag = 0
        dual_model.Params.Threads = self.n_threads
        dual_model.Params.Method = 1

        # Make infeasible/unbounded statuses distinguishable.
        dual_model.Params.DualReductions = 0
        dual_model.Params.InfUnbdInfo = 1

        # All possible downstream rewards lie in [-eta, 1].
        pi = {
            n: dual_model.addVar(
                lb=-float(self.eta),
                ub=1.0,
                vtype=GRB.CONTINUOUS,
                name=f"pi_{n}",
            )
            for n in tree_nodes
        }

        gamma = {}

        for arc_number, arc in enumerate(network_arcs):
            gamma[arc] = dual_model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"gamma_{arc_number}",
            )

        dual_model.update()

        # -------------------------------------------------------------
        # Dual feasibility
        # -------------------------------------------------------------
        for arc_number, arc in enumerate(network_arcs):
            origin = arc_origin(arc)
            destination = arc_destination(arc)
            cost = float(all_arcs[arc]["cost"])

            if is_sink(destination):
                dual_model.addConstr(
                    pi[origin] + gamma[arc] >= cost,
                    name=f"prediction_dual_{arc_number}",
                )
            else:
                dual_model.addConstr(
                    pi[origin]
                    - pi[destination]
                    + gamma[arc]
                    >= cost,
                    name=f"routing_dual_{arc_number}",
                )

        # -------------------------------------------------------------
        # Dual objective
        # -------------------------------------------------------------
        dual_objective = (
            pi[root]
            + gp.quicksum(
                float(all_arcs[arc]["capacity"])
                * gamma[arc]
                for arc in network_arcs
            )
        )

        dual_model.setObjective(
            dual_objective,
            GRB.MINIMIZE,
        )

        dual_model.optimize()

        if dual_model.Status == GRB.INF_OR_UNBD:
            dual_model.reset()
            dual_model.optimize()

        if dual_model.Status != GRB.OPTIMAL:
            status_name = {
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.UNBOUNDED: "UNBOUNDED",
                GRB.INF_OR_UNBD: "INF_OR_UNBD",
            }.get(
                dual_model.Status,
                str(dual_model.Status),
            )

            model_filename = (
                f"fractional_dual_group_{group_id}.lp"
            )

            dual_model.write(model_filename)

            raise RuntimeError(
                "Fractional dual subproblem was not solved "
                "optimally.\n"
                f"Group: {group_id}\n"
                f"Status: {status_name}\n"
                f"Model written to: {model_filename}"
            )

        subproblem_value = float(
            dual_model.ObjVal
        )

        pi_values = {
            n: float(pi[n].X)
            for n in tree_nodes
        }

        gamma_values = {}

        for arc in network_arcs:
            value = float(gamma[arc].X)

            if value <= tol:
                value = 0.0

            gamma_values[arc] = value

        # -------------------------------------------------------------
        # Construct the Benders cut
        # -------------------------------------------------------------
        cut_rhs = gp.LinExpr()
        cut_rhs.addConstant(
            pi_values[root]
        )

        for arc in network_arcs:
            gamma_value = gamma_values[arc]

            if gamma_value <= tol:
                continue

            cut_rhs += (
                gamma_value
                * all_arcs[arc]["exp"]
            )

        # -------------------------------------------------------------
        # Verify tightness at the current fractional solution
        # -------------------------------------------------------------
        cut_value_at_current_point = (
            pi_values[root]
        )

        for arc in network_arcs:
            cut_value_at_current_point += (
                gamma_values[arc]
                * float(all_arcs[arc]["capacity"])
            )

        tightness_error = abs(
            cut_value_at_current_point
            - subproblem_value
        )

        if tightness_error > 1e-6:
            raise RuntimeError(
                "Fractional Benders cut is not tight at "
                "the separation point.\n"
                f"Group: {group_id}\n"
                f"Dual value: {subproblem_value}\n"
                f"Cut evaluation: "
                f"{cut_value_at_current_point}\n"
                f"Error: {tightness_error}"
            )

        return {
            "group_id": group_id,
            "all_arcs": all_arcs,
            "pi": pi_values,
            "gamma": gamma_values,
            "subproblem_value": subproblem_value,
            "cut_rhs": cut_rhs,
        }



    # =============================================================
    # Grouped Benders callback
    # =============================================================

    def mycallback_old(self, model, where): #this is for before adding cuts in the root node
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


    def mycallback(self, model, where):
        """
        Branch-and-Benders-cut callback.

        MIPSOL:
            Separate integer master solutions with the fast longest-path DP
            and add lazy constraints.

        MIPNODE:
            Separate selected fractional master solutions using the
            max-cost-flow LP and add user cuts.
        """

        violation_tolerance = 1e-6

        try:
            # =========================================================
            # Integer incumbent separation
            # =========================================================
            # if where == GRB.Callback.MIPNODE:
            #     return
            if where == GRB.Callback.MIPSOL:
                callback_start = time.time()
                model._callback_counter_integer += 1

                g_sol = model.cbGetSolution(model._vars_g)
                b_sol = model.cbGetSolution(model._vars_b)
                beta_sol = model.cbGetSolution(model._vars_beta)

                violated_groups = []

                for group_id in model._master.group_ids:
                    result = (
                        model._master
                        .solve_subproblem_and_generate_cut(
                            b=b_sol,
                            beta=beta_sol,
                            group_id=group_id,
                        )
                    )

                    violation = (
                        float(g_sol[group_id])
                        - float(result["subproblem_value"])
                    )

                    if violation > violation_tolerance:
                        violated_groups.append(
                            (
                                violation,
                                group_id,
                                result,
                            )
                        )

                # Add the most violated first.
                violated_groups.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )

                number_of_added_cuts = 0

                for violation, group_id, result in violated_groups:
                    model.cbLazy(
                        model._vars_g[group_id]
                        <= result["cut_rhs"]
                    )

                    number_of_added_cuts += 1

                callback_time = time.time() - callback_start

                model._total_callback_time_integer += callback_time

                if number_of_added_cuts > 0:
                    model._callback_counter_integer_success += 1
                    model._total_callback_time_integer_success += (
                        callback_time
                    )

                return

            # =========================================================
            # Fractional-node separation
            # =========================================================
            if where == GRB.Callback.MIPNODE:
                #return  #if you dont want fractional cuts
                node_status = model.cbGet(
                    GRB.Callback.MIPNODE_STATUS
                )

                if node_status != GRB.OPTIMAL:
                    return

                node_count = int(
                    model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                )

                # Initially perform fractional separation only at the root.
                # This prevents excessive LP subproblem overhead.
                if node_count > 0:  #node_count <= 10 if there was a huge bound imporvment in root
                    return

                # Avoid repeatedly separating essentially the same root
                # solution too many times.
                model._callback_counter_general += 1

                max_root_separation_rounds = 50

                if (
                    model._callback_counter_general
                    > max_root_separation_rounds
                ):
                    return

                callback_start = time.time()

                g_rel = model.cbGetNodeRel(model._vars_g)
                b_rel = model.cbGetNodeRel(model._vars_b)
                beta_rel = model.cbGetNodeRel(model._vars_beta)

                violated_groups = []

                for group_id in model._master.group_ids:
                    result = (
                        model._master
                        .solve_fractional_subproblem_and_generate_cut(
                            b_values=b_rel,
                            beta_values=beta_rel,
                            group_id=group_id,
                        )
                    )

                    violation = (
                        float(g_rel[group_id])
                        - float(result["subproblem_value"])
                    )

                    if violation > violation_tolerance:
                        violated_groups.append(
                            (
                                violation,
                                group_id,
                                result,
                            )
                        )

                violated_groups.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )

                # Start with a limited number of fractional cuts per round.
                max_fractional_cuts_per_round = 20

                number_of_added_cuts = 0

                for (
                    violation,
                    group_id,
                    result,
                ) in violated_groups[
                    :max_fractional_cuts_per_round
                ]:
                    model.cbCut(
                        model._vars_g[group_id]
                        <= result["cut_rhs"]
                    )

                    number_of_added_cuts += 1

                callback_time = time.time() - callback_start

                model._total_callback_time_general += callback_time

                if number_of_added_cuts > 0:
                    model._callback_counter_general_success += 1
                    model._total_callback_time_general_success += (
                        callback_time
                    )

                print(
                    "[Root fractional Benders] "
                    f"round={model._callback_counter_general}, "
                    f"violated={len(violated_groups)}, "
                    f"cuts={number_of_added_cuts}, "
                    f"time={callback_time:.3f}s",
                    flush=True,
                )

        except Exception as exc:
            model._callback_exception = exc

            print(
                "\nBranch-and-Benders callback failed:\n"
                f"{type(exc).__name__}: {exc}",
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
    



class _CPSATIncumbentCallback(cp_model.CpSolverSolutionCallback):
    """Collect incumbent statistics in the same format as FlowOCT."""

    def __init__(
        self,
        correct_vars,
        deferred_vars,
        pattern_sizes,
        split_vars,
        num_datapoints,
        lambdaa,
        eta,
        objective_scale,
    ):
        super().__init__()
        self.correct_vars = list(correct_vars)
        self.deferred_vars = list(deferred_vars)
        self.pattern_sizes = list(pattern_sizes)
        self.split_vars = list(split_vars)
        self.num_datapoints = int(num_datapoints)
        self.lambdaa = float(lambdaa)
        self.eta = float(eta)
        self.objective_scale = int(objective_scale)
        self.history = []

    def on_solution_callback(self):
        correct = sum(self.Value(var) for var in self.correct_vars)
        deferred = sum(
            size * self.Value(var)
            for size, var in zip(self.pattern_sizes, self.deferred_vars)
        )
        splits = sum(self.Value(var) for var in self.split_vars)

        train_acc = correct / self.num_datapoints
        train_transparency = 1.0 - deferred / self.num_datapoints
        objective = (
            train_acc
            - self.lambdaa * splits
            - self.eta * deferred / self.num_datapoints
        )
        best_bound = self.BestObjectiveBound() / (
            self.objective_scale * self.num_datapoints
        )

        if abs(objective) > 1e-10:
            gap = max(0.0, (best_bound - objective) / abs(objective))
        else:
            gap = None

        self.history.append(
            {
                "time": self.WallTime(),
                "obj": objective,
                "best_bound": best_bound,
                "gap": gap,
                "train_acc": train_acc,
                "train_transparency": train_transparency,
                "num_splits": splits,
            }
        )


class CPSATOCT:
    """Direct OR-Tools CP-SAT formulation of the hybrid decision tree.

    The public constructor and the extracted ``b``, ``p`` and ``beta``
    solutions mirror ``FlowOCT``.  Routing is modeled without graph flows:
    each aggregated binary pattern has one current-node index per depth, and
    ``AddElement`` constraints retrieve the state and split feature of that
    node.
    """

    OFF = 0
    SPLIT = 1
    BLACK_BOX = 2
    CLASS_BASE = 3

    def __init__(
        self,
        X,
        y,
        features,
        tree,
        lambdaa,
        eta,
        alpha,
        bb_prediction,
        time_limit,
        n_threads=1,
    ):
        self.X = X
        self.features = list(features)
        self.tree = tree
        self.lambdaa = 0.0 if lambdaa is None else float(lambdaa)
        self.eta = 0.0 if eta is None else float(eta)
        self.alpha = 0.0 if alpha is None else float(alpha)
        self.time_limit = None if time_limit is None else float(time_limit)
        self.n_threads = _validate_n_threads(n_threads)

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("min_transp must lie in [0, 1].")
        if self.lambdaa < 0.0:
            raise ValueError("lambdaa must be nonnegative.")
        if self.eta < 0.0:
            raise ValueError("eta must be nonnegative.")
        if not self.features:
            raise ValueError("At least one binary feature is required.")

        self.datapoints = list(X.index)
        self.num_datapoints = len(self.datapoints)
        if self.num_datapoints == 0:
            raise ValueError("The training set is empty.")

        self.X_array = np.asarray(X.loc[:, self.features])
        if not np.all(np.isin(self.X_array, [0, 1])):
            unexpected = np.unique(
                self.X_array[~np.isin(self.X_array, [0, 1])]
            )
            raise ValueError(
                "CPSATOCT requires binary features encoded as 0/1. "
                f"Unexpected values include {unexpected[:10]}."
            )
        self.X_array = self.X_array.astype(int)

        if hasattr(y, "loc"):
            try:
                y_values = y.loc[X.index]
            except (KeyError, TypeError):
                y_values = y
        else:
            y_values = y
        self.y_array = np.asarray(y_values).astype(str)
        if len(self.y_array) != self.num_datapoints:
            raise ValueError("The number of labels does not match X.")

        self.black_box_pred = np.asarray(bb_prediction).astype(str)
        if len(self.black_box_pred) != self.num_datapoints:
            raise ValueError(
                "The number of black-box predictions does not match X."
            )

        self.class_labels = [str(k) for k in np.unique(self.y_array)]
        self.labels = self.class_labels + ["bb"]
        self.class_to_index = {
            label: index for index, label in enumerate(self.class_labels)
        }

        unknown_bb_labels = sorted(
            str(label)
            for label in set(self.black_box_pred) - set(self.class_labels)
        )
        if unknown_bb_labels:
            raise ValueError(
                "Every black-box prediction must be one of the training "
                f"classes. Unknown values: {unknown_bb_labels[:10]}."
            )

        self.internal_nodes = [int(node) for node in self.tree.Nodes]
        self.terminal_nodes = [int(node) for node in self.tree.Leaves]
        self.all_nodes = self.internal_nodes + self.terminal_nodes
        self.internal_node_set = set(self.internal_nodes)
        self.depth = self._infer_depth()
        self._validate_tree_template()
        self._build_pattern_groups()

        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        self.node_state = {}
        self.node_feature = {}
        self.node_active = {}
        self.node_split = {}
        self.branch_on = {}

        self.pattern_position = {}
        self.pattern_alive = {}
        self.pattern_stop = {}
        self.pattern_final_state = {}
        self.pattern_deferred = {}
        self.pattern_correct_count = {}

        self.b = None
        self.p = None
        self.beta = None
        self._created = False
        self._objective_scale = 1

    def _infer_depth(self):
        if hasattr(self.tree, "depth"):
            try:
                return int(self.tree.depth)
            except (TypeError, ValueError):
                pass
        return int(math.floor(math.log2(max(self.all_nodes))))

    def _validate_tree_template(self):
        expected_internal = list(range(1, 2 ** self.depth))
        expected_terminal = list(
            range(2 ** self.depth, 2 ** (self.depth + 1))
        )
        if (
            self.internal_nodes != expected_internal
            or self.terminal_nodes != expected_terminal
        ):
            raise ValueError(
                "CPSATOCT expects the same complete heap-indexed template "
                "as FlowOCT: Nodes=1,...,2^d-1 and "
                "Leaves=2^d,...,2^(d+1)-1."
            )

    def _build_pattern_groups(self):
        """Aggregate rows sharing (feature vector, black-box prediction)."""
        signature_to_rows = defaultdict(list)
        for row in range(self.num_datapoints):
            signature = (
                tuple(int(value) for value in self.X_array[row]),
                str(self.black_box_pred[row]),
            )
            signature_to_rows[signature].append(row)

        self.pattern_ids = list(range(len(signature_to_rows)))
        self.pattern_x = {}
        self.pattern_bb_label = {}
        self.pattern_size = {}
        self.pattern_label_counts = {}
        self.pattern_members = {}
        self.datapoint_to_pattern = {}

        for pattern_id, (signature, rows) in enumerate(
            signature_to_rows.items()
        ):
            feature_pattern, bb_label = signature
            label_counts = [0] * len(self.class_labels)
            for row in rows:
                label_counts[
                    self.class_to_index[str(self.y_array[row])]
                ] += 1
                self.datapoint_to_pattern[
                    self.datapoints[row]
                ] = pattern_id

            self.pattern_x[pattern_id] = list(feature_pattern)
            self.pattern_bb_label[pattern_id] = str(bb_label)
            self.pattern_size[pattern_id] = len(rows)
            self.pattern_label_counts[pattern_id] = label_counts
            self.pattern_members[pattern_id] = [
                self.datapoints[row] for row in rows
            ]

        self.num_unique_patterns = len(self.pattern_ids)

    def create_primal_problem(self):
        if self._created:
            return self
        self._create_tree_variables()
        self._create_routing_variables()
        self._add_transparency_constraint()
        self._set_objective()
        self._created = True
        return self

    def _create_tree_variables(self):
        num_features = len(self.features)
        no_feature = num_features
        max_state = self.CLASS_BASE + len(self.class_labels) - 1

        for node in self.all_nodes:
            state = self.model.NewIntVar(0, max_state, f"state[{node}]")
            feature = self.model.NewIntVar(
                0, no_feature, f"feature[{node}]"
            )
            active = self.model.NewBoolVar(f"active[{node}]")
            split = self.model.NewBoolVar(f"split[{node}]")

            allowed_states = [(self.OFF, 0, 0)]
            if node in self.internal_node_set:
                allowed_states.append((self.SPLIT, 1, 1))
            allowed_states.append((self.BLACK_BOX, 1, 0))
            allowed_states.extend(
                (self.CLASS_BASE + class_index, 1, 0)
                for class_index in range(len(self.class_labels))
            )
            self.model.AddAllowedAssignments(
                [state, active, split], allowed_states
            )

            if node in self.internal_node_set:
                feature_literals = []
                for feature_index in range(num_features):
                    literal = self.model.NewBoolVar(
                        f"branch_on[{node},{feature_index}]"
                    )
                    self.branch_on[node, feature_index] = literal
                    feature_literals.append(literal)

                # branch_on[node, f] <=> feature[node] == f.  The remaining
                # value ``no_feature`` is selected when all literals are
                # false, while the equality below keeps that condition tied
                # explicitly to whether the node splits.
                _add_map_domain(self.model, feature, feature_literals)
                self.model.Add(sum(feature_literals) == split)
            else:
                self.model.Add(feature == no_feature)

            self.node_state[node] = state
            self.node_feature[node] = feature
            self.node_active[node] = active
            self.node_split[node] = split

        self.model.Add(self.node_active[1] == 1)
        for node in self.internal_nodes:
            self.model.Add(
                self.node_active[2 * node] == self.node_split[node]
            )
            self.model.Add(
                self.node_active[2 * node + 1] == self.node_split[node]
            )

        # Safe strengthening for binary features: an optimal tree never needs
        # to test the same feature twice on one root-to-node path.
        for node in self.internal_nodes:
            path = [node]
            ancestor = node // 2
            while ancestor >= 1:
                if ancestor in self.internal_node_set:
                    path.append(ancestor)
                ancestor //= 2
            for feature_index in range(num_features):
                self.model.AddAtMostOne(
                    self.branch_on[path_node, feature_index]
                    for path_node in path
                )

    def _create_routing_variables(self):
        max_state = self.CLASS_BASE + len(self.class_labels) - 1
        no_feature = len(self.features)

        states_by_level = []
        features_by_level = []
        splits_by_level = []
        for level in range(self.depth + 1):
            level_nodes = list(
                range(2 ** level, 2 ** (level + 1))
            )
            states_by_level.append(
                [self.node_state[node] for node in level_nodes]
            )
            features_by_level.append(
                [self.node_feature[node] for node in level_nodes]
            )
            splits_by_level.append(
                [self.node_split[node] for node in level_nodes]
            )

        for pattern_id in self.pattern_ids:
            positions = [
                self.model.NewIntVar(
                    0,
                    2 ** level - 1,
                    f"position[{pattern_id},{level}]",
                )
                for level in range(self.depth + 1)
            ]
            alive = [
                self.model.NewBoolVar(
                    f"alive[{pattern_id},{level}]"
                )
                for level in range(self.depth + 1)
            ]
            current_states = []
            stops = []

            self.model.Add(positions[0] == 0)
            self.model.Add(alive[0] == 1)

            for level in range(self.depth + 1):
                current_state = self.model.NewIntVar(
                    0,
                    max_state,
                    f"current_state[{pattern_id},{level}]",
                )
                current_split = self.model.NewBoolVar(
                    f"current_split[{pattern_id},{level}]"
                )
                self.model.AddElement(
                    positions[level],
                    states_by_level[level],
                    current_state,
                )
                self.model.AddElement(
                    positions[level],
                    splits_by_level[level],
                    current_split,
                )
                self.model.Add(current_state != self.OFF).OnlyEnforceIf(
                    alive[level]
                )

                current_states.append(current_state)
                self.pattern_position[pattern_id, level] = positions[level]
                self.pattern_alive[pattern_id, level] = alive[level]

                if level < self.depth:
                    current_feature = self.model.NewIntVar(
                        0,
                        no_feature,
                        f"current_feature[{pattern_id},{level}]",
                    )
                    go_right = self.model.NewBoolVar(
                        f"go_right[{pattern_id},{level}]"
                    )
                    self.model.AddElement(
                        positions[level],
                        features_by_level[level],
                        current_feature,
                    )
                    self.model.AddElement(
                        current_feature,
                        self.pattern_x[pattern_id] + [0],
                        go_right,
                    )

                    self.model.Add(alive[level + 1] <= alive[level])
                    self.model.Add(
                        alive[level + 1] <= current_split
                    )
                    self.model.Add(
                        alive[level + 1]
                        >= alive[level] + current_split - 1
                    )

                    stop = self.model.NewBoolVar(
                        f"stop[{pattern_id},{level}]"
                    )
                    self.model.Add(
                        stop + alive[level + 1] == alive[level]
                    )
                    self.model.Add(
                        positions[level + 1]
                        == 2 * positions[level] + go_right
                    ).OnlyEnforceIf(alive[level + 1])
                    self.model.Add(
                        positions[level + 1] == 0
                    ).OnlyEnforceIf(alive[level + 1].Not())
                else:
                    stop = alive[level]

                stops.append(stop)
                self.pattern_stop[pattern_id, level] = stop

            self.model.AddExactlyOne(stops)

            final_state = self.model.NewIntVar(
                self.BLACK_BOX,
                max_state,
                f"final_state[{pattern_id}]",
            )
            for level, stop in enumerate(stops):
                self.model.Add(
                    final_state == current_states[level]
                ).OnlyEnforceIf(stop)

            deferred = self.model.NewBoolVar(
                f"deferred[{pattern_id}]"
            )
            self.model.Add(
                final_state == self.BLACK_BOX
            ).OnlyEnforceIf(deferred)
            self.model.Add(
                final_state != self.BLACK_BOX
            ).OnlyEnforceIf(deferred.Not())

            label_counts = self.pattern_label_counts[pattern_id]
            bb_class_index = self.class_to_index[
                self.pattern_bb_label[pattern_id]
            ]
            gain_by_state = [
                0,
                0,
                label_counts[bb_class_index],
                *label_counts,
            ]
            correct_count = self.model.NewIntVar(
                0,
                self.pattern_size[pattern_id],
                f"correct_count[{pattern_id}]",
            )
            self.model.AddElement(
                final_state, gain_by_state, correct_count
            )

            self.pattern_final_state[pattern_id] = final_state
            self.pattern_deferred[pattern_id] = deferred
            self.pattern_correct_count[pattern_id] = correct_count

    def _add_transparency_constraint(self):
        self.total_deferred_count = sum(
            self.pattern_size[pattern_id]
            * self.pattern_deferred[pattern_id]
            for pattern_id in self.pattern_ids
        )
        max_deferred_fraction = (
            Fraction(1, 1)
            - Fraction(str(self.alpha)).limit_denominator(10**6)
        )
        self.model.Add(
            max_deferred_fraction.denominator
            * self.total_deferred_count
            <= max_deferred_fraction.numerator * self.num_datapoints
        )

    def _set_objective(self):
        lambda_fraction = Fraction(str(self.lambdaa)).limit_denominator(
            10**6
        )
        eta_fraction = Fraction(str(self.eta)).limit_denominator(10**6)
        self._objective_scale = math.lcm(
            lambda_fraction.denominator,
            eta_fraction.denominator,
        )

        self.total_correct_count = sum(
            self.pattern_correct_count[pattern_id]
            for pattern_id in self.pattern_ids
        )
        self.total_split_count = sum(
            self.node_split[node] for node in self.internal_nodes
        )

        correct_coefficient = self._objective_scale
        split_coefficient = -(
            self._objective_scale
            * self.num_datapoints
            * lambda_fraction.numerator
            // lambda_fraction.denominator
        )
        deferred_coefficient = -(
            self._objective_scale
            * eta_fraction.numerator
            // eta_fraction.denominator
        )

        max_abs_objective = (
            abs(correct_coefficient) * self.num_datapoints
            + abs(split_coefficient) * len(self.internal_nodes)
            + abs(deferred_coefficient) * self.num_datapoints
        )
        if max_abs_objective >= 2**62:
            raise OverflowError(
                "The integer-scaled CP-SAT objective is too large. Use "
                "lambdaa and eta with fewer decimal places."
            )

        self.integer_objective = (
            correct_coefficient * self.total_correct_count
            + split_coefficient * self.total_split_count
            + deferred_coefficient * self.total_deferred_count
        )
        self.model.Maximize(self.integer_objective)

    def solve(self):
        if not self._created:
            self.create_primal_problem()

        self.solver.parameters.num_search_workers = self.n_threads
        if self.time_limit is not None:
            self.solver.parameters.max_time_in_seconds = self.time_limit

        callback = _CPSATIncumbentCallback(
            correct_vars=[
                self.pattern_correct_count[pattern_id]
                for pattern_id in self.pattern_ids
            ],
            deferred_vars=[
                self.pattern_deferred[pattern_id]
                for pattern_id in self.pattern_ids
            ],
            pattern_sizes=[
                self.pattern_size[pattern_id]
                for pattern_id in self.pattern_ids
            ],
            split_vars=[
                self.node_split[node] for node in self.internal_nodes
            ],
            num_datapoints=self.num_datapoints,
            lambdaa=self.lambdaa,
            eta=self.eta,
            objective_scale=self._objective_scale,
        )

        start_time = time.time()
        status = self.solver.Solve(self.model, callback)
        total_solving_time = time.time() - start_time
        status_name = self.solver.StatusName(status)
        has_solution = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        if has_solution:
            self._extract_solution()
            objective_value = self.solver.ObjectiveValue() / (
                self._objective_scale * self.num_datapoints
            )
            best_bound = self.solver.BestObjectiveBound() / (
                self._objective_scale * self.num_datapoints
            )
            if status == cp_model.OPTIMAL:
                relative_gap = 0.0
            elif abs(objective_value) > 1e-10:
                relative_gap = (
                    max(0.0, best_bound - objective_value)
                    / abs(objective_value)
                    * 100.0
                )
            else:
                relative_gap = None
        else:
            objective_value = None
            best_bound = None
            relative_gap = None

        return {
            "Status": int(status),
            "StatusName": status_name,
            "obj_value": objective_value,
            "ObjBound": best_bound,
            "MIPGap": relative_gap,
            "NodeCount": self.solver.NumBranches(),
            "Runtime": self.solver.WallTime(),
            "SolCount": int(has_solution),
            "callback_history": callback.history,
            "num_original_datapoints": self.num_datapoints,
            "num_unique_patterns": self.num_unique_patterns,
            "pattern_reduction_ratio": (
                1.0
                - self.num_unique_patterns / self.num_datapoints
            ),
            "NumConflicts": self.solver.NumConflicts(),
            "NumBranches": self.solver.NumBranches(),
            "total_solving_time": total_solving_time,
        }

    def _extract_solution(self):
        self.b = {}
        self.p = {}
        self.beta = {}

        for node in self.all_nodes:
            state = self.solver.Value(self.node_state[node])

            if node in self.internal_node_set:
                for feature_index, feature_name in enumerate(self.features):
                    self.b[node, feature_name] = float(
                        self.solver.Value(
                            self.branch_on[node, feature_index]
                        )
                    )

            self.p[node] = float(state >= self.BLACK_BOX)

            for label in self.class_labels:
                class_index = self.class_to_index[label]
                self.beta[node, label] = float(
                    state == self.CLASS_BASE + class_index
                )
            self.beta[node, "bb"] = float(state == self.BLACK_BOX)

class HybridDT():
    def __init__(self, black_box_classifier=None,depth = None, lambdaa = None, eta = None,min_transp = None , estimator = 'FlowOCT', verbosity = ['hybrid'], random_state=42, bb_pretrained=False, n_threads=1 ):
        self.depth = depth
        self.lambdaa = lambdaa
        self.min_transp = min_transp
        self.eta = eta
        self.bb_pretrained=bb_pretrained
        self.verbosity = verbosity
        self.estimator = estimator
        self.n_threads = _validate_n_threads(n_threads)
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
        # 2) Fit the interpretable part.
        if self.estimator == 'FlowOCT':
            if gp is None:
                raise ImportError(
                    "FlowOCT requires gurobipy. Use estimator='CPSAT' to "
                    "run the OR-Tools formulation without Gurobi."
                )
            if "hybrid" in self.verbosity:
                print("Fitting the Decision Tree using FlowOCT...")

            myestimator = FlowOCT(
                X, y, features, self.tree, self.lambdaa, self.eta,
                self.min_transp, bb_prediction, time_limit, self.n_threads
            )
            model_info = myestimator.create_primal_problem().solve()

        elif self.estimator == 'BendersOCT':
            if gp is None:
                raise ImportError(
                    "BendersOCT requires gurobipy. Use estimator='CPSAT' "
                    "to run the OR-Tools formulation without Gurobi."
                )
            if "hybrid" in self.verbosity:
                print("Fitting the Decision Tree using BendersOCT...")

            myestimator = BendersOCT(
                X, y, features, self.tree, self.lambdaa, self.eta,
                bb_prediction, time_limit, self.n_threads
            )
            model_info = myestimator.create_master_problem().solve()

            self.num_unique_subproblems = myestimator.num_unique_subproblems
            self.subproblem_group_sizes = dict(myestimator.group_size)
            self.datapoint_to_subproblem_group = dict(
                myestimator.datapoint_to_group
            )

        elif self.estimator in {'CPSAT', 'CPSATOCT', 'CP-SAT'}:
            if "hybrid" in self.verbosity:
                print("Fitting the Decision Tree using OR-Tools CP-SAT...")

            myestimator = CPSATOCT(
                X, y, features, self.tree, self.lambdaa, self.eta,
                self.min_transp, bb_prediction, time_limit, self.n_threads
            )
            model_info = myestimator.create_primal_problem().solve()

            self.num_unique_patterns = myestimator.num_unique_patterns
            self.pattern_group_sizes = dict(myestimator.pattern_size)
            self.datapoint_to_pattern_group = dict(
                myestimator.datapoint_to_pattern
            )
            self.cpsat_solver = myestimator.solver
            self.cpsat_model = myestimator.model

        else:
            raise ValueError(
                "Unknown estimator. Choose 'FlowOCT', 'BendersOCT', "
                "or 'CPSAT'."
            )

        print(model_info, flush=True)


        #return prediction and accuracy, pred_types
        ##########################################################
        # Preparing the output after model is fitted
        ##########################################################
        self.status = model_info['Status']
        self.status_name = model_info.get('StatusName')
        self.optgap = model_info['MIPGap']
        self.callback_history = model_info["callback_history"]
        self.model_info = model_info
        self.training_estimator = myestimator

        if self.estimator in {'CPSAT', 'CPSATOCT', 'CP-SAT'}:
            if model_info['SolCount'] == 0:
                raise RuntimeError(
                    "CP-SAT did not find a feasible solution within the "
                    "specified time limit."
                )
            self.b = dict(myestimator.b)
            self.beta = dict(myestimator.beta)
            self.p = dict(myestimator.p)
        else:
            self.b = myestimator.model.getAttr("X", myestimator.b)
            self.beta = myestimator.model.getAttr("X", myestimator.beta)
            self.p = myestimator.model.getAttr("X", myestimator.p)

        self.is_fitted = True
        self.labels = np.append(np.unique(y), 'bb')
        self.features = list(features)
        self.black_box_train_acc = bb_acc

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
