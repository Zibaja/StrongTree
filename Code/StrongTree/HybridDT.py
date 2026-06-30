import numpy as np
from Tree import Tree
import gurobipy as gp
from gurobipy import GRB
from Hybrid_utils import *

import networkx as nx
import matplotlib.pyplot as plt


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
        # 2) Fit the interpretable part of the model
        if "hybrid" in self.verbosity:
            print("Fitting the Decision Tree...")

        flowoct = FlowOCT(X,y, features, self.tree , self.lambdaa , self.eta, self.min_transp,bb_prediction, time_limit)
        model_info = flowoct.create_primal_problem().solve()
        
        print(model_info, flush=True)

        #return prediction and accuracy, pred_types
        ##########################################################
        # Preparing the output after model is fitted
        ##########################################################
        self.status = model_info['Status']
        self.optgap = model_info['MIPGap']
        self.callback_history = model_info["callback_history"]
        self.b = flowoct.model.getAttr("X", flowoct.b) # remember primal_problem.b, is a tuple dict from gurobi
        self.beta = flowoct.model.getAttr("X", flowoct.beta) #also getAttr outputs tupledict
        self.p = flowoct.model.getAttr("X", flowoct.p)
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


    def predict(self, X):
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