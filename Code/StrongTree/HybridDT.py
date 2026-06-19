import numpy as np
from Tree import Tree
import gurobipy as gp
from gurobipy import GRB


class FlowOCT:
    def __init__(self, X,y, features, tree, lambdaa, eta, alpha,bb_prediction, time_limit):
       

        self.data = X #X is datafarme
        self.datapoints = X.index
        self.labels = np.append(y.unique(), 'bb') # all classed k in K and bb
        self.num_datapoints = self.data.shape[0]
        '''
        cat_features is the set of all categorical features. 
        reg_features is the set of all features used for the linear regression prediction model in the leaves.  
        '''
        self.cat_features = features
        self.tree = tree
        self.lambdaa = lambdaa
        self.eta = eta
        self.alpha = alpha
        self.black_box_pred = bb_prediction 
        
        # parameters
        self.m = {}
        for i in self.datapoints:
            self.m[i] = 1


        # Decision Variables
        self.b = 0 #branching variables
        self.p = 0 #leaf variables
        self.beta = 0 #label assignment variables
        self.zeta = 0 # z[i,n,t] is the flow from node n to sink nodes
        self.z = 0 # z[i,n] is the incoming flow to node n for datapoint i
        self.d = 0 #variables for outgoing flow of deferred nodes

        # Gurobi model
        self.model = gp.Model('FlowOCT')
       
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
        bb_labels = self.black_box_pred.unique()
        bb_pred = {k:np.zeros(X.shape[0]) for k in bb_labels}
        for k in bb_labels:
            bb_pred[k][np.where(self.black_box_pred==k)]=1


        ############################### define variables
        # b[n,f] ==1 iff at node n we branch on feature f
        self.b = self.model.addVars(self.tree.Nodes, self.cat_features, vtype=GRB.BINARY, name='b')
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self.p = self.model.addVars(self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name='p')
        
        #For classification beta[n,k]=1 iff at node n we predict class k
        self.beta = self.model.addVars(self.tree.Nodes + self.tree.Leaves, self.labels, vtype=GRB.CONTINUOUS, lb=0,
                                       name='beta')
        # zeta[i,n] is the amount of flow through the edge connecting node n to sink node t_k for datapoint i
        self.zeta = self.model.addVars(self.datapoints, self.tree.Nodes + self.tree.Leaves,self.labels, vtype=GRB.CONTINUOUS, lb=0,
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
                (self.z[i, n] == self.z[i, n_left] + self.z[i, n_right] + gp.quicksum(self.zeta[i, n, k] for k in self.labels)) for i in self.datapoints)

        # z[i,n] = sum(z[i,n,k] , k) for i,n in terminal nodes
        for n in self.tree.Leaves:
            self.model.addConstrs(gp.quicksum(self.zeta[i, n, k] for k in self.labels) == self.z[i, n] for i in self.datapoints)

        #z[i,1]==1 forall i 
        self.model.addConstrs(self.z[i,1]==1 for i in self.datapoints)


        # z[i,l(n)] <= m[i] * sum(b[n,f], f if x[i,f]=0)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_left_children(n))] <= self.m[i] * gp.quicksum(
                self.b[n, f] for f in self.cat_features if self.data.at[i, f] == 0)) for n in self.tree.Nodes)

        # z[i,r(n)] <= m[i] * sum(b[n,f], f if x[i,f]=1)    forall i, n in Nodes
        for i in self.datapoints:
            self.model.addConstrs((self.z[i, int(self.tree.get_right_children(n))] <= self.m[i] * gp.quicksum(
                self.b[n, f] for f in self.cat_features if self.data.at[i, f] == 1)) for n in self.tree.Nodes)

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
            for k in self.labels:
                self.model.addConstrs(
                    self.zeta[i, n, k] <= self.beta[n, k] for i in self.datapoints)
         # zeta[i,n,k] <= pred_bb[i,k] * beta[n,'bb'] for all n, i, k for bb nodes
        for n in self.tree.Nodes + self.tree.Leaves:
            for k in self.labels:
                self.model.addConstrs(self.zeta[i,n,k]<= bb_pred[k][i] * self.beta[n,'bb'] for i in self.datapoints)
        
        # sum(beta[n,k]==p_n for all n)
        self.model.addConstrs(
            (gp.quicksum(self.beta[n, k] for k in self.labels) == self.p[n]) for n in
            self.tree.Nodes + self.tree.Leaves)
        #min_transparency constraint
        self.model.addConstr(1- ((gp.quicksum(self.d[i,n] for i in self.datapoints
                                              for n in self.tree.Nodes + self.tree.Leaves))/self.num_datapoints) >= self.alpha)

        #set of constraints related to d variables
        for n in self.tree.Nodes + self.tree.Leaves:
            self.model.addConstrs(self.d[i,n] <= gp.quicksum(self.zeta[i,n,k] for k in self.labels) for i in self.datapoints)
            self.model.addConstrs(self.d[i,n] <= self.beta[n,'bb'] for i in self.datapoints)
            self.model.addConstrs(self.d[i,n] >= gp.quicksum(
                self.zeta[i,n,k] for k in self.labels)-(1-self.beta[n,'bb']) for i in self.datapoints)
            


        # define objective function


        obj = ((1/self.num_datapoints)* gp.quicksum(
            self.zeta[i,n,str(self.y[i])] for i in self.datapoints for n in self.tree.Nodes + self.tree.Leaves )) - (self.lambdaa * (
                gp.quicksum(self.b[n,f] for n in self.tree.Nodes for f in self.cat_features))) - ((self.eta/self.num_datapoints) * (
                    gp.quicksum(self.d[i,n] for i in self.datapoints for n in self.tree.Nodes + self.tree.Leaves  )))


        self.model.setObjective(obj, GRB.MAXIMIZE)



class HybridDT():
    def __init__(self, black_box_classifier=None,depth = None, lambdaa = None, eta = None,min_transp = None , estimator = 'FlowOCT', verbosity = ['hybrid'], random_state=42, bb_pretrained=False ):
        #TODO : add all the parameters here
        self.depth = depth
        self.lambdaa = lambdaa
        self.min_transp = min_transp
        self.eta = eta
        self.bb_pretrained=bb_pretrained
        self.verbosity = verbosity
        self.estimator = estimator
        np.random.seed(random_state)

        # Creation of the black-box part of the Hybrid model
        if black_box_classifier is None:
            if self.bb_pretrained:
                raise ValueError("Parameters indicate that the black-box is pretrained but it is not provided!")
            print("Unspecified black_box_classifier parameter, using sklearn RandomForestClassifier() for black-box part of the model.")
            from sklearn.ensemble import RandomForestClassifier
            black_box_classifier = RandomForestClassifier()
        self.BlackBoxClassifier = black_box_classifier
        self.black_box_part = self.BlackBoxClassifier

        # If parameters indicate that BB is pretrained, verify it now
        if self.bb_pretrained:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.black_box_part)
        tree = Tree(depth)
        self.pre_train_tree = tree
        # Done!
        self.is_fitted = False
        if "hybrid" in self.verbosity:
            print("Hybrid model created!")

    
    def fit(self, X_train, y_train):
        #fit the bb 
        #pass the prediction by bb
        #call flowoct
        #solve and optimize as a part of training
        #return prediction and accuracy, pred_types
        pass


    def predict(self, X_test):
        pass