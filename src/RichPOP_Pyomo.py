from pyomo.environ import ConcreteModel, RangeSet, Var, NonNegativeReals, Objective, minimize
from pyomo.environ import Expression, Constraint, Binary, SolverFactory, value, Param
import numpy as np
import csv


''' 01. Read problem inputs '''

# Read asset mean returns, asset stdevs, and correlation matrix
instance = '1'  # select instance between 1 and 5 for the port1 to port5 txt files
txt_file_path = 'inputs/port' + instance + '.txt'
num_assets = 0
avg_returns = []
st_devs = []
corr_matrix = []
with open(txt_file_path, 'r') as file:
    lines = file.readlines()
    num_assets = int(lines[0])  # extract the number of assets
    for i in range(1, num_assets + 1):  # extract avg return and standard deviation for each asset i
        values = lines[i].split()  # split by space
        asset_avg_return = float(values[0])
        asset_st_dev = float(values[1])
        avg_returns.append(asset_avg_return)
        st_devs.append(asset_st_dev)
    for line in lines[num_assets + 1:]:  # extract correlation between all pairs of assets
        values = line.split()
        i = int(values[0])
        j = int(values[1])
        correlation = float(values[2])
        corr_matrix.append((i, j, correlation))  # the first element in each row is the asset index

# Compute the covariance matrix from asset std devs and correlation matrix
cov_matrix = np.zeros((num_assets, num_assets))  # initialize a zero matrix for covariance
for i, j, corr in corr_matrix:  # careful! i and j are assets that start in 1, not in 0
    cov = st_devs[i - 1] * st_devs[j - 1] * corr
    cov_matrix[i - 1][j - 1] = cov
    cov_matrix[j - 1][i - 1] = cov  # covariance matrix is symmetric

# Read return thresholds to build the efficient frontier
txt_file_path = 'inputs/portef' + instance + '.txt'
return_thresholds = []
with open(txt_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines: # extract return thresholds (first element in each line)
        values = line.split()    
        return_threshold = float(values[0])
        return_thresholds.append(return_threshold)


''' 02. Define problem parameters '''

n_min = 1
n_max = num_assets
min_xi = 0.0
max_xi = 1.0 # maximum investment in any asset i
big_M = 1e20 # very large number
des_threshold = 0.0 # desirability threshold
fea_threshold = 0.0 # feasibility threshold
sus_threshold = 0.0 # sustainability threshold
np.random.seed(42)  # set the seed to a specific value for reproducibility
a = 7  # each asset level is random between a and b
b = 10 # b > a
des_levels = a + (b - a) * np.random.rand(num_assets) 
fea_levels = a + (b - a) * np.random.rand(num_assets)
sus_levels = a + (b - a) * np.random.rand(num_assets)


''' 03. Generate the Pyomo model '''

model = ConcreteModel() # define the Pyomo model
model.I = RangeSet(1, num_assets) # define set of indexes
model.x = Var(model.I, domain=NonNegativeReals) # define variables


''' 04. Define the objective function (min risk) and how to compute the associated portfolio return '''

def portf_risk_rule(model):
    p_risk = sum(sum(cov_matrix[i-1][j-1] * model.x[i] * model.x[j] for j in model.I) for i in model.I)
    return p_risk
model.portf_risk = Objective(rule=portf_risk_rule, sense=minimize)
model.portf_return = Expression(expr=sum(avg_returns[i-1] * model.x[i] for i in model.I))
model.portf_des = Expression(expr=sum(des_levels[i-1] * model.x[i] for i in model.I))
model.portf_fea = Expression(expr=sum(fea_levels[i-1] * model.x[i] for i in model.I))
model.portf_sus = Expression(expr=sum(sus_levels[i-1] * model.x[i] for i in model.I))


''' 05. Add constraints '''

# The sum of all asset weights cannot exceed 100% of available budget
model.sum_weights_cons = Constraint(expr = sum(model.x[i] for i in model.I) <= 1.0)
# The portfolio return cannot be lower than the given threshold
model.return_threshold = Param(initialize=0.0, mutable=True)  # initialize the return threshold as a parameter
model.return_cons = Constraint(expr = model.portf_return >= model.return_threshold)
# Each asset weigth has to be positive or zero
model.low_weight_cons = Constraint(model.I, rule=lambda model, i: model.x[i] >= 0)
# The portfolio desirability level cannot be lower than the corresponding threshold
model.des_cons = Constraint(expr = model.portf_des >= des_threshold)
# The portfolio feasibility level cannot be lower than the corresponding threshold
model.fea_cons = Constraint(expr = model.portf_fea >= fea_threshold)
# The portfolio sustainability level cannot be lower than the corresponding threshold
model.sus_cons = Constraint(expr = model.portf_sus >= sus_threshold)
# The number of assets in portfolio has to be between n_min and n_max
model.is_asset_selected = Var(model.I, within=Binary) # define a binary variable
for i in model.I: # initialize the binary variable
    model.is_asset_selected[i] = 0   
def is_asset_selected_rule_1(model, i): # if x[i] > 0 then is_asset_selected[i] == 1
    return model.x[i] <= model.is_asset_selected[i]
model.is_asset_selected_cons_1 = Constraint(model.I, rule=is_asset_selected_rule_1)
def is_asset_selected_rule_2(model, i): # if x[i] == 0 then is_asset_selected[i] == 0
    return model.is_asset_selected[i] <= model.x[i] * big_M
model.is_asset_selected_cons_2 = Constraint(model.I, rule=is_asset_selected_rule_2)
def count_selected_assets_rule(model):
    return sum(model.is_asset_selected[i] for i in model.I)
model.num_selected_assets = Expression(rule=count_selected_assets_rule)
model.portf_size_lb_cons = Constraint(expr = model.num_selected_assets >= n_min)
model.portf_size_ub_cons = Constraint(expr = model.num_selected_assets <= n_max)
# Each asset weight has to be lower than or equal to the maximum investment allowed per asset
model.up_weight_cons = Constraint(model.I, rule=lambda model, i: model.x[i] <= max_xi)
# If an asset is selected, its weight must be higher than or equal to a minimum inviestment
def selected_asset_min_invest_rule(model, i):
    return model.x[i] >= min_xi * model.is_asset_selected[i]
model.selected_asset_min_invest_cons = Constraint(model.I, rule=selected_asset_min_invest_rule)


''' 06. Choose a solver engine '''

solver = SolverFactory('gurobi') # possible solvers: GLPK, CBC, CPLEX, Gurobi, etc.


''' 07. Solve the optimization problem for each return threshold '''

print('Solving the POP for different return thresholds...')
results = []
k = 0
for threshold in return_thresholds:
    k = k + 1
    print('Solving POP ' + str(k) + ' of ' + str(len(return_thresholds)))
    for i in model.I: # reset all is_asset_selected to 0
        model.is_asset_selected[i] = 0
    model.return_threshold = threshold # update the return threshold parameter
    solver.solve(model) # solve the optimization problem
    opt_weights = [round(value(model.x[i]), 4) for i in model.I] # extract opt sol (asset weights)
    risk = round(value(model.portf_risk()), 6) # extract minimized risk value
    exp_return = round(value(model.portf_return), 6)
    des = round(value(model.portf_des), 1)
    fea = round(value(model.portf_fea), 1)
    sus = round(value(model.portf_sus), 1)
    n_selected = value(model.num_selected_assets)
    results.append({'Return': exp_return, 'Risk': risk, 'Desirab': des, 'Feasib': fea, 
                    'Sustainab': sus, 'N selected': n_selected, 'Weights': opt_weights})


''' 08. Export results to a CSV file '''

csv_file_path = 'outputs/port' + instance + '_results.csv'
with open(csv_file_path, mode='w', newline='') as file:
    fieldnames = ['Return', 'Risk', 'Desirab', 'Feasib', 'Sustainab', 'N selected', 'Weights']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)
print('Results have been exported to CSV file.')
