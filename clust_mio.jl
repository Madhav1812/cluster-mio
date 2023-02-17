using LinearAlgebra, Random, Gurobi, JuMP, Statistics, ScikitLearn, DecisionTree, MLDataUtils, CategoricalArrays

##Function compute_mse_off: Function for calculating MSE for cluster based algorithms, will be used in validation
##Inputs for compute_mse_off:
#X - fixed design matrix
#y - outcomes
#beta - vector of fixed regression parameters
#label - cluster assignments
#Z - auxiliary design matrix
#rand - matrix of cluster based effects

function compute_mse_off(X,y,beta,label,Z,rand)
    n,p = size(X)
    q = 1
    if length(size(Z))>1
        q = size(Z)[2]
    end
    rand_off = zeros(n)
    for i in 1:n
        j = label[i]
        if q==1
            rand_off[i] = Z[i]*rand[j]
        else
            rand_off[i] = sum(Z[i,:].*rand[j,:])
        end
    end    
    return sum((rand_off.+X*beta.-y).^2)/n
end

##Function pred_mio: Predicts outcome using cluster effects and assignment probabilities
##Inputs for compute_mse_off:
#X - fixed design matrix
#beta - vector of fixed regression parameters
#prob - cluster assignment probabilities
#Z - auxiliary design matrix
#rand - matrix of cluster based effects

function pred_mio(X,beta,prob,Z,rand)
    n,p = size(X)
    q = 1
    if length(size(Z))>1
        q = size(Z)[2]
    end
    rand_off = zeros(n)
    for i in 1:n
        prob_vec = [j for j in values(prob[i,:])]
        if q==1
            rand_off[i] = Z[i]*(rand'*prob_vec)
        else
            rand_off[i] = sum(Z[i,:].*(rand'*prob_vec))
        end
    end    
    return(rand_off.+X*beta)
end   

##Function inner_opt: Construct inner problem on the simplex points
##Inputs for inner_opt:
#X - fixed design matrix
#Y - outcomes
#p - number of fixed regression parameters
#s - vector of selected cluster effects
#μ - ridge hyperparameter 

function inner_opt(X,Y,p,s,μ)
    ind = findall(s.>0.5)
    #Take chosen coordinates
    n = length(Y)
    tot = 2*n
    Xs = X[:,ind]
    #Using derivative formula
    α = Y-Xs*(inv(μ*I+Xs'*Xs)*(Xs'*Y))
    obj = dot(Y,α)/tot
    temp = X'*α
    grad = -(1/μ).*temp.^2 ./tot 
    #Return objective value and gradient
  return obj, grad
end

##Function mixed_effects_regression: Implement MIO algorithm for recovering fixed and cluster regression coefficients
##Inputs for mixed_effects_regression:
#X - fixed design matrix
#Y - outcomes
#p_fix - number of fixed regression parameters
#K - number of clusters
#λ - vector of sparsity constraints for each higher order cluster effect
#μ - ridge hyperparameter
#s0 - initialization of selected cluster effects (set to null by default)

function mixed_effects_regression(X, Y, p_fix, K, λ, μ, s0=[]; solver_output=0)
    n,p = size(X)
    var_length = length(λ)
    m = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m, "LazyConstraints", 1)
    set_optimizer_attribute(m, "OutputFlag", solver_output) 
    
    @variable(m, s[j = 1:p], Bin)
    @variable(m, t >= 0)
    
    #Constraint on selected indices
    @constraint(m, [l = 1:var_length], sum(s[(p_fix+(l-1)*K+1):(p_fix+l*K)]) <= λ[l])
    @constraint(m, [j = 1:p_fix], s[j]==1)
    
    #Arbitrary initialization
    if length(s0) == 0
        s0 = zeros(p)
        s0[1:p_fix] .= 1
        for l in 1:var_length
            s0[(p_fix+(l-1)*K+1):(p_fix+(l-1)*K+λ[l])].=1
        end
    end
    obj0, grad0 = inner_opt(X,Y,p_fix,s0,μ)
    
    #Constraint for outer approximation
    @constraint(m, t >= obj0 + dot(grad0, s - s0))
    
    #Objective
    @objective(m, Min, t)
    
    #In order to implement the outer approximation, we use the following function formulation
    cb_calls = Cint[]
    s_hist = []
    
    function outer_approximation(cb_data, cb_where::Cint)
        push!(cb_calls, cb_where)
        if cb_where!=GRB_CB_MIPSOL && cb_where!=GRB_CB_MIPNODE
            return
        end
        if cb_where==GRB_CB_MIPNODE
            resultP = Ref{Cint}()
            GRBcbget(cb_data, cb_where, GRB_CB_MIPNODE_STATUS, resultP)
            if resultP[]!=GRB_OPTIMAL
                return
            end
        end
        
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        s_val = callback_value.(cb_data, s)
        t_val = callback_value(cb_data, t)
        append!(s_hist, [s_val])
        obj, grad_s = inner_opt(X,Y,p_fix,s_val,μ)
        offset = sum(grad_s.*s_val)
        if t_val<obj
            con = @build_constraint(t>=obj+sum(grad_s[j]*s[j] for j=1:p)-offset)
            MOI.submit(m, MOI.LazyConstraint(cb_data),con)
        end
    end

    MOI.set(m, Gurobi.CallbackFunction(), outer_approximation)

    optimize!(m)
    s_opt = JuMP.value.(s)
    s_nonzeros = findall(x->x>0.5,s_opt)
    beta = zeros(p)
    X_s = X[:,s_nonzeros]
    #Construct the non-zero beta values
    beta[s_nonzeros] = (1/μ)*X_s'*(Y-X_s*((μ*I+X_s'*X_s)\(X_s'*Y)))
    
    return(value.(beta))
end

##Function cemio: Implement tuned algorithm on training, validation and test datasets
##Inputs for cemio:
#X - fixed design matrix (training)
#y - outcomes (training)
#Z - auxiliary design matrix (training)
#A - assigment matrix (training)
#labels - cluster assignment (training)
#X_val - fixed design matrix (validation)
#y_val - outcomes (validation)
#Z_val - auxiliary design matrix (validation)
#A_val - assigment matrix (validation)
#lab_val - cluster assignment (validation)
#X_test - fixed design matrix (testing)
#Z_test - auxiliary design matrix (testing)
#sparse - potential sparsity levels
#p_fix - number of fixed regression parameters
#k - number of clusters
#μ - ridge hyperparameter
#s0 - initialization of selected cluster effects (set to null by default)

function cemio(X,y,Z,A,labels,X_val,y_val,Z_val,A_val,lab_val,X_test,Z_test,sparse,p_fix,k,μ,s0=[])
    X_full = X
    for j in 1:q
        X_full = hcat(X_full, Z[:,j].*A)
    end  
    val_mse = zeros(length(sparse))
    for i in 1:length(sparse)
        this_sparse = sparse[i]
        λ = repeat([Int(ceil(k*this_sparse))],inner = q)
        bet_temp = mixed_effects_regression(X_full,y,p,k,λ,μ)
        rand_temp = bet_temp[p+1:p+q*k]
        rand_temp = reshape(rand_temp,k,q)
        bet_temp = bet_temp[1:p]
        val_mse[i] = compute_mse_off(X_val,y_val,bet_temp,lab_val,Z_val,rand_temp)
    end
    cv_ind = findmin(val_mse)[2]
    
    X_full_val = X_val
    for j in 1:q
        X_full_val = hcat(X_full_val, Z_val[:,j].*A_val)
    end 
    
    λ = repeat([Int(ceil(k*sparse[cv_ind]))],inner = q)
    beta_mio = mixed_effects_regression(vcat(X_full,X_full_val),vcat(y,y_val),p,k,λ,μ)
    rand_mio = beta_mio[p+1:p+q*k]
    rand_mio = reshape(rand_mio,k,q)
    beta_mio = beta_mio[1:p]
    
    grid_cluster = DecisionTreeClassifier(max_depth = Int(ceil(log2(k))))
    DecisionTree.fit!(grid_cluster,vcat(X,X_val),vcat(labels,lab_val))
        
    prob_test = DecisionTree.predict_proba(grid_cluster,X_test)
    
    y_pred = pred_mio(X_test,beta_mio,prob_test,Z_test,rand_mio)
    
    return(beta_mio, rand_mio)
end    

#Note that mixed_effects_regression is the core inferential algorithm and can be used directly on data to find cluster effects
#cemio provides the code for the predictive algorithm
