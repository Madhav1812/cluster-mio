using LinearAlgebra, Random, Gurobi, JuMP, Statistics, ScikitLearn, DecisionTree, MLDataUtils, DataFrames, CategoricalArrays

#We can use any of these metrics for comparing the performance of the algorithms
#compute_mse for population level estimation
function compute_mse(X, y, beta)
    n,p = size(X)
    return sum((X*beta .- y).^2)/n
end

#compute_r2 to compare population level regressional performance
function compute_r2(X, y, beta)
    SSres = sum( (y .- X*beta).^2 )
    SStot = sum( (y .- Statistics.mean(y)).^2 )
    return 1-SSres/SStot
end

#compute_mse_oct for cluster-aware prediction
function compute_mse_oct(X, y, beta, prob, Z, rand)
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
    return sum((rand_off.+X*beta .- y).^2)/n
end  

#compute_r2_oct for cluster-aware regressional performance
function compute_r2_oct(X, y, beta, prob, Z, rand)
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
    SSres = sum( (y .- X*beta.-rand_off).^2 )
    SStot = sum( (y .- Statistics.mean(y)).^2 )
    return 1-SSres/SStot
end  

#Simulation functions
##We have a function to construct Gaussian and sparse random effects, allowing the analysis given in the paper
#sigma_clust is a measure of dispersion among the cluster effects
function rand_eff_gaussian(k,q,n_i,sigma_clust)
    n = sum(n_i)
    pert = randn(q,k)
    gamma = sqrt(sigma_clust)*pert
    return gamma
end

#sparse is a notion of sparsity among the cluster effects
function rand_eff_sparse(k,q,n_i,sigma_clust,sparse)
    n = sum(n_i)
    #This framework is flexible for simulation, in this case, we assumed a symmetric distribution among the non-zero values
    pert = append!(2 .*bitrand(Int(ceil(k*sparse))).-1,zeros(Int(k-ceil(k*sparse))))
    for i in 2:q
        pert = hcat(pert,append!(2 .*bitrand(Int(ceil(k*sparse))).-1,zeros(Int(k-ceil(k*sparse)))))
    end
    gamma = sigma_clust*pert
    return gamma
end

##Function data_gen_clust: Main data generating pipeline, using confounding assumption
#sigma_eps is a notion of variance of the background noise
function data_gen_clust(beta,k,p,q,n_i,sigma_eps,gamma)
    n = sum(n_i)
    if q==1
        Z = ones(n)
    else
        Z = hcat(ones(n),randn(n,q-1))
    end 
    A = zeros((n,k))
    ct = 1
    rt = 1
    for i in n_i 
       for j in 1:i
            A[rt,ct]=1
            rt = rt+1
       end
       ct = ct+1 
    end  
    X = ones(n)
    for var in 1:p
       X = hcat(X,sum(randn(n,q)*gamma'.*A, dims=2) +randn(n))
    end
    X = X[:,2:(p+1)]
    eps = sqrt(sigma_eps)*randn(n) 
    labels = Array{Int}(undef,n)
    for i in 1:n
        labels[i] = findmax(A[i,:])[2]
    end
    y = zeros(n)
    i = 1
    for l in labels 
        if q == 1
           y[i]=X[i,:]'*beta+first(Z[i,:].*gamma[l])+eps[i]
        else 
           y[i]=X[i,:]'*beta+Z[i,:]'*gamma[:,l]+eps[i]
        end   
        i = i+1
    end   
    return y,X,Z,A,labels
end

#We run the simulation on the MIO algorithm, but any algorithm could be used here
include("clust_mio.jl")

#This is an example simulation, the parameters can be altered accordingly
iter_max = 100
k,p,q = 4,10,1
μ = 1
n_i = repeat([50],inner = k)
beta = rand([-1,0,1], p)
sigma_clust = 10
sigma_eps = 0.5
sparse = [0.1,0.2,0.25,1/3,0.4,0.5,0.6,2/3,0.75,0.8,0.9,1]
true_sparse = 0.5
res_names = [:beta,:gamma,:sparse_mio,:mse_cemio]
results = DataFrame(zeros(length(sparse),4),:auto)
rename!(results,res_names)
res_store = zeros(4)
for iter in 1:iter_max
        gamma = rand_eff_sparse(k,q,n_i,sigma_clust,true_sparse)
        y,X,Z,A,labels = data_gen_clust(beta,k,p,q,n_i,sigma_eps,gamma)
        y_val,X_val,Z_val,A_val,lab_val = data_gen_clust(beta,k,p,q,n_i,sigma_eps,gamma) 
        y_test,X_test,Z_test,_,_ = data_gen_clust(beta,k,p,q,n_i,sigma_eps,gamma)
        beta_mio, rand_mio = cemio(X,y,Z,A,labels,X_val,y_val,Z_val,A_val,lab_val,X_test,Z_test,sparse,p,k,μ)
        res_store = res_store +[sum((beta-beta_mio).^2),sum((rand_mio-gamma).^2),mean(rand_mio.!=0),compute_mse_oct(X_test,y_test,beta_mio,prob_test,Z_test,rand_mio)]
end
results = res_store./iter_max
