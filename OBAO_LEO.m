function [gbest_fitness_list,gbest,gbest_objective_list,gbest_constraint_list] = OBAO_LEO(fun,max_iter,n,dim,params)

% Parameters
alpha = 0.1;
delta = 0.1;

%% Initialization
[~,~,lb,ub] = fun(params);

pop = zeros(n,dim);
fitness_list = zeros(1,n);

% Statistical results
gbest_fitness_list = zeros(1,max_iter);
gbest_objective_list = zeros(1,max_iter);
gbest_constraint_list = zeros(max_iter,numel(params.constraint_names));

gbest_fitness_value = inf;
gbest = [];
gbest_objective_value = inf;
gbest_constaint_value = [];

for i = 1:n
    pos = unifrnd(lb,ub,1,dim);
    pop(i,:) = pos;
    [fitness_list(i),data] = fun(params,pos);
    temp_objective_value = sum(params.weight.*data.objective_values);
    if (fitness_list(i) < gbest_fitness_value) && (isreal(fitness_list(i)) == 1)
        gbest_fitness_value = fitness_list(i);
        gbest = pop(i,:);
        gbest_constaint_value = data.constraint_values;
        if temp_objective_value < gbest_objective_value
            gbest_objective_value = temp_objective_value;
        end
    end
end

%% Main loop
for it = 1:max_iter
    G2 = 2*rand()-1;
    G1 = 2*(1-(it/max_iter));
    D1 = 1:dim;
    U = 0.00565;  % .0265
    r1 = 10;
    r = r1+U*D1;
    omega = 0.005;
    phi1 = 3*pi/2;
    phi = -omega*D1+phi1;
    x = r.*sin(phi);
    y = r.*cos(phi);
    QF = it^((2*rand()-1)/(1-max_iter)^2);
    
    for i = 1:size(pop,1)
        if it <= (2/3)*max_iter
            if rand < 0.5
                X = gbest*(1-it/max_iter)+(mean(pop(i,:))-gbest)*rand();
            else
                X = gbest.*Levy(dim)+pop((floor(n*rand()+1)),:)+(y-x)*rand;
            end
            new_X = OBL(X,lb,ub,1);
        else
            if rand < 0.5
                X = (gbest-mean(pop))*alpha-rand+((ub-lb)*rand+lb)*delta;
            else
                X = QF*gbest-(G2*pop(i,:)*rand)-G1.*Levy(dim)+rand*G2;
            end
            new_X = OBL(X,lb,ub,2);
        end
        X = space_bound(X,lb,ub);
        new_X = space_bound(new_X,lb,ub);
        if fun(params,new_X) < fun(params,X)
            X = new_X;
        end
        [X_fitness,data] = fun(params,X);
        if X_fitness < fitness_list(i)
            pop(i,:) = X;
            fitness_list(i) = X_fitness;
        end

        % Update the global best individual
        temp_objective_value = sum(params.weight.*data.objective_values);
        if (fitness_list(i) < gbest_fitness_value) && (isreal(fitness_list(i)) == 1)
            gbest_fitness_value = fitness_list(i);
            gbest = pop(i,:);
            gbest_constaint_value = data.constraint_values;
            if temp_objective_value < gbest_objective_value
                gbest_objective_value = temp_objective_value;
            end
        end
    end
    
    gbest_fitness_list(it) = gbest_fitness_value;
    gbest_objective_list(it) = gbest_objective_value;
    gbest_constraint_list(it,:) = gbest_constaint_value;
    
    new_pop_prop = 0.1;
    new_pop = leo(pop,fitness_list,it,max_iter,new_pop_prop);
    [~,sorted_ind] = sort(fitness_list,'descend');
    for l = 1:ceil(new_pop_prop*n)
        pop(sorted_ind(l),:) = space_bound(new_pop(l,:),lb,ub);
        fitness_list(sorted_ind(l)) = fun(params,pop(sorted_ind(l),:));
    end
end
end

function o = Levy(d)
beta = 1.5;
sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u = randn(1,d)*sigma;
v = randn(1,d);
step = u./abs(v).^(1/beta);
o = step;
end

function X = space_bound(X, lb, ub)
% Solutions that go out of the search space are reinitialized
tp = X > ub;
tm = X < lb;
dim = numel(X);
X = X.*~(tp+tm)+(lb+(ub-lb).*rand(1,dim)).*(tp|tm);
end

