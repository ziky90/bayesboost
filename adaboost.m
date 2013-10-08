function [mu, sigma, p, alpha, classes] = adaboost(data, T)
    %implementation of the adaboost algorithm
    %data: imput data
    %T: number of classifiers
    
    %matrices and vectors initialization
    M = size(data,1);
    mu = zeros(T, 2, 2);
    sigma = zeros(T, 2, 2);
    p = zeros(T, 2);
    alpha = zeros(T, 1);
    classes = [0;1];
    e = zeros(T, 1);
    w = ones(M,1) * 1/M;
    
    for t=1:T
        
        [mu(t,:,:) sigma(t,:,:)] = bayes_weight(data, w);
        p(t,:) = prior(data, w);
    
        g = discriminant(data(:,1:2), squeeze(mu(t,:,:)), squeeze(sigma(t,:,:)), squeeze(p(t,:)));
        [dummy class] = max(g, [], 2);
        class = class-1;
        
        e(t) = 1 - sum(w.* (class == data(:,end)));      
        
        alpha(t) = 1/2 * log((1-e(t)) / e(t));    
    
        %new wegihts computation
        for i=1:length(w)
            if(class(i) == data(i,end))
                w(i) = w(i) * exp(-alpha(t));
            else
                w(i) = w(i) * exp(alpha(t));
            end
        end
        
        %weights normalization
        w = w / sum(w);
     end
    