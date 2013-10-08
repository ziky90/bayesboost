function c = adaboost_discriminant(data, mu, sigma, p, alpha, classes, T)
    %particular class computing based on the given set of hypotesis
    %data = training data without labels
    %mu = mu parameters of all the hypotesis
    %sigma = sigma parameters of all the hypotesis
    %p = prior probablities for all the weak clasifiers
    %classes = set of all the classes used
    %T = number of weak classifiers used

    c = zeros(size(data, 1), 1);
    
    c1 = zeros(size(data, 1), 1);
    c2 = zeros(size(data, 1), 1);
    for t=1:T
        g = discriminant(data(:,1:2), squeeze(mu(t,:,:)), squeeze(sigma(t,:,:)), squeeze(p(t,:)));
        [dummy class] = max(g, [], 2);
        class = class-1;
        
        c1 = c1 + (alpha(t) * (class == 0));
        c2 = c2 + (alpha(t) * (class == 1));
    end
    
    for i=1:size(data, 1)
        if(c1(i) > c2(i))
            c(i) = 0;
        else
            c(i) = 1;
        end
    end
    
    