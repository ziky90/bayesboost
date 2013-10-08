function [mu, sigma] = bayes(data)
    %function that calculates maximum posterior (MAP) parameters
    %data = data set with last column containing the classes

    %mu and sigma matrices initialization
    mu = ones(2,2);
    sigma = ones(2,2);
    
    %number of elements in class computation
    class1size = sum(data(:,3) == 0);
    class2size = sum(data(:,3) == 1);
    
    %mu calculations
    mu(1,1) = sum(data(1:class1size, 1)) / class1size;
    mu(1,2) = sum(data(1:class1size, 2)) / class1size;
    mu(2,1) = sum(data(class1size+1:end, 1)) / class2size;
    mu(2,2) = sum(data(class1size+1:end, 2)) / class2size;
    
    %sigma calculations
    sigma(1,1) = sqrt(sum(((data(1:class1size, 1) - mu(1,1))).^2) / class1size);
    sigma(1,2) = sqrt(sum(((data(1:class1size, 2) - mu(1,2))).^2) / class1size);
    sigma(2,1) = sqrt(sum(((data(class1size+1:end, 1) - mu(2,1))).^2) / class2size);
    sigma(2,2) = sqrt(sum(((data(class1size+1:end, 2) - mu(2,2))).^2) / class2size);