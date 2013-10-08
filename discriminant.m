function g = discriminant(data, mu, sigma, p)
    %discriminant function that decides about the classification for the
    %given hypotesis
    %data = input data without labels
    %mu = parameter mu of the given hypotesis
    %sigma = parameter sigma of the given hypotesis
    %p = prior probablities

    g = zeros(size(data));
    for i=1:size(data,1)
        g(i,1) = log(p(1)) - sum(log(sigma(1,:))) - sum(((data(i,:) - mu(1,:)).^2)./ (2*sigma(1,:).^2));
        g(i,2) = log(p(2)) - sum(log(sigma(2,:))) - sum(((data(i,:) - mu(2,:)).^2)./ (2*sigma(2,:).^2));
    end