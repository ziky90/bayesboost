close all;
clear all;

%loading and showing the pictures
hand = imread('hand.ppm', 'ppm');
book = imread('book.ppm', 'ppm');
imagesc(hand);
figure;
imagesc(book);

%viewing the data
data1 = normalize_and_label(hand, 0);
data2 = normalize_and_label(book, 1);
test_data = [data1; data2];
figure;
hold on;
plot(data2(:,1), data2(:,2), '.');
plot(data1(:,1), data1(:,2), '.r');
legend('Hand holding book', 'Hand');
xlabel('green');
ylabel('red');

%2 Bayes classifier
%assignment 1
%plotting 95%-confidence interval
[mu sigma] = bayes(test_data)
theta = [0:0.01:2*pi];
x1 = 2*sigma(1,1)*cos(theta) + mu(1,1);
y1 = 2*sigma(1,2)*sin(theta) + mu(1,2);
x2 = 2*sigma(2,1)*cos(theta) + mu(2,1);
y2 = 2*sigma(2,2)*sin(theta) + mu(2,2);

plot(x1, y1, 'r');
plot(x2, y2);


%assignment 2
%calculating the discriminant function and creating the dummy classifier
[M N] = size(test_data);
p = prior(test_data);
g = discriminant(test_data(:,1:2), mu, sigma, p);
[dummy class] = max(g, [], 2);
class = class - 1;
error_test = 1.0-sum(class == test_data(:,end))/M

%drawing overlay decision boundary
ax = [0.2 0.5 0.2 0.45];
axis(ax);
x = ax(1):0.01:ax(2);
y = ax(3):0.01:ax(4);
[z1 z2] = meshgrid(x, y);
z1 = reshape(z1, size(z1,1)*size(z1,2), 1);
z2 = reshape(z2, size(z2,1)*size(z2,2), 1);
g = discriminant([z1 z2], mu, sigma, p);
gg = g(:,1) - g(:,2);
gg = reshape(gg, length(y), length(x));
[c,h] = contour(x, y, gg, [0.0 0.0]);
set(h, 'LineWidth', 3);
hold off;


%normalizing the book image
book_rg = zeros(size(book,1), size(book,2), 2);
for y=1:size(book,1)
  for x=1:size(book,2)
    s = sum(book(y,x,:));
    if (s > 0)
      book_rg(y,x,:) = [double(book(y,x,1))/s double(book(y,x,2))/s];
    end
  end
end

%classifying alll the pixels of the image and creating mask of all pixels
%that are classified to belong to other image
tmp = reshape(book_rg, size(book_rg,1)*size(book_rg,2), 2);
g = discriminant(tmp, mu, sigma, p);
gg = g(:,1) - g(:,2);
gg = reshape(gg, size(book_rg,1), size(book_rg,2));
mask = gg < 0;
mask3D(:,:,1) = mask;
mask3D(:,:,2) = mask;
mask3D(:,:,3) = mask;

%applying the mask on original immage
result_im = uint8(double(book) .* mask3D);
figure;
imagesc(result_im);


%3 Boosting
%assignment 3
w = ones(M,1) * 1/M;
[mu1 sigma1] = bayes_weight(test_data, w);


%assignment 4
%adaboost
T = 6;
[mu sigma p alpha classes] = adaboost(test_data, T)
class = adaboost_discriminant(test_data(:,1:N-1), mu, sigma, p, alpha, classes, T);
boost_error_test = 1.0-sum(class == test_data(:,end))/M


%plotting the decision boundary
figure;
hold on;
plot(data2(:,1), data2(:,2), '.');
plot(data1(:,1), data1(:,2), '.r');
legend('Hand holding book', 'Hand');
ax = [0.2 0.5 0.2 0.45];
axis(ax);
x = ax(1):0.01:ax(2);
y = ax(3):0.01:ax(4);
[z1 z2] = meshgrid(x, y);
z1 = reshape(z1, size(z1,1)*size(z1,2), 1);
z2 = reshape(z2, size(z2,1)*size(z2,2), 1);
g = adaboost_discriminant([z1 z2], mu, sigma, p, alpha, classes, T);
gg = reshape(g, length(y), length(x));
[c,h] = contour(x, y, gg, [0.5 0.5]);
set(h, 'LineWidth', 3);
hold off;


%hand removal problem visualization
tmp = reshape(book_rg, size(book_rg,1)*size(book_rg,2), 2);
g = adaboost_discriminant(tmp, mu, sigma, p, alpha, classes, T);
gg = reshape(g, size(book_rg,1), size(book_rg,2));
mask = gg > 0.5;
mask3D(:,:,1) = mask;
mask3D(:,:,2) = mask;
mask3D(:,:,3) = mask;
result_im = uint8(double(book) .* mask3D);
figure;
imagesc(result_im);
