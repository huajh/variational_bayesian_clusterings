K = 32;  % the number of gaussian
dim = 2; % the dimesion of data
N = 1000; % the number of data
% type: cirlce,toy
option =struct('type','cirlce');
[data,mu,n] = CreateGmmSample(K,N,option);
figure;
plot(data(1,:),data(2,:),'b.');
hold on;
plot(mu(1,:),mu(2,:),'m+','LineWidth',2);
hold off;