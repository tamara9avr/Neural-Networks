clc, clear, close all

%Podaci
A=2;
B=3;
f1=15;
f2=3;
N=1500;

%Funkcija
x=linspace(0,1,N);
h=A*sin(2*pi*f1*x)+B*sin(2*pi*f2*x);
std=0.2*min(A,B);
y=h+std*rand(1,N);

figure, hold all
plot(x,h,'r')
plot(x,y,'g')
xlabel('x osa')
ylabel('y osa')

ulaz = x;
izlaz = y;

%Podela podataka na trening i test skup
ind = randperm(N);
ind_trening = ind(1:0.9*N);
ind_test = ind(0.9*N+1:N);

ulaz_trening = ulaz(:,ind_trening);
izlaz_trening = izlaz(ind_trening);

ulaz_test=ulaz(:,ind_test);
izlaz_test=izlaz(ind_test);

%Kreiranje neuralne mreze
net = fitnet([10 6]);
net.divideFcn='';
net.trainFcn = 'trainlm';

net.trainParam.epochs = 3000;
net.trainParam.goal = 1e-5;
net.trainParam.min_grad = 1e-4;

net.layers{1}.transferFcn = 'logsig';

%Treniranje neuralne mreze
[net,tr]=train(net, ulaz_trening, izlaz_trening);

%Performanse
stvarni_izlaz=net(ulaz_test);

figure, plot(tr.perf)
figure, plotregression(izlaz_test, stvarni_izlaz)

figure, hold all
plot(ulaz_test, izlaz_test, 'ro')
plot(ulaz_test, stvarni_izlaz, 'g*')
xlabel('x osa')
ylabel('y osa')

figure, hold all
plot(ulaz, izlaz, 'r', 'LineWidth', 2)
plot(ulaz, net(ulaz), 'b', 'LineWidth', 2)
xlabel('x osa')
ylabel('y osa')
