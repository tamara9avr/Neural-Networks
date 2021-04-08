clc, clear, close all
%% Ucitavanje
load dataset3.mat
ulaz = pod(:, 1:2)';
klase = pod(:,3)';

N=length(klase);

K1 = ulaz(:, klase==1);
K2 = ulaz(:, klase==2);
K3 = ulaz(:, klase==3);

figure, hold all
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'yd')

izlaz = zeros(3,N);
izlaz(1,klase==1)=1;
izlaz(2,klase==2)=1;
izlaz(3,klase==3)=1;

%% Podela podataka
ind= randperm(N);
ind_trening = ind(1:0.9*N);
ind_test=ind(0.9*N+1:N);

ulaz_trening = ulaz(:,ind_trening);
izlaz_trening = izlaz(:,ind_trening);

ulaz_test= ulaz(:,ind_test);
izlaz_test = izlaz(:,ind_test);

%% Kreiranje neuralnih mreza
%Mreza koja nedovoljno obucava
net1 = patternnet([5,3]);
%Mreza koja dobro obucava
net2 = patternnet([30,15,10]);
%Mreza koja preobucava
net3=patternnet(1);

%Parametri mreza
net1.divideFcn='';
net1.trainFcn = 'trainscg';

net1.trainParam.epochs = 1500;
net1.trainParam.goal = 1e-3;
net1.trainParam.min_grad = 1e-4;

net2.divideFcn='';
net2.trainFcn = 'trainscg';

net2.trainParam.epochs = 1500;
net2.trainParam.goal = 1e-3;
net2.trainParam.min_grad = 1e-4;

net3.divideFcn='';
net3.trainFcn = 'trainscg';

net3.trainParam.epochs = 1500;
net3.trainParam.goal = 1e-3;
net3.trainParam.min_grad = 1e-4;

%% Treniranje neuralnih mreza
[net1,tr1] = train(net1, ulaz_trening, izlaz_trening);
[net2,tr2] = train(net2, ulaz_trening, izlaz_trening);
[net3,tr3] = train(net3, ulaz_trening, izlaz_trening);

%% Performanse neurlanih mreza
pred1 = net1(ulaz_test);
figure, plotconfusion(izlaz_test,pred1, 'net1 test');
pred_tr1 = net1(ulaz_trening);
figure, plotconfusion(izlaz_trening,pred_tr1, 'net1 trening');

figure, plot(tr1.perf)

pred2 = net2(ulaz_test);
figure, plotconfusion(izlaz_test,pred2, 'net2 test');
pred_tr2 = net2(ulaz_trening);
figure, plotconfusion(izlaz_trening,pred_tr2, 'net2 trening');

figure, plot(tr2.perf)

pred3 = net3(ulaz_test);
figure, plotconfusion(izlaz_test,pred3, 'net3 test');
pred_tr3 = net3(ulaz_trening);
figure, plotconfusion(izlaz_trening,pred_tr3, 'net3 trening');

figure, plot(tr3.perf)

%% Precision, Recall
[e, cm] = confusion(izlaz_test, pred1);
cm=cm';
Precision = cm(1,1) / sum(cm(1,:))
Recall = cm(1,1)/sum(cm(:,1))

%% Granice odlucivanja
Ntest = 500;
x1Test = repmat(linspace(-1,1,Ntest),1,Ntest);  
x2Test = repelem(linspace(-1, 1, Ntest),Ntest); 

ulazGO = [x1Test;x2Test];

predGO1 = net1(ulazGO);
predGO2 = net2(ulazGO);
predGO3 = net3(ulazGO);

[~,izlazGO1]=max(predGO1);
[~,izlazGO2]=max(predGO2);
[~,izlazGO3]=max(predGO3);

K1tets1=ulazGO(:,izlazGO1==1);
K2test1=ulazGO(:,izlazGO1==2);
K3test1=ulazGO(:,izlazGO1==3);

K1test2=ulazGO(:,izlazGO2==1);
K2test2=ulazGO(:,izlazGO2==2);
K3test2=ulazGO(:,izlazGO2==3);

K1test3=ulazGO(:,izlazGO3==1);
K2test3=ulazGO(:,izlazGO3==2);
K3test3=ulazGO(:,izlazGO3==3);

%% Plot
figure, hold all
plot(K1tets1(1, :), K1tets1(2, :), '.')
plot(K2test1(1, :), K2test1(2, :), '.')
plot(K3test1(1, :), K3test1(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'y+')

figure, hold all
plot(K1test2(1, :), K1test2(2, :), '.')
plot(K2test2(1, :), K2test2(2, :), '.')
plot(K3test2(1, :), K3test2(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'y+')

figure, hold all
plot(K1test3(1, :), K1test3(2, :), '.')
plot(K2test3(1, :), K2test3(2, :), '.')
plot(K3test3(1, :), K3test3(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'y+')

