clc, clear all

data = readtable('Metro_dataset.csv');
inputs=data(:,1:7);
izlaz = data(:,8);
izlaz = table2array(izlaz)';

ulaz = table2array(inputs(:,2:5));

for i =["holiday","weather_main","weather_description"]
    cat = categorical(inputs.(i));
    gr = grp2idx(cat);
    ulaz = [ulaz gr];
end

ulaz = ulaz';

figure, plot(ulaz(1,:), izlaz, '.');
figure, plot(ulaz(2,:), izlaz,'.');
figure, plot(ulaz(3,:), izlaz,'.');
figure, plot(ulaz(4,:), izlaz,'.');
figure, plot(ulaz(5,:), izlaz,'.');
figure, plot(ulaz(6,:), izlaz,'.');
figure, plot(ulaz(7,:), izlaz,'.');
%%
N = length(izlaz);

ind= randperm(N);
ind_trening = ind(1:0.9*N);
ind_val=ind(0.9*N+1:N);

ulazVal = ulaz(:,ind_val);
ulazTrening = ulaz(:, ind_trening);

izlazTrening = izlaz(:,ind_trening);
izlazVal = izlaz(:,ind_val);


%% NM
bestAcc = 0;

arhitektura=[20 20];
for lr=[0.05, 0.1, 0.2, 0.3]
    for reg=[0.3, 0.4, 0.5]
        for mom=[0.5, 0.6, 0.7]
            
            net = fitnet(arhitektura);
            
            net.layers{1}.transferFcn = 'poslin';
            net.layers{2}.transferFcn = 'poslin';
            net.layers{3}.transferFcn = 'softmax';
            
            net.trainFcn = 'traingdm';
            net.divideFcn = 'divideind';
            net.divideParam.trainInd = 1 : 0.9*N;
            %net.divideParam.valInd = 0.9*N+1 : N;
            net.divideParam.testInd=[];
            
            net.trainParam.lr = lr;
            net.trainParam.mc = mom;
            net.performParam.regularization = reg;
            
            net.trainParam.epochs = 1000;
            net.trainParam.goal = 1e-4;
            net.trainParam.min_grad = 1e-5;
            net.trainParam.max_fail = 10;
            
            [net, tr] = train(net, ulazTrening, izlazTrening);
            
            predVal = net(ulazVal);
            [c,cm] = confusion(izlazVal, predVal);
            cm = cm';
            A = 1 - c;
            
            if A>bestAcc
                bestAcc=A;
                bestMom=mom;
                bestlr=lr;
                bestreg=reg;
                bestep=tr.best_epoch;
            end
        end
    end
end

%%
net = fitnet(arhitektura);

net.trainFcn = 'traingdm';
net.divideFcn = '';

net.trainParam.lr = bestlr;
net.trainParam.mc = bestMom;
net.performParam.regularization = bestreg;

net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-4;
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 10;

[net, tr] = train(net, ulazTrening, izlazTrening);

figure, plot(tr.perf);

bestlr
bestMom
bestreg

%figure, plotregression(ulazVal, net(ulazVal));

figure, hold all
plot(ulazVal(1,:), net(ulazVal), 'o');
plot(ulazVal(1,:), izlazVal, '*');

figure, hold all
plot(ulazVal(2,:), net(ulazVal), 'o');
plot(ulazVal(2,:), izlazVal, '*');

figure, hold all
plot(ulazVal(3,:), net(ulazVal), 'o');
plot(ulazVal(3,:), izlazVal, '*');

figure, hold all
plot(ulazVal(4,:), net(ulazVal), 'o');
plot(ulazVal(4,:), izlazVal, '*');

figure, hold all
plot(ulazVal(5,:), net(ulazVal), 'o');
plot(ulazVal(5,:), izlazVal, '*');

figure, hold all
plot(ulazVal(6,:), net(ulazVal), 'o');
plot(ulazVal(6,:), izlazVal, '*');

figure, hold all
plot(ulazVal(7,:), net(ulazVal), 'o');
plot(ulazVal(7,:), izlazVal, '*');