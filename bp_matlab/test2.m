clear;clc
close all



%Inum=input('请输入输入层节点数量为：\n')
%Hnum=input('请输入隐含层节点数量为：\n')
%Onum=input('请输入输出层节点数量为：\n')

a_train=load('hdpe_T15-1.txt');
a_gen=load('hdpe_G15-1.txt');

[m,n]=size(a_train);
[m1,n1]=size(a_gen);

Inum=15;
Hnum = 30;
Onum=1;

E_max=1e-4;
Train_num=1000;
eta=0.6;
aerf=0.4;

In_train = a_train(:,1:Inum); 
[x_t,ps] = mapminmax(In_train');
x_t = x_t'; %训练集输入
d_t = a_train(:,Inum+1:n);
[d_t,ps1] = mapminmax(d_t',0,1);
d_t = d_t'; %训练集输出

x_gen = a_gen(:,1:Inum);
d_gen = a_gen(:,Inum+1:n);
x_gen = mapminmax(x_gen',ps);
x_gen = x_gen';
d_train = a_train(:,Inum+1:n);
d_train = mapminmax(d_train',ps1);
d_train = d_train';
d_gen = mapminmax(d_gen',ps1);
d_gen = d_gen';

Wih=rands(Inum,Hnum);
Who=rands(Hnum,Onum);
dw_wih=zeros(Inum,Hnum);
dw_who=zeros(Hnum,Onum);

for i=1:Train_num
    
    Hin=x_t*Wih;
    Hout=fun_s(Hin);
    Opin=Hout*Who;
    Opot=fun_s(Opin);
    
    E_p = (d_t-Opot);
    E_train=sum(0.5*E_p.^2)/m;
    E(i)=E_train;
    
    if rem(i,20)==0
        plot(E(1:i),'r--')
        legend('训练误差曲线')
        pause(0.001)
        drawnow
    end
    
    if E_train < E_max
        flag=1;
        break;
    end
    
    
    detea_ho=Opot.*(1-Opot).*E_p;
    for j=1:m
        dw_ho(:,j) = eta * detea_ho(j) * Hout(j,:)';
    end
    dw_who=sum(dw_ho,2)/m + aerf * dw_who;
    
    for j=1:m
        detea_ih(j,:)=Hout(j,:)'.*(1-Hout(j,:))'.*detea_ho(j).*Who;
    end
    
    for j=1:m
        for k=1:Hnum
            dw_ih(:,k,j)=eta * detea_ih(j,k) * x_t(j,:)';
        end
    end
    dw_wih=sum(dw_ih,3)/m+aerf*dw_wih;
    Wih=Wih +dw_wih;
    Who=Who +dw_who;
end

Hin_train=x_t*Wih;
Hout_train=fun_s(Hin_train);
Opin_train=Hout_train*Who;
Opot_train=fun_s(Opin_train);

y_t = mapminmax('reverse',Opot_train,ps1);
% y_t = Opot_train;
d_train = mapminmax('reverse',d_train,ps1);
figure('name','训练相对误差','numberTitle','off')

E_r=abs(d_train-y_t)./d_train;
plot(E_r,'r-*')

figure('name','训练输出比较图','numberTitle','off')
plot(d_train,'g')
hold on
plot(y_t,'r')
legend({'训练实际输出','训练网络输出'},'Location','SouthWest')

Hin_gen=x_gen*Wih;
Hout_gen=fun_s(Hin_gen);
Opin_gen=Hout_gen*Who;
Opot_grn=fun_s(Opin_gen);
y_gen = Opot_grn;
y_gen = mapminmax('reverse',y_gen,ps1);
d_gen = mapminmax('reverse',d_gen,ps1);

figure('name','泛化输出比较图','numberTitle','off')
plot(d_gen,'g')
hold on
plot(y_gen,'r')
legend({'泛化实际输出','泛化网络输出'},'Location','SouthWest')




function output = fun_s( input )
    output = 1./(1+exp(-input));
end