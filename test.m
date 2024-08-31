I = 50;  % 遗传算法的迭代次数
N = 50;  % 种群大小
pm = 0.01;  % 变异概率
pc = 0.8;  % 交叉概率
umax = 2;  % 变量范围的上限
umin = -2;  % 变量范围的下限
L = 9;  % 染色体长度
bval = round(rand(N,2*L));  % 初始化种群，即产生一定数量的二进制数，作为染色体基因
bestv = -inf;  % 初始化最优适应度为负无穷
for ii=1:I
    for i=1:N
        y1=0;
        for j=1:1:2*L
            y1=y1+bval(i,2*L-j+1)*2^(j-1);
        end
        x=(umax-umin)*y1/(2^(2*L)-1)+umin;

        obj(i)=-x.^2-4*x+1;
        xx(i,:)=x;
    end
    func = obj;
    p = func ./ sum ( func );
    q = cumsum ( p );
    [ fmax , indmax ]= max ( func );
    if fmax >= bestv
        bestv =fmax;
        bvalxx = bval ( indmax ,:);
        optxx = xx ( indmax ,:);
    end
    Bfitl ( ii )= bestv ;
    for i =1:( N -1)
        r = rand ;
        tmp = find ( r <= q );
        newbval ( i ,:)= bval ( tmp (1),:);
    end
    newbval ( N ,:)= bvalxx ;
    bval = newbval ;
    for i =1:2:( N -1)
        cc=rand;
        if cc<pc
            point=ceil(rand*2*L-1);
            ch = bval ( i ,:);
            bval ( i , point +1:2* L )= bval ( i +1, point +1:2* L );
            bval ( i +1, point +1:2* L )= ch (1, point +1:2* L );
        end
    end
    bval ( N ,:)= bvalxx ;
    mm = rand ( N ,2* L )< pm ;
    mm(N,:)=zeros(1,2*L);
    bval(mm)=1-bval(mm);
end
plot(Bfitl);
bestv
optxx
bvalxx