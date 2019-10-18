clear; clc;
fc=readfis('t05_fuzzy');

capitata = readtable('399835-C_capitata.csv');
fraterculus = readtable('399836-A_fraterculus.csv');
capitataSize = size(capitata);
fraterculusSize = size(fraterculus);

capitataCount = 0;
fraterculusCount = 0;
notCount  = 0;
data = [capitata.F0_Autocorrela____o, capitata.F0_FFT, capitata.F1_FFT - capitata.F0_FFT];
for i=1:capitataSize(1,1)
    res = evalfis(data(i),fc);
    if res<0.4
        fraterculusCount = fraterculusCount+1;
    else
        if res>0.6
            capitataCount = capitataCount+1;
        else
            notCount = notCount+1;
        end
    end
    
    Resultado1(i,1)=data(i,1);
    Resultado1(i,2)=data(i,2);
    Resultado1(i,3)=data(i,3);
    Resultado1(i,4)=res;
end

capitataAcerto = capitataCount/capitataSize(1,1)*100;
capitataErro   = fraterculusCount/capitataSize(1,1)*100;
capitataNot    = notCount/capitataSize(1,1)*100;


capitataCount = 0;
fraterculusCount = 0;
notCount  = 0;
data = [fraterculus.F0_Autocorrela____o, fraterculus.F0_FFT, fraterculus.F1_FFT - fraterculus.F0_FFT];
for i=1:fraterculusSize(1,1)
    res = evalfis(data(i),fc);
    if res<0.4
        fraterculusCount = fraterculusCount+1;
    else
        if res>0.6
            capitataCount = capitataCount+1;
        else
            notCount = notCount+1;
        end
    end
   
    Resultado2(i,1)=data(i,1);
    Resultado2(i,2)=data(i,2);
    Resultado2(i,3)=data(i,3);
    Resultado2(i,4)=res;
end

fraterculusAcerto = fraterculusCount/fraterculusSize(1,1)*100;
fraterculusErro   = capitataCount/fraterculusSize(1,1)*100;
fraterculusNot    = notCount/fraterculusSize(1,1)*100;