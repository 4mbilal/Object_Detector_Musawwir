clear all
clc
close all

LUT2 = csvread('E:\RnD\Current_Projects\Musawwir\Frameworks\SW\Dataset\Person\Caltech\code\data-INRIA\res\HSG_MR_FPPI.csv');
MR2 = LUT2(1,:)';
FPPI2 = LUT2(2,:)';
loglog(FPPI2,MR2)
exp(mean(log(MR2)))


total = 530;
frames = 288;
LUT = csvread('E:\RnD\Current_Projects\Musawwir\Frameworks\SW\Dataset\Person\Caltech\code\data-INRIA\res\HSG_Detection_Stats.csv');
% pause
FP = LUT(1,1:160);
TP = LUT(2,1:429);


FP = [FP',zeros(length(FP),1)];
TP = [TP',ones(length(TP),1)];
DT = [FP;TP];


[Y,I]=sort(DT(:,1));
DT=DT(I,:);
TP = DT(:,2);
FP = 1-TP;
% pause
TP = flipud(cumsum(flipud(TP)));
FP = flipud(cumsum(flipud(FP)));
% pause
MR = (total-TP)*1/total;
FPPI = FP/frames;
loglog(FPPI,MR)
% plot(FPPI,MR)
% semilogx(FPPI,MR)
% ax = gca;
% ax.YTick = [.05 .1:.1:.5 .64 .8];

%Same functionality but implemented in C++



