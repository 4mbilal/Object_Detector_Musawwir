% Compile using following command, nojvm and -R reduce startup time but no
% graphing function can be used.
% mcc -m -R -nojvm Matlab_SVM_Interface


clear all
close all
clc

fid = fopen('E:\RnD\Current_Projects\Musawwir\Frameworks\SW\Dataset\Person\train\features_dump.dat')
values = fread(fid,'single');
fclose(fid);
l = length(values);

examples_count = values(1);
FeatureVectorLength = values(2);
values = values(3:end);
values = reshape(values,FeatureVectorLength+1,examples_count)';
labels = values(:,end);
features = values(:,1:end-1);
size(features)

svm_model = fitcsvm(features,labels,'KernelFunction','linear','BoxConstraint',0.01);%,'Cost',c);
weights = [svm_model.Beta' , svm_model.Bias];

[eigvec,mu,eigval] = pca( features );
