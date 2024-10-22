%% Markov Chain Experiments


%%
load('transition_matrix.mat')
v0 = [1 0 0 0 0 0 0 0 0 0 0];

years = 50;
PCI = zeros(years, 11);

for j = 1:years
    PCI(j,:) = v0*M^j
end

%plot(PCI())

%% Plot Whk-Verteilung nach n Jahren
bar(100:-10:0,PCI(50,:))