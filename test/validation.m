data = rand(50, 5);
% data = load('/home/adriano/workspace/subspace-learning/data/mammals1k.data');
X = data';
A = X;
l = 0.1;

addpath('/home/adriano/workspace/subspace-learning/lrr');
[Z, E] = solve_lrr(X, A, l, 0, 0, true);

save('/tmp/data', 'data', '-ascii');

system('python /home/adriano/workspace/lrr/test/validation.py');

ZZ = load('/tmp/Z');
EE = load('/tmp/E');

dif_Z = max(max(abs(Z - ZZ)));
dif_E = max(max(abs(E - EE)));
disp([dif_Z, dif_E])
