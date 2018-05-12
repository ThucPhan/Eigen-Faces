function d = distance_mx(a,b)
%    a - (DxM) matrix 
%    b - (DxN) matrix
%    d - (MxN) Euclidean distances between vectors in a and b

aa = sum(a.*a,1); %double each element in a then sum matrix columns - (1xM)
bb = sum(b.*b,1); %double each element in b then sum matrix columns - (1xN)

M = size(aa,2);
N = size(bb,2);

d = sqrt( abs (aa(ones(N,1),:)' + bb(ones(M,1),:) - 2*a'*b) ); 
