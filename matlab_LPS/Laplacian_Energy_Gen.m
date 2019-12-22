% **********************************************************
% Author: Yiqun Wang(ÍõÒÝÈº)
% https://github.com/yiqun-wang/LPS
% **********************************************************
function [E, Cf] = Laplacian_Energy_Gen(V, F, k)

shape.VERT = V';shape.TRIV = F';
[~, shape.n] = size(V);
options.symmetrize = 1;
options.normalize = 0;
type = 'conformal';
stitching = false;
K = 3*k;

[L,A] = compute_mesh_laplacian_plusA_half(shape.VERT',shape.TRIV',type,options);

if stitching
    boundary_edge = compute_boundary_all(shape.TRIV'); % for boundary
    % A=full(A);
    % L=full(L);
    W=diag(diag(L))-L;
    inner = 1:shape.n;
    inner(boundary_edge) = [];
    boundary = sort(boundary_edge);
    bs = size(boundary, 2);
    is = size(inner, 2);
    AA = zeros(size(A) + is);
    AA(1:shape.n, 1:shape.n) = A;
    AA(shape.n+1:end, shape.n+1:end) = A(inner,inner);
    LL = zeros(size(W) + is);
    LL(1:shape.n, 1:shape.n) = W;
    LL(shape.n+1:end,boundary)=W(inner,boundary);
    LL(boundary, shape.n+1:end)=W(boundary, inner);
    LL(shape.n+1:end, shape.n+1:end)=W(inner,inner);
    LL = diag(sum(LL,2)) - LL;
    A=sparse(AA);
    L=sparse(LL);
    VERT = zeros(shape.n+is, 3);
    VERT(1:shape.n, :) = shape.VERT;
    VERT(shape.n+1:end, :) = shape.VERT(inner, :);
    shape.VERT = VERT;
    shape.n = size(LL, 1);
end

[V,D] = eigs(L,A,K+1,-1);
%ascending sort and eliminate DC.
V=fliplr(V(:,1:end-1)); D=rot90(D(1:end-1,1:end-1),2);
C = V' * A* shape.VERT;

Cf = D * sqrt(sum(C.^2,2));
Cf = Cf'; 
E = 0;
end


