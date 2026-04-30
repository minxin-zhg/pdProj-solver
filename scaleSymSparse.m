function [S,A,itns] = scaleSymSparse(A,sclTol)
%        [S,A,itns] = scaleSym(A,sclTol)
%
%   Given an n by n symmetric A, scaleSym returns the scaled
%   matrix A and the row and column scales in the vector S.
%
%   A(scaled)  = diag(S))*A*diag(S)
%   Ax = b => (A(scaled)\(b.*S)).*S

  Done    =  0;
  itns    =  0;
  sclItns = 20;
  if nargin <= 3
    sclTol = 10^(-3);
  end

  [n,m]  = size(A);
  S      = ones(n,1);

  while  itns <= sclItns,
    itns = itns + 1;
    D    = zeros(1,n);

    D    = max(abs(A));
    D(find(D <= 0)) = 1;

  % Check if A has been scaled enough.
    tol = max(abs(1 - D));

    if  tol <= sclTol  ||  itns >= sclItns
      Done = 1;
    end

  % Scale A.
    D     = sqrt(1./D);
    diagD = spdiags(D',0,n,n);
    A     = diagD*A*diagD;
    S     = S.*D';

    if  Done
      break;
    end
  end
end