function [d,eigMod,Hmods,Hmod,exitInfo] = ktSolverSparse(K,rhs,n,m,minPosEig,Hmod)
%        [d,eigMod,Hmods,Hmod,exitInfo] = ktSolverSparse(K,rhs,n,m,minPosEig,Hmod)
%
% ktSolverSparse solves the (n+m) by (n+m) transformed primal-dual KKT system
% using the Matlab routine LDL.
%
%  exitInfo =  0 the KKT matrix was not modified.
%              1 the KKT matrix is singular.
%              2 the KKT matrix was modified.
%             -1 the KKT inertia could not be modifed.

  kL       = 1/3;       kU   = 8;
  Hmin     = 10^(-20);  Hmax = 10^(40);
  Hmod0    = 10^(-4);
  Hmod0    = 10^(-1);

  exitInfo = 0;
  Hmods    = 0;
  eigMod   = 0;

  [L,D]    = ldl(K);
  [eigs]   = eig(D);
  maxEigK  = max(abs(eigs));    minEigK = min(abs(eigs));
  condD    = maxEigK/minEigK;

  numPos   = length( find(eigs> minPosEig*maxEigK) );
  numNeg   = length( find(eigs<-minPosEig*maxEigK) );
  numSing  = n - numPos + m - numNeg;
% K = sparse(K);   % caused inexplicable fatal exit on STRTCHDVNE
  if  numPos == n  &&  numNeg == m
    exitInfo = 0;
  else
    if numSing > 0
      exitInfo = 1;
    else
      exitInfo = 2;
    end

    if Hmod == 0
      Hmod = Hmod0;
    else
      Hmod = max(Hmin,kL*Hmod);
    end
    K0(1:n,1:n) =  K(1:n,1:n);
    normH       =  norm(K0(1:n,1:n),1);
    e           =  ones(n,1);
    convexified =  0;

    while  ~convexified
      Hmods      = Hmods + 1;
      eigMod     = Hmod;
      K(1:n,1:n) = K0(1:n,1:n) + diag(Hmod*e);
      [L,D,P]    = ldl(K);
      [eigs]     = eig(D);
      maxEigK    = max(abs(eigs));    minEigK = min(abs(eigs));
      condD      = maxEigK/minEigK;

      numPos     = length( find(eigs> 0) );
      numNeg     = length( find(eigs< 0) );
      numSing    = n - numPos + m - numNeg;
      if  numPos == n  &&  numNeg == m
        convexified = 1;
      else
        Hmod     = kU*Hmod;
        if Hmod > Hmax
          break;
        end
      end
    end

    if  ~convexified
      exitInfo = -1;
    end
  end

  if condD > eps^(-2/3)
    K          = sparse(K);
    [S,K,itns] = scaleSymSparse(K);
    d          = (K\(rhs.*S)).*S;
  else
    d          = K\rhs;
  end
end
