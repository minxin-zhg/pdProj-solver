function [solution,outcome,Stats] = pdProj(prob,outfile,pdb_parms)
%        [solution,outcome,Stats] = pdProj(prob,outfile,pdb_parms)
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%  Projected-search primal-dual all-shifted penalty-barrier method.
%
%  pdProj computes a primal-dual solution (x,y) of the problem:
%
%                  minimize                f(x)
%
%                  subject to  ( bl )    (   x  )    ( bu )
%                              (    ) <= (      ) <= (    ),
%                              ( cl )    ( c(x) )    ( cu )
%
%  where f(x), c(x) and their first and second derivatives are defined by
%  the CUTEst Matlab tools:
%
%  The vector c(x) can be empty.
%  bl and cl can be -inf, bu and cu can be +inf.
%
%  This file contains the auxiliary functions:
%    pdbMerit       pdbqArmijoLS    pdbslackReset   pdb_print     infeasMul
%
%    PrimalDualInf  pdbOptCheck     pdbIspCheck   IspDualInf
%    my_union       vError
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% The method is described in:
% "A Projected-Search Interior Method for Nonlinear Optimization",
% by Philip E. Gill and Minxin Zhang.
%
% The underlying interior method is based on a primal-dual shifted
% penalty-barrier function in which both the primal and dual variables are
% shifted.  A method based on shifting only the dual variables is given in:
% "A Shifted Primal-Dual Penalty-Barrier Function for Nonlinear
% Optimization", by Philip E. Gill, Vyacheslav Kungurtsev and Daniel
% P. Robinson.
%
% At each iteration a summary line is printed with the following information:
%
% Itn       -- the iteration number
% nf        -- the cumulative total of function evaluations
% mu B      -- the current value of muB, the barrier parameter
% mu P      -- the current value of muP, the penalty parameter
% mu L      -- the current value of muL, the flexible line-search parameter
% Step      -- the line-search step length
% Norm p    -- the infinity norm of the primal line-search direction
% jf        -- the number of function evaluations for the current line-search
% Objective -- the value of the objective function
% Merit     -- the merit function defined  with muB, muL
% norm F    -- the norm of the perturbed KKT conditions
% M test    -- the proximity of  an M iterate
% M tol     -- the tolerance for an M iterate
% PrimInf   -- the primal infeasibilities
% DualInf   -- the dual infeasibilities
% BndViol   -- the violation of the bounds (should not be larger than the shift)
% InfComp   -- the complementarity measure for the infeasible stationary point problem
% EigMod    -- the magnitude of the ipopt-style diagonal modification
% Mod       -- the number of LDL factorizations needed for the diagonal modification
%
% An abbreviated summary of the iteration is printed at the end of each line:
%
% O  -- O iteration
% M  -- M iteration
% F  -- F iteration
% b  -- the multipliers were bounded
% i  -- the KKT matrix K was modified to have correct inertia
% s  -- the KKT matrix K was modified because of near-singularity
% x  -- some shifted variable bounds are infeasible
% X  -- some shifted slack    bounds are infeasible

% -------------------------------------------------------------------------
% Authors : Philip E. Gill and Minxin Zhang
% Date    : May 1st, 2026
% Purpose : Solves a nonlinearly constrained optimization problem.
% -------------------------------------------------------------------------
%
% 10 Oct 2016: First version.
% 20 Oct 2016: Updated for CUTEst.
% 17 Jun 2017: Flexible line search added.
% 26 Jun 2018: Infeasible slacks included in penalty term.
% 16 Sep 2018: Primal and dual infeasibilities used for termination.
% 17 Sep 2018: Scaling option added.
% 04 Nov 2018: Average step length for indefinite iterations.
% 10 Jun 2021: Modified for projected-search all-shifted extension
% 12 Jun 2021: Parameters printed. Output arguments updated for pdbTest.m
% 01 Jan 2022: muF, jLoF and jUpF renamed as muA, jLoA and jUpA
% 06 Jun 2023: g, f and J removed.
% 20 Jul 2023: muR and muP renamed muP and muL.
% 04 Aug 2023: No global variables.
% 07 Aug 2023: Code reordered to fix printing of normF, norm gM, etc.
% 20 Aug 2023: Removed initial step based on the average step.
% 01 May 2026: PolyForm Noncommercial License version.
% -------------------------------------------------------------------------

  format compact

% -------------------------------
% Assign local control parameters
% -------------------------------
  parms         = feval(pdb_parms);

  muB0          = parms.muB          ;  % barrier parameter
  muP0          = parms.muP          ;  % penalty parameter
  muL0          = parms.muL          ;  % flexible line search parameter
  muLMin        = parms.muLMin       ;  % minimum value of muL
  maxIterations = parms.maxIterations;  % maximum iterates allowed
  minPosEig0    = parms.minPosEig    ;  % smallest allowed pos. eigenvalues of B
  tolPrimFeas   = parms.tolPrimFeas  ;  % primal feasibility tolerance
  tolDualFeas   = parms.tolDualFeas  ;  % dual   feasibility tolerance
  tolIsp        = parms.tolIsp       ;  % Infeasible stationary point tolerance
  dxMax         = parms.dxMax        ;  % parameter used to define max step length
  tolM          = parms.tolM         ;  % unconst. opt tolerance for the merit function
  lines         = parms.lines        ;  % lines between printing header
  infinity      = parms.infinity     ;  % user-defined +infinity
  Assert        = parms.Assert       ;  % (0/1) check consistency
  yMax          = parms.yMax         ;  % largest ye for M-iterate
  unBoundedf    = parms.unBoundedf   ;  % unbounded objective value
  debugItn      = parms.debugItn     ;  % iteration for setting dbstop
  scaleOption   = parms.scaleOption  ;  % (0/1) scaling off/on
  maxNormF      = parms.maxNormF     ;  % Upper bound for normF in line search
% -----------------------
% End: control parameters
% -----------------------

% Initialize the run statistics

  Stats       = zeros(1,6);
  jCvEitns    = 1;  CvEitns = 0;
  jOitns      = 2;  Oitns   = 0;
  jFitns      = 3;  Fitns   = 0;
  jMitns      = 4;  Mitns   = 0;
  jnf         = 5;  nf      = 0;  % Number of function evaluations
  jitn        = 6;  itn     = 0;  % Iteration counter

% Initialize the counters

  jf          =  0;
  jmods       =  0;

  half        = 1/2;
  Blank       = [' ', ' ', ' ', ' ', ' ', ' '];   Info = Blank;

  pdb_print(parms,outfile);

% Define the indices associated with finite bounds.
% The first n elements correspond to the bounds on  x.
% The next  m elements correspond to the bounds on  c.

  m        = prob.m;
  n        = prob.n;

  bl       = prob.bl;
  bu       = prob.bu;
  cl       = prob.cl;
  cu       = prob.cu;

  bL       = [ bl; cl ];
  bU       = [ bu; cu ];

  if  ~isempty(find(cl <= -infinity & cu >= +infinity,1))
    fprintf('ERROR: free general constraints are not allowed. \n\n')
    return
  end

% Find the indices of the fixed variables and fixed slacks.
% A fixed slack means that the constraint is an equality.
%
% A variable that is not fixed is feasible and free to move.
% The indices of these variables constitute jfreex.
% A free variable may or may not have a finite upper or lower bound.

  jfreex   = find(bl ~= bu);         nfreex  = length(jfreex);
  jfixedx  = find(bl == bu);         nfixedx = length(jfixedx);

  ifreec   = find(cl ~= cu);
  ifixedc  = find(cl == cu);         nfixedc = length(ifixedc);

% Find jfree, jfixed and jfixedc,  all subsets of 1:n+m.

  jfree    = find(bL ~= bU);         nfree   = length(jfree);
  jfixed   = find(bL == bU);         nfixed  = length(jfixed);
  jfixedc  = n + ifixedc;

% Check that the number of free and fixed variables is consistent
% with numbers specified by the CUTEst format for the constraints.

  equality = [ prob.equatn ];
  if  nfixed+nfree ~= n+m || nfixedc ~= length(find(equality))
    fprintf('ERROR: mismatch between fixed and bounded variables.\n\n')
    return
  end

% Find jLo and jUp, the indices of the free bounded variables, i.e.,
%   jLo is a free variable or slack with a finite lower bound.
%   jUp is a free variable or slack with a finite upper bound.
% An index of a variable or slack subject to a range constraint will
% be in both jLo and jUp.  Variables with indices in jLo and jUp
% are included in the barrier term.

% A slack may be fixed temporarily by removing its index from
% jLo and/or jUp.  This means that the index is temporarily excluded
% from the barrier term.

% icLorU is the list of general constraints with finite lower or upper
% bounds.  No free general constraints (i.e., slacks) are allowed,
% which implies that icLorU is always the same as ifreec.

% jxLorU is the list of variables and slacks with finite lower or upper
% bounds.  A free variable may have no finite upper or lower bound,
% which implies that jxLorU is a subset of jfreex.

  jxLo     = find(bl ~= bu  &  bl > -infinity);
  jxUp     = find(bl ~= bu  &  bu < +infinity);

  jxLorU   = find(bl ~= bu  &  (bl > -infinity | bu < +infinity));
  icLorU   = find(cl ~= cu  &  (cl > -infinity | cu < +infinity));

  jcLorU   = n+icLorU;              % subset of 1:n+m

  icLo     = find(~equality & (cl > -infinity));
  jLo      = my_union(jxLo,n+icLo); % subset of 1:n+m

  icUp     = find(~equality & (cu <  infinity));
  jUp      = my_union(jxUp,n+icUp); % subset of 1:n+m

  if  length(jcLorU) + nfixedc < m
    fprintf('WARNING: %3g free slacks.\n\n', m-length(jcLorU)-nfixedc)
  end

% Keep copies of the constraint types.

  jLo0     = jLo;              jUp0   = jUp;

% If a variable moves outside its shifted bound, the violation is
% included in the merit function as a shifted augmented Lagrangian term
% with penalty muA.  These terms are indexed using jLoA and jUpA.

  jLoA     =  [];              jUpA   = [];

  noLogB   = isempty(jLo) &&  isempty(jUp);

  head1    = '  Itn    nf';
%             12345123456
  head2a   = '';                               % UC
  head2b   = '   mu B ';                       % BC
  head2c   = '   mu P    mu L ';               %    EQ(no bounds)
  head2d   = '   mu B    mu P    mu L ';       %    EQ(   bounds) NC
%             123456781234567812345678
  head3    = '   Step   Norm p jf    Objective  ';
%             1234567812345678123123456789012345
  head4a   = '';                                       % UC
  head4b   = '     Merit       norm F  M test   M tol';% BC EQ NC
%             123456789012345123456781234567812345678
  head5a   = ' DualInf ';                              % UC
  head5b   = ' PrimInf  DualInf ';                     % BC EQ NC
%             12345678-12345678-
  head6a   = '';                               % UC EQ(no bounds)
  head6b   = ' BndViol';                       % BC NC
  head6c   = ' BndViol InfComp';               % BC NC
%             1234567812345678

  head7    = '  EigMod Mod';

  if      m ==0
    if  noLogB        % unconstrained
      head2 = head2a; head4 = head4a; head5 = head5a; head6 = head6a;
    else              % bound constrained
      head2 = head2b; head4 = head4b; head5 = head5b; head6 = head6b;
    end
  else
    if  noLogB        % equality constrained (no bounds)
      head2 = head2c; head4 = head4b; head5 = head5b; head6 = head6a;
    else              % equalities and inequalities
      head2 = head2d; head4 = head4b; head5 = head5b; head6 = head6c;
    end
  end

  header   = [ head1 head2 head3 head4 head5 head6 head7 ];

  vars     = (1:n)';
  slacks   = (n+1:n+m)';   cons = slacks;  % Transpose for m = 0.

  mOnes    = ones (m,1);
  nOnes    = ones (n,1);
  nmOnes   = ones (n+m,1);
  mZeros   = zeros(m,1);
  nZeros   = zeros(n,1);
  nmZeros  = zeros(n+m,1);

% Initialize vectors that get scattered with data.

  x        = nmZeros;  %  [ x     s    ]
  dx       = nmZeros;  %  [ dx   ds    ]

% Primal variables.
% =================
%
%  x(j)     is x(j), j = 1:n
%  x(n+i)   is s(i), i = 1:m
%  In reference [1], x is  [ x  s ]
%
%  xe       is an estimate of the optimal value of x.
%
% A variable is free, fixed, or temporarily fixed.
%
% For free variables:
% -------------------
%
%  xL(j)    is x(j)  - bl(j) + muB, j = 1:n
%  xL(n+i)  is s(i)  - cl(i) + muB, i = 1:m
%
%  xU(j)    is bu(j) - x(j)  + muB, j = 1:n
%  xU(n+i)  is cu(i) - s(i)  + muB, i = 1:m
%
%  xeL(j)   is xe(j) - bl(j) + muB, j = 1:n
%  xeL(n+i) is se(i) - cl(i) + muB, i = 1:m
%  xeU(j)   is bu(j) - x (j) + muB, j = 1:n
%  xeU(n+i) is cu(i) - se(i) + muB, i = 1:m
%
%  xeL      is an estimate of the optimal value of xL.
%  xeU      is an estimate of the optimal value of xU.

%  In the notation of Reference [1]:
%       x   is  [ x              s            ],
%       xe  is  [ xe             se           ],
%       xL  is  [(x_1  + muB e) (s_1  + muB e)],
%       xU  is  [(x_2  + muB e) (s_2  + muB e)],
%       xeL is  [(xe_1 + muB e) (se_1 + muB e)],
%       xeU is  [(xe_2 + muB e) (se_2 + muB e)].
%
% For fixed variables:
% --------------------
%
%  xL(j)    is x(j) - bl(j),   j = 1:n
%  xL(n+i)  is s(i) - cl(i),   i = 1:m
%
%  xU(j)    is x(j) - bu(j),   j = 1:n
%  xU(n+i)  is cu(i) - s(i),   i = 1:m
%
% For temporary bounds:
% ---------------------
%
%  jLoA     are the indices of variables violating  x  - bL + muB.
%  jUpA     are the indices of variables violating  bU - x  + muB.
%
%  Implicitly, these variables define temporary equalities
%    A x - b = 0,
%  with
%    A = [  ILoA
%          -IUpA ]
%  where ILoA and IUpA are the matrices of normals of the temporary lower
%  and upper bounds and b is the vector of associated right-hand sides.
%
%  Ax - b is enforced using an augmented Lagrangian term, so that
%  x(jLoA) and x(jUpA) occur only in the  augmented Lagrangian term
%

% Dual variables.
% ===============
%
%  y(i)     is the multiplier for  c(i) - s(i)   = 0, i = 1:m.
%
%  z    = zFX + zL - zU is the (n+m)-vector of multiplier estimates.
%
%  zFX(j)   is   0  if x(i) is a free  variable,  j = 1:n.
%  zFX(n+i) is   0  if s(i) is a free  variable,  i = 1:m.
%  zFX(n+i) is y(i) if s(i) is a fixed variable,  i = 1:m.
%  zFX      is stored in the free components of z.
%
%  zL(j)    is the multiplier for  x(j) - bl(j) >= 0, j = 1:n.
%  zL(n+i)  is the multiplier for  s(i) - cl(i) >= 0, i = 1:m.
%
%  zU(j)    is the multiplier for  bu(j) - x(j) >= 0, j = 1:n.
%  zU(n+i)  is the multiplier for  cu(i) - s(i) >= 0, i = 1:m.
%
%  In the notation of Reference [1], z   is  [ z    w   ],
%                                    zL  is  [ z_1  w_1 ],
%                                    zU  is  [ z_2  w_2 ].
%
%  piL(j)   is the barrier multiplier for x(j) - bl(j) + muB >= 0, j = 1:n.
%  piL(n+i) is the barrier multiplier for s(i) - cl(j) + muB >= 0, i = 1:m.
%
%  piU(j)   is the barrier multiplier for bu(j) - x(j) + muB >= 0, j = 1:n.
%  piU(n+i) is the barrier multiplier for cu(j) - s(j) + muB >= 0, i = 1:m.
%
%  piZ = piL - piU is the vector of barrier multipliers.
%                  piZ is an estimate of zL - zU.
%
%  piY      is the vector of augmented Lagrangian multipliers.
%                  piY is an estimate of y.
%
%  piVL(jLoA) are the penalty multipliers for  x(jLoA) -  bL(jLoA)
%  piVU(jUpA) are the penalty multipliers for bU(jUpA) -   x(jUpA)
%  piV = piVL - piVU is the vector of augmented Lagrangian multipliers.
%                    piV is an estimate of v.

%  Reference
%  ---------
%  [1] Equations for a Projected-Search Path-Following Method for
%      Nonlinear Optimization,  P. E. Gill and Minxin Zhang,
%      Technical Report CCoM-22-2, June 1, 2022.

  xL       = nmZeros;  xU       = nmZeros;
  xLinv    = nmZeros;  xUinv    = nmZeros;

%    Algorithm Overview
%    ------------------
%    Compute initial primal variables;
%    Compute problem functions;
%    Compute initial slack  variables;
%    Compute initial dual   variables and pdb parameters;
%
%    begin Loop
%      Compute merit function and gradient;
%      Test for first-order optimality or termination.
%      Print iteration details;
%
%      if  first-order optimal then
%         Factor K;
%         Check second-order optimality;
%         if  optimal, break;
%      end
%      Factor K and compute the search direction;
%      Compute a maximum step;
%      Find the quasi-Armijo step and the new merit function;
%      Reset the slack variables;
%      Check for pdb parameter update;
%      if  parameter update then
%         Update pdb parameters;
%      end
%    end Loop

  muL        = muL0*mOnes;
  muP        = muP0*mOnes;
  muB        = muB0*nmOnes;
  muA0       = muB0;
  muA        = muB0*nmOnes;

  icViolL    = [];   icViolU   = [];
  jcViolL    = [];   jcViolU   = [];
  tjfixedcL  = [];   tjfixedcU = []; tjfixedc = [];

  muAfac     = 10;
  muPfac     = 10;
  muBfac     = 10;

  S          = [];
  status     = 0;
  outcome    = '';

% -------------------------------------------------------------------------
% Compute the initial primal and dual variables;
% -------------------------------------------------------------------------
  x(vars)    =  prob.x;
  x(jfixedx) =  bl(jfixedx);  % =  bu(jfixedx)

  ye         = -prob.v;

  IntStart   =  false;
  pert       =  1;

  if IntStart
    for  i   = jfreex'
      if  bl(i) > -infinity
        pertL = max(muB(i)/(abs(bl(i))+1),pert);
        if  bu(i) < infinity
        % Range
          pertU = max(muB(i)/(abs(bu(i))+1),pert);
          xMid  = half*(bl(i) + bu(i));
          if  xMid < bl(i) + pertL || xMid > bu(i)-pertU
            x(i) = xMid;
          elseif  x(i) < xMid
            x(i) = max(bl(i)+pertL, x(i));
          else
            x(i) = min(bu(i)-pertU, x(i));
          end
        else
        % lower bound
          x(i) = max(bl(i)+pertL, x(i));
        end
      else
        if  bu(i) < infinity
          pertU = max(muB(i)/(abs(bu(i))+1),pert);
          x(i)  = min(bu(i)-pertU, x(i));
        end
      end
    end
  else
    for  i = jfreex'
      if  bl(i) > -infinity

        if  bu(i) < infinity
        % Range
          if  x(i) < bl(i) || x(i) > bu(i)
            if  x(i) < bl(i)
              x(i) = bl(i);
            else
              x(i) = bu(i);
            end
          end
        else
        % lower bound
          x(i) = max(bl(i), x(i));
        end
      else
        if  bu(i) < infinity
          x(i)  = min(bu(i), x(i));
        end
      end
    end
  end

% -------------------------------------------------------------------------
% Compute the problem functions. Scale if requested.
% -------------------------------------------------------------------------
  [fx,gx]   = prob.obj(x(vars));   nf = nf + 1;

  if  m > 0
    [cx,Jx] = prob.cons(x(vars));
  else
    cx      = zeros(m,1);          Jx = zeros(m,n);
  end

  if  (sum(isnan([fx; gx])) + sum(isinf([fx; gx]))) ~= 0
    fprintf('ERROR: Undefined problem functions at initial point.\n\n')
    outcome         = 'Initial funs undefined';
    solution.status = status;
    solution.x      = x(vars);
    solution.y      = zeros(m,1);
    solution.fx     = fx;
    return
  end

  if  scaleOption == 1
  % S          = ( Sc ; Sr )
  % Jx(scaled) = inv(diag(Sr))*Jx*diag(Sc)
  % x(scaled)  = inv(Sc)*x,     slacks(scaled) = inv(Sr)*slacks,

    [S]        = scaleDerivs([Jx ; gx']);
    Sobj       = S(n+m+1);               S          = S(1:n+m);

    fx         = fx/Sobj;                gx         = gx.*S(vars)/Sobj;
    Jx         = diag(1./S(cons))*Jx*diag(S(vars));
    cx         = cx./S(cons);
    x(vars)    = x(vars)./S(vars);       ye         = ye.*S(cons);
  % cl         = cl./S(cons);            cu         = cu./S(cons);
  % bL         = bL./S;                  bU         = bU./S;
    bL(jLo)    = bL(jLo)./S(jLo);        bU(jUp)    = bU(jUp)./S(jUp);
    bL(jfixed) = bL(jfixed)./S(jfixed);  bU(jfixed) = bU(jfixed)./S(jfixed);
    cl         = bL(cons)       ;        cu         = bU(cons);
  else
    Sobj       = 1;
  end

  normgx       = norm(gx);
  normcx       = norm(cx);
  normJx       = norm(Jx,Inf);

  Prob.m       = m;
  Prob.n       = n;
  Prob.bL      = bL;
  Prob.bU      = bU;
  Prob.vars    = vars;
  Prob.slacks  = slacks;
  Prob.jfreex  = jfreex;
  Prob.nfreex  = nfreex;
  Prob.jfixed  = jfixed;
  Prob.jfixedx = jfixedx;
  Prob.nfixed  = nfixed;

% -------------------------------------------------------------------------
% Initialize the slacks.
% -------------------------------------------------------------------------
  x(slacks)  = cx;
  x(jfixedc) = cl(ifixedc);

  if IntStart
    for i = ifreec'
      j   = n + i;
      if  cl(i) > -infinity
        pertL = max(muB(i)/(abs(cl(i))+1),pert);
        if  cu(i) < infinity
        % Range
          pertU = max(muB(i)/(abs(cu(i))+1),pert);
          xMid  = half*(cl(i) + cu(i));
          if  xMid < cl(i) + pertL || xMid > cu(i)-pertU
            x(j) = xMid;
          elseif  x(j) < xMid
            x(j) = max(cl(i)+pertL, x(j));
          else
            x(j) = min(cu(i)-pertU, x(j));
          end
        else
        % Lower bound
          x(j) = max(cl(i)+pertL, x(j));
        end
      elseif  cu(i) < infinity
      % Upper bound
        pertU  = max(muB(i)/(abs(cu(i))+1),pert);
        x(j)   = min(cu(i)-pertU, x(j));
      end
    end
  else
    for i = ifreec'
      j   = n + i;
      if  cl(i) > -infinity
        if  cu(i) < infinity
        % Range
          if  x(j) < cl(i) || x(j) > cu(i)
            if  x(j) < cl(i)
              x(j) = cl(i);
            else
              x(j) = cu(i);
            end
          end
        else
        % Lower bound
          x(j) = max(cl(i), x(j));
        end
      elseif  cu(i) < infinity
        % Upper bound
        x(j)   = min(cu(i), x(j));
      end
    end
  end
  cs         = cx - x(slacks);

% -------------------------------------------------------------------------
% Initialize the dual variables and pd parameters.
% Compute the optimality criteria at the initial point.
% -------------------------------------------------------------------------
  ze         = [ gx - Jx'*ye ; ye ];
  zeL        =  nmZeros;            zeU         = nmZeros;
  zeL(jLo)   =  pert*max(ze(jLo), nmOnes(jLo));
  zeU(jUp)   = -pert*min(ze(jUp),-nmOnes(jUp));

  xe         =  max(min(x,bU),bL);
  xeL        =  nmZeros;            xeU         = nmZeros;
  xeL(jLo)   =  max( x(jLo)-bL(jLo)+muB(jLo), muB(jLo));
  xeU(jUp)   =  max(bU(jUp)- x(jUp)+muB(jUp), muB(jUp));

  zL         =  nmZeros;            zU          = nmZeros;
  zL(jLo)    =  zeL(jLo);           zU(jUp)     = zeU(jUp);

  z          =  zL - zU;            z(jfixed)   = ze (jfixed);
  y          =  z(slacks);          ye          = y;

  piL        =  nmZeros;            piU         = nmZeros;

  zML        =  nmZeros;            zMU         = nmZeros;
  zM         =  nmZeros;

  vL         =  nZeros;             vU          = nZeros;
  v          =  nZeros;
  dvL        =  nZeros;             dvU         = nZeros;
  veL        =  nZeros;             veU         = nZeros;
  ve         =  nZeros;
  vML        =  nZeros;             vMU         = nZeros;
  piVL       =  nZeros;             piVU        = nZeros;
  piV        =  nZeros;

  Hmod       =  0;
  eigMod     =  0;
  Mtest      =  0;
  mods       =  0;
  normdx     =  0;
  normx      =  norm(x,Inf);
  normgM     =  0;
  step       =  0;

  Eitns      =  0;

  minPosEig  =  minPosEig0;

  zScale     =  max([ 1, normgx, max(1,norm(y))*normJx ]);
  cScale     =  max([ 1, normcx*normJx ]);

  [maxComp,maxViolAll,maxViolBnds,maxStny] ...
             = pdbOptCheck(gx,Jx,cs,x,y,v,z,zL,zU,zScale,Prob);
  [maxCompIsp] ...
             = pdbIspCheck(Jx,cs,x,cScale,bL,bU);
  [primInf,dualInf] ...
             = PrimalDualInf(gx,Jx,x,cx,y,zScale,Prob);
  dualInfIsp = IspDualInf(Jx,cs,x,cScale,bL,bU);

  phiMax     = max(maxViolAll+maxComp+10,1000);

  NearOpt    = false;

  solution.status  = status;
  solution.x       = x(vars);
  solution.y       = y;
  solution.fx      = fx;
  solution.gx      = gx;
  solution.cx      = cx;
  solution.primInf = primInf;
  solution.itn     = itn;
  solution.nf      = nf;
  solution.mods    = mods;

  [fMmuL,fMmuP,gMmuP,F,xL,xU,xLinv,xUinv,          ...
                 piL,piU,piZ,piY,piVL,piVU,piV,    ...
                 zML,zMU,zM,yM,vML,vMU,vM,DBL,DBU] ...
      = pdbMerit(muA,muB,muL,muP,x,y,              ...
                 fx,gx,cx,Jx,cs,                   ...
                 jLo,jUp,icLo,icUp,jLoA,jUpA,      ...
                 z,zL,zU,v,vL,vU,                  ...
                 xeL,xeU,zeL,zeU,ye,veL,veU,Prob);
  normF   = norm(F    ,Inf);
  normgM  = norm(gMmuP,Inf);
  fM0     = fMmuP;

% -------------------------------------------------------------------------
% Main loop.
% -------------------------------------------------------------------------
  while (1)
    if  any(zL(jLo) + muB(jLo) <= 0) || any(zU(jUp) + muB(jUp) <= 0)
      [zL,zU] = infeasMul(muB,muBfac,gx,y,Jx,jLo,jUp,v,zL,zU,Prob);
    end

  % -----------------------------------------------------------------------
  % Test for optimality or termination.
  % -----------------------------------------------------------------------
    zScale     = max([ 1, normgx, max(1,norm(y))*normJx ]);
    cScale     = max([ 1, normcx*normJx ]);

    [maxComp,maxViolAll,maxViolBnds,maxStny]     ...
               = pdbOptCheck(gx,Jx,cs,x,y,v,z,zL,zU,zScale,Prob);
    maxCompIsp = pdbIspCheck(Jx,cs,x,cScale,bL,bU);
    [primInf,dualInf]                            ...
               = PrimalDualInf(gx,Jx,x,cx,y,zScale,Prob);
    dualInfIsp = IspDualInf(Jx,cs,x,cScale,bL,bU);

    if      primInf     <= tolPrimFeas        && ...
            dualInf     <= tolDualFeas
      status = 1;
    elseif  primInf     <= tolPrimFeas        &&  fx  <= unBoundedf
      status = 3;
    elseif  maxViolAll  >= tolPrimFeas        && ...
            maxViolBnds <= tolPrimFeas        && ...
            dualInfIsp  <= tolIsp*primInf     && ...
            step*normdx <= 1.0e-12*(1+normx)  && ...
            itn         >  0
      status = 2;
    elseif (primInf     <= tolPrimFeas*10     && ...
            dualInf     <= tolDualFeas   )    || ...
           (primInf     <= tolPrimFeas        && ...
            dualInf     <= tolDualFeas*10)
      NearOpt = true;
    end

  % -----------------------------------------------------------------------
  % Print iteration details.
  % -----------------------------------------------------------------------
    nLine    = rem( itn, lines );
    if  nLine == 0
      fprintf('\n%s\n', header);  fprintf(outfile, '\n%s\n', header);
    end

    str7     = sprintf('%8.1e%4g',            eigMod, jmods );

    if     m == 0
      if  noLogB  % unconstrained
        str1   = sprintf('%5g%6g',               itn, nf );
        str2   = sprintf('%s',                   '' );
        str3   = sprintf('%8.1e%8.1e%3g%15.7e',  step, normdx, jf, fx*Sobj );
        str4   = sprintf('%s',                   '' );
        str5   = sprintf('%8.1e ',               dualInf );
        str6   = sprintf('%s',                   '' );
        if  dualInf <= tolDualFeas
          str5(1)  = '(';  str5(9)  = ')';
        end
      else       % bound constrained
        str1   = sprintf('%5g%6g',               itn, nf );
        str2   = sprintf('%8.1e',                min(muB) );
        str3   = sprintf('%8.1e%8.1e%3g%15.7e',  step, normdx, jf, fx*Sobj );
        str4   = sprintf('%15.7e%8.1e%8.1e%8.1e',fMmuL, normF, Mtest, tolM );
        str5   = sprintf('%8.1e %8.1e ',         primInf, dualInf );
        str6   = sprintf('%8.1e',                maxViolBnds );
        if  primInf    <= tolPrimFeas
          str5(1)  = '(';  str5(9)  = ')';
        end
        if  dualInf    <= tolDualFeas
          str5(10) = '(';  str5(18) = ')';
        end
      end
    else % m > 0
      if  noLogB  % equality constrained
        normcs = norm(cs, Inf);
        str1   = sprintf('%5g%6g',               itn, nf );
        str2   = sprintf('%8.1e%8.1e',           min(muP), min(muL) );
        str3   = sprintf('%8.1e%8.1e%3g%15.7e',  step, normdx, jf, fx*Sobj );
        str4   = sprintf('%15.7e%8.1e%8.1e%8.1e',fMmuL, normF, Mtest, tolM );
        str5   = sprintf('%8.1e %8.1e ',         primInf, dualInf );
        str6   = sprintf('%s',                   '' );
        if  primInf    <= tolPrimFeas
          str5(1)  = '(';  str5(9)  = ')';
        end
        if  dualInf    <= tolDualFeas
          str5(10) = '(';  str5(18) = ')';
        end
      else       % equalities and inequalities
        normcs = norm(cs, Inf);
        str1   = sprintf('%5g%6g',               itn, nf );
        str2   = sprintf('%8.1e%8.1e%8.1e',      min(muB), min(muP), min(muL) );
        str3   = sprintf('%8.1e%8.1e%3g%15.7e',  step, normdx, jf, fx*Sobj );
        str4   = sprintf('%15.7e%8.1e%8.1e%8.1e',fMmuL, normF, Mtest, tolM );
        str5   = sprintf('%8.1e %8.1e ',         primInf, dualInf );
        str6   = sprintf('%8.1e%8.1e',           maxViolBnds, dualInfIsp );
        if  primInf    <= tolPrimFeas
          str5(1)  = '(';  str5(9)  = ')';
        end
        if  dualInf    <= tolDualFeas
          str5(10) = '(';  str5(18) = ')';
        end
      end
    end

    if  strcmpi(Info,Blank)
      str    = [str1 str2 str3 str4 str5 str6 str7];
      fprintf('%s\n', str);    fprintf(outfile, '%s\n', str);
    else
      str8   = sprintf(' %s',                     Info);
      str    = [str1 str2 str3 str4 str5 str6 str7 str8];
      fprintf('%s\n', str);    fprintf(outfile, '%s\n', str);
      Info(1:4) = Blank(1:4);
    end

  % -----------------------------------------------------------------------
  % Test for convergence or termination.
  % -----------------------------------------------------------------------
    Terminated = status > 0 || itn > maxIterations - 1;
    if  Terminated
      if      status == 1
        outcome = 'Converged';
      elseif  status == 2
        outcome = 'Inf Stationary Point';
      elseif  status == 3
        outcome = 'Unbounded problem';
      elseif  itn > maxIterations - 1
        if  NearOpt
          status  = 4;
          outcome = 'Near Optimal';
        else
          status  = 5;
          outcome = 'Too many iterations';
        end
      else
        status  = 6;
      end
      break
    end

  % -----------------------------------------------------------------------
  % Compute the merit function and its gradient.
  % Variables xL, xU, xLinv, xUinv, etc., are also computed.
  % -----------------------------------------------------------------------
    [fMmuL,fMmuP,gMmuP,F,xL,xU,xLinv,xUinv,          ...
                   piL,piU,piZ,piY,piVL,piVU,piV,    ...
                   zML,zMU,zM,yM,vML,vMU,vM,DBL,DBU] ...
        = pdbMerit(muA,muB,muL,muP,x,y,              ...
                   fx,gx,cx,Jx,cs,                   ...
                   jLo,jUp,icLo,icUp,jLoA,jUpA,      ...
                   z,zL,zU,v,vL,vU,                  ...
                   xeL,xeU,zeL,zeU,ye,veL,veU,Prob);
    normF   = norm(F    ,Inf);
    normgM  = norm(gMmuP,Inf);

    if  scaleOption == 1   % Unscale
      x(vars) = x(vars).*S(vars);  y = Sobj*y./S(cons);
    end

    if  m > 0
      Hx      = prob.hess(x(vars),-y );
    else
      Hx      = prob.hess(x(vars));
    end

    if  scaleOption == 1
      Hx      = diag(S(vars))*Hx*diag(S(vars))/Sobj;
      x(vars) = x(vars)./S(vars);  y = y.*S(cons)/Sobj;
    end

    itn = itn + 1;
    if  itn >= debugItn
      jj = 1; % Set break here
    end

  % -----------------------------------------------------------------------
  % Compute the scaled KKT matrix and the primal-dual search direction.
  % -----------------------------------------------------------------------
    xScale     = nOnes;          xScale2   = nOnes;
    ADA        = nZeros;         DZinv     = nZeros;

    ADA(jLoA)  = 1./muA(jLoA);   ADA(jUpA) = 1./muA(jUpA);

    DY         = muP.*mOnes;
    DW         = mZeros;         DWinv     = mZeros;

    DWinv(icLorU) =     xLinv(jcLorU).*(zL(jcLorU)+muB(jcLorU))+xUinv(jcLorU).*(zU(jcLorU)+muB(jcLorU));
    DW   (icLorU) = 1./(xLinv(jcLorU).*(zL(jcLorU)+muB(jcLorU))+xUinv(jcLorU).*(zU(jcLorU)+muB(jcLorU)));

    DZinv(jxLo)   =                  xLinv(jxLo).*(zL(jxLo)+muB(jxLo));
    DZinv(jxUp)   =    DZinv(jxUp) + xUinv(jxUp).*(zU(jxUp)+muB(jxUp));

  % xScale2(jxLorU) = ADA(jxLorU);

    yScale2    = mOnes;           yScale2(icLorU) = DWinv(icLorU);
    yScale     = mOnes;           yScale (icLorU) = sqrt(DWinv(icLorU));
    Iy         = mZeros;          Iy     (icLorU) = mOnes(icLorU);

    Hk         =  Hx(jfreex,jfreex);
    Dk1        =  diag(DZinv(jfreex)) + diag(ADA(jfreex));
    Jk         =  diag(yScale)*Jx(:,jfreex);
    Dk2        =  diag(yScale2.*DY + Iy);
    K          =  [  Hk+Dk1   Jk'
                     Jk      -Dk2 ];
    rhs1       =  gx(jfreex) - Jx(:,jfreex)'*y      - piZ(jfreex) - piV(jfreex);
    rhs2       =  yScale.*(DY.*(y - piY)  +  DW.*(y - piZ(slacks)));
    rhs        = -[ rhs1
                    rhs2 ];
  % =======================================================================
  % ktsolver solves the (nfreex+m) by (nfreex+m) condensed primal-dual KKT
  % system using the symmetric indefinite method implemented in the Matlab
  % command  LDL.  If necessary a diagonal modification of H is used to
  % give a modified K with the correct inertia.
  % =======================================================================
    [dfree,eigMod,jmods,Hmod,exitInfo] ...
                 = ktSolverSparse( K,rhs,nfreex,m,minPosEig,Hmod );
    mods         =  mods + jmods;

    if  m > 0
      dfree(nfreex+1:nfreex+m) = -yScale.*dfree(nfreex+1:nfreex+m);
    end
    d            =  nmZeros;
    d(jfreex)    =  dfree(1:nfreex); d(slacks)= dfree(nfreex+1:nfreex+m);

    dy           =  d(slacks);     yStep      =  y + dy;
    dx(vars)     =  d(vars);       dx(slacks) = -DW.*(yStep - piZ(slacks));

    xStep        =  x + dx;        normdx     =  norm(dx(vars),Inf);

    dzL          =  nmZeros;       dzU        =  nmZeros;
    dzL(jLo)     = -xLinv(jLo).*(zL(jLo).*(xStep(jLo) -    bL(jLo)) ...
                   +  muB(jLo).*(zL(jLo) -   zeL(jLo) + xStep(jLo) -    xe(jLo)));
    dzU(jUp)     = -xUinv(jUp).*(zU(jUp).*(   bU(jUp) - xStep(jUp)) ...
                   +  muB(jUp).*(zU(jUp) -   zeU(jUp) +    xe(jUp) - xStep(jUp)));

    piVL         =  nZeros;           piVU    =  nZeros;
    piVL(jLoA)   =  veL(jLoA) - (xStep(jLoA) -    bL(jLoA))./muA(jLoA);
    piVU(jUpA)   =  veU(jUpA) - (   bU(jUpA) - xStep(jUpA))./muA(jUpA);
    piV          =  piVL      - piVU;

    dvL          =  nZeros;           dvU     =  nZeros;
    dvL(jLoA)    =  piVL(jLoA) - vL(jLoA);
    dvU(jUpA)    =  piVU(jUpA) - vU(jUpA);

    dzX          =  nmZeros;
    dzX(jfixedc) =  yStep(ifixedc)  - z(jfixedc);
    dzX(jfixedx) =  gx(jfixedx) + Hx(jfixedx,:)*dx(vars) ...
                                - Jx(:,jfixedx)'*yStep - z(jfixedx);
    if Assert
    % ---------------------------------------------------------------------
    % Based on the Assert parameter, the computations are verified by
    % computing all quantities using the uncondensed PD equations and
    % doubly-augmented system.
    % ---------------------------------------------------------------------
      checkKKT(muA,muB,muL,muP,eigMod,x,y,            ...
               fx,gx,cx,Jx,Hx,                        ...
               xL,xU,xLinv,xUinv,                     ...
               piL,piU,piZ,piY,piVL,piVU,piV,         ...
               dx,dy,dzL,dzU,                         ...
               jLo,jUp,jxLo,jxUp,icLo,icUp,jLoA,jUpA, ...
               zL,zU,vL,vU,                           ...
               zeL,zeU,ye,veL,veU,Prob)
    end

    if      exitInfo ~= 0
      CvEitns   = CvEitns + 1;
      if    exitInfo == -1
        outcome = 'No modification';  break
      else
        Eitns = Eitns + 1;
        if      exitInfo == 1
          Info(3) = 's';
        elseif  exitInfo == 2
          Info(3) = 'i';
        elseif  exitInfo == 3
          Info(3) = 'I';
        end
      end
    else
      Eitns   = 0;
    end

  % -----------------------------------------------------------------------
  % Compute the flexible quasi-Armijo step and the new merit function.
  % -----------------------------------------------------------------------
    muPused    = false;
    muPreduced = false;

    step0    = 1;

    stepLimit  =  dxMax/normdx;
    step       =  min([ 1 step0 stepLimit ]);

    [step,fMmuL,fMmuP,gMmuP,                             ...
                       x,y,vL,vU,z,zL,zU,                ...
                       fx,gx,cx,Jx,cs,F,                 ...
                       xL,xU,xLinv,xUinv,                ...
                       piL,piU,piZ,piY,piVL,piVU,piV,    ...
                       zML,zMU,zM,yM,vML,vMU,vM,DBL,DBU, ...
                       jf,iExit,muPused,maxNormF]        ...
        = pdbqArmijoLS(muA,muB,muL,muP,maxNormF,         ...
                       step,fMmuL,fMmuP,gMmuP,F,         ...
                       fx,gx,cx,Jx,cs,                   ...
                       jLo,jUp,icLo,icUp,jLoA,jUpA,      ...
                       x,y,vL,vU,z,zL,zU,                ...
                       dx,dy,dvL,dvU,dzL,dzU,dzX,        ...
                       scaleOption,S,Sobj,               ...
                       xeL,xeU,ye,veL,veU,zeL,zeU,       ...
                       prob,Prob,parms);

    normgx   = norm(gx);

    normF    = norm(F    ,Inf);
    normgM   = norm(gMmuP,Inf);

    if  m > 0
      normcx  = norm(cx);                   normJx  = norm(Jx,Inf);
    end

    nf = nf + jf;

    if  iExit > 3
      if  NearOpt
        status  = 4;
        outcome = 'Near Optimal';
      else
        outcome = 'Linesearch failure';
        status  = 5;
      end
      break
    end

  % Update sum of step lengths associated with a sequence of modified KKTs
  % It  is used to compute the average step.

    if  Assert
      if  any( x(jLo)-bL(jLo)+muB(jLo) <= 0) || ...
          any(bU(jUp)- x(jUp)+muB(jUp) <= 0)
        fprintf('WARNING : x out of bounds (LS).\n')
      end
      if  any(zL(jLo)+muB(jLo)         <= 0) || ...
          any(zU(jUp)+muB(jUp)         <= 0)
        fprintf('WARNING : z out of bounds (LS).\n')
      end
    end

  % -----------------------------------------------------------------------
  % Optimize the slack variables.
  % -----------------------------------------------------------------------
    if  ~isempty(jcLorU)
      if  muPused
        sReset  = cx - muP.*(ye + half*(z(slacks) - y) + muB(slacks));
      else
        sReset  = cx - muL.*(ye + half*(z(slacks) - y) + muB(slacks));
      end
      x(slacks) = pdbslackReset(n,m,x,bL,bU,sReset,jcLorU,pdb_parms);
    end
    cs          = cx  - x(slacks);

    if  Assert
      if  any(x(jLo)-bL(jLo)+muB(jLo) <= 0) || any(bU(jUp)-x(jUp)+muB(jUp) <= 0)
        fprintf('WARNING : Slacks out of bounds.\n')
      end
    end

  % -----------------------------------------------------------------------
  % Check the criteria for updating ye, etc.
  % The optimality measures at the new point are needed.
  % -----------------------------------------------------------------------
    zScale  = max([ 1, normgx, max(1,norm(y))*normJx ]);
    cScale  = max([ 1, normcx*normJx ]);

    [maxComp,maxViolAll,maxViolBnds,maxStny] ...
                      = pdbOptCheck(gx,Jx,cs,x,y,v,z,zL,zU,zScale,Prob);
    maxCompIsp        = pdbIspCheck(Jx,cs,x,cScale,bL,bU);
    [primInf,dualInf] = PrimalDualInf(gx,Jx,x,cx,y,zScale,Prob);
    dualInfIsp        = IspDualInf(Jx,cs,x,cScale,bL,bU);

    Mtestx        = norm(gMmuP(jfreex),Inf)                      ;
    Mtesty        = norm([y          - piY; yM - zM(slacks)],Inf);
    MtestFL       = norm(vL(jLoA)    - piVL(jLoA),Inf)           ;
    MtestFU       = norm(vU(jUpA)    - piVU(jUpA),Inf)           ;
    MtestzL       = norm(zL(jLo)     - piL (jLo ),Inf)           ;
    MtestzU       = norm(zU(jUp)     - piU (jUp ),Inf)           ;

    Mtest         = norm([Mtestx Mtesty MtestFL MtestFU MtestzL MtestzU],Inf) ...
                         /max([1 fM0]);

    phiO          = primInf + dualInf;

    if  phiO <= phiMax
      Info(1)     = 'O';                         Oitns       = Oitns + 1;
      phiMax      = half*phiMax;
      xe          = max(min(x,bU),bL);
      xeL(jLo)    = max(xL(jLo),muB(jLo));       xeU(jUp)    = max(xU(jUp),muB(jUp));
      ye          = y;

      zeL         = nmZeros;                     zeU         = nmZeros;
      zeL(jLo)    = zL (jLo);                    zeU(jUp)    = zU (jUp) ;
      ze          = z;

      veL(jLoA)   = vL (jLoA);                   veU(jUpA)   = vU (jUpA);
      ve          = veL - veU;
    else

      if  Mtest <= tolM
        Info(1) = 'M';                           Mitns       = Mitns + 1;
        if  abs(y) > yMax
          ye          = max(-yMax, min(y,           yMax));
          zeL(slacks) = max(-yMax, min(zeL(slacks), yMax));
          zeU(slacks) = max(-yMax, min(zeU(slacks), yMax));
          ze (slacks) = zeL(slacks) - zeU(slacks);
          Info(2) = 'b';
        end

        if  norm(cs) > tolM
          muP         = muP/muPfac;
          muPreduced  = true;
        end

        if  maxComp > tolM || any( x(jLo)-bL(jLo)+tolM < 0) ...
                           || any(bU(jUp)- x(jUp)+tolM < 0) ...
                           || any(zL(jLo)        +tolM < 0) ...
                           || any(zU(jUp)        +tolM < 0)
          muB    = muB./muBfac;
          muA    = muA./muAfac;
        end

        xe       = max(min(x,bU),bL);
        xeL(jLo) = max(min(xe(jLo)-bL(jLo)+muB(jLo),yMax),muB(jLo));
        xeU(jUp) = max(min(bU(jUp)-xe(jUp)+muB(jUp),yMax),muB(jUp));

        tolM     = half*tolM;
      else
        Info(1)  = 'F';                          Fitns       = Fitns + 1;
      end
    end

  % -----------------------------------------------------------------------
  % Check if any temporarily fixed slacks have become feasible with respect
  % to their shifts.  If so, return them to normality.
  % -----------------------------------------------------------------------
    jOkayL = tjfixedcL(cx(tjfixedcL-n)-bL(tjfixedcL)  +muB(tjfixedcL) > 0);
    jOkayU = tjfixedcU(bU(tjfixedcU)  -cx(tjfixedcU-n)+muB(tjfixedcU) > 0);
    jOkay  = my_union(jOkayL,jOkayU);

    if  any(jOkay)
      x (jOkay) = cx(jOkay-n);
      xe(jOkay) = max(min(x(jOkay),bU(jOkay)),bL(jOkay));
      jfixedc   = setdiff(jfixedc,jOkay);
      ifixedc   = jfixedc - n;

      if  any(jOkayL)
        jLo            =  my_union(jLo ,jOkayL  )    ;
        icLo           =  my_union(icLo,jOkayL-n)    ;
        zL (jOkayL)    =  max(eps,abs(piY(jOkayL-n)));
        zeL(jOkayL)    =  max(eps,abs(ye (jOkayL-n)));
        xeL(jOkayL)    =  max(x(jOkayL)-bL(jOkayL)+muB(jOkayL),muB(jOkayL));

      % Check if the freed slacks are part of a range.

        jcLoUOkay      =  intersect(jOkayL,jUp0       );
        jUp            =  my_union (jUp   ,jcLoUOkay  );
        icUp           =  my_union (icUp  ,jcLoUOkay-n);
        icLorU         =  my_union (icLorU,jOkayL-n   );
        jcLorU         =  n + icLorU;             % subset of 1:n+m
        tjfixedcL      =  setdiff  (tjfixedcL,jOkayL)  ;
        zU (jcLoUOkay) =  mZeros   (jcLoUOkay-n)       ;
        zeU(jcLoUOkay) =  mZeros   (jcLoUOkay-n)       ;
        xeU(jcLoUOkay) =  max(bU(jcLoUOkay)-x(jcLoUOkay)+muB(jcLoUOkay),muB(jcLoUOkay));
      end

      if  any(jOkayU)
        jUp            =  my_union(jUp ,jOkayU  )    ;
        icUp           =  my_union(icUp,jOkayU-n)    ;
        zU (jOkayU)    =  max(eps,abs(piY(jOkayU-n)));
        zeU(jOkayU)    =  max(eps,abs(ye (jOkayU-n)));
        xeU(jOkayU)    =  max(bU(jOkayU)-x(jOkayU)+muB(jOkayU),muB(jOkayU));

      % Check if the freed slacks are part of a range.

        jcLoUOkay      =  intersect(jOkayU,jLo0)       ;
        jLo            =  my_union (jLo   ,jcLoUOkay  );
        icLo           =  my_union (icLo  ,jcLoUOkay-n);
        icLorU         =  my_union (icLorU,jOkayU-n   );
        jcLorU         =  n + icLorU;             % subset of 1:n+m
        tjfixedcU      =  setdiff  (tjfixedcU,jOkayU)  ;
        zL (jcLoUOkay) =  mZeros   (jcLoUOkay-n)       ;
        zeL(jcLoUOkay) =  mZeros   (jcLoUOkay-n)       ;
        xeL(jcLoUOkay) =  max(x(jcLoUOkay)-bL(jcLoUOkay)+muB(jcLoUOkay),muB(jcLoUOkay));
      end

      tjfixedc         = my_union(tjfixedcL,tjfixedcU);

      cs               = cx - x(slacks);

      if  ~any(tjfixedc)
        Info(6) = ' ';     % no slacks are outside their shifted bounds.
      end

    % Reassign the barrier multipliers zL, zU and z.

      z(jLo) = zL(jLo);
      z(jUp) = 0;                            z(jUp)  = z(jUp)   -  zU(jUp);
    end

  % -----------------------------------------------------------------------
  % See if any penalized infeasible variables have become feasible with
  % respect to the shifts.  If so, return them to normality.
  % -----------------------------------------------------------------------
    jOkayL   = jLoA( x(jLoA)-bL(jLoA)+muB(jLoA) > 0);
    jOkayU   = jUpA(bU(jUpA)- x(jUpA)+muB(jUpA) > 0);
    jOkay    = my_union(jOkayL,jOkayU);

    if  any(jOkay)
      xe(jOkay) = max(min(x(jOkay),bU(jOkay)),bL(jOkay));
      if  any(jOkayL)
        jLo            =  my_union(jLo ,jOkayL);
        jxLo           =  my_union(jxLo,jOkayL);
        piL(jOkayL)    =  piVL(jOkayL)         ;
        zL (jOkayL)    =  max( vL(jOkayL),eps) ;
        zeL(jOkayL)    =  max(veL(jOkayL),eps) ;
        xeL(jOkayL)    =  max(x(jOkayL)-bL(jOkayL)+muB(jOkayL),muB(jOkayL));
        vL (jOkayL)    =  0;

      % Check if the penalized variables are part of a range.

        jxLoUOkay      =  intersect(jOkayL,jUp0     );
        jUp            =  my_union (jUp   ,jxLoUOkay);
        jxUp           =  my_union (jxUp  ,jxLoUOkay);
        jLoA           =  setdiff  (jLoA  ,jOkayL)   ;
        jxLorU         =  my_union (jxLorU,jOkayL)   ;
        xeU(jxLoUOkay) =  max(bU(jxLoUOkay)-x(jxLoUOkay)+muB(jxLoUOkay),muB(jxLoUOkay));
      end

      if  any(jOkayU)
        jUp            =  my_union(jUp ,jOkayU);
        jxUp           =  my_union(jxUp,jOkayU);
        piU(jOkayU)    =  piVU(jOkayU)         ;
        zU (jOkayU)    =  max( vU(jOkayU),eps) ;
        zeU(jOkayU)    =  max(veU(jOkayU),eps) ;
        xeU(jOkayU)    =  max(bU(jOkayU)-x(jOkayU)+muB(jOkayU),muB(jOkayU));
        vU (jOkayU)    =  0;

      % Check if the penalized variables are part of a range.

        jxLoUOkay      =  intersect(jOkayU,jLo0     );
        jLo            =  my_union (jLo   ,jxLoUOkay);
        jxLo           =  my_union (jxLo  ,jxLoUOkay);
        jUpA           =  setdiff  (jUpA  ,jOkayU)   ;
        jxLorU         =  my_union (jxLorU,jOkayU)   ;
        xeL(jxLoUOkay) =  max(x(jxLoUOkay)-bL(jxLoUOkay)+muB(jxLoUOkay),muB(jxLoUOkay));
      end

      if  ~any(jLoA) && ~any(jUpA)
        Info(5) = ' ';   % no variables are outside their shifted bounds.
      end

      v                = vL   - vU;

    % Reassign the barrier multipliers zL, zU and z.

      z(jLo)           = zL(jLo);
      z(jUp)           = 0;                z(jUp)  = z(jUp)   -  zU(jUp);
    end

  % =====================================================================
  % The shifts may have changed.  Check for variables and slacks not
  % inside their shifted bounds.
  % =====================================================================
    jViolL   = jLo( x(jLo)-bL(jLo)+muB(jLo) <= 0);
    jViolU   = jUp(bU(jUp)- x(jUp)+muB(jUp) <= 0);

    if  any(jViolL) || any(jViolU)
    % ------------
    % Lower bounds
    % ------------
      if  any(jViolL)
        jxViolL        = jViolL(jViolL <= n);
        jcViolL        = jViolL(jViolL >  n);

      % Check for any slacks not inside their shifted lower bound.

        if  any(jcViolL)

          Info(6)      = 'X';

        % Some slacks violate their shifted lower bounds.
        % Fix the offending slacks at their bounds until c(x)
        % lies in the shifted feasible region.

          icViolL      = jcViolL - n;

          muP(icViolL) = muB(jcViolL)./muAfac;
          x  (jcViolL) = bL (jcViolL);               xL(jcViolL) = 0;
          ye (icViolL) = max(-yMax,min(zeL(jcViolL),yMax));
          zL (jcViolL) = 0;                      %   zU(jcViolL) = 0;

        % If the slack lower bounds are part of a range, ignore the upper bound.

          jcViolU      = intersect(jcViolL,jUp);

        % Remove any offending slacks from jLo and jUp. This has the
        % effect of removing them from the barrier term.

          xe(jcViolL)  = max(min(x(jcViolL),bU(jcViolL)),bL(jcViolL));
          jLo          = setdiff(jLo   ,jcViolL);
          jUp          = setdiff(jUp   ,jcViolU);

          icLo         = setdiff(icLo  ,icViolL);
          icUp         = setdiff(icUp  ,jcViolU-n);

          jcLorU       = setdiff(jcLorU,jcViolL); icLorU = setdiff(icLorU,jcViolL-n);
          jcLorU       = setdiff(jcLorU,jcViolU); icLorU = setdiff(icLorU,jcViolU-n);

        % Update tjfixedcL, the slacks temporarily fixed at their lower bound.

          tjfixedcL    = my_union(tjfixedcL, jcViolL);
          jfixedc      = my_union(jfixedc  , jcViolL);
          ifixedc      = jfixedc - n;
          z(jcViolL)   = zL(jcViolL);     % z(jcViolL) correspond to newly-fixed slacks

        end

      % Check for any variables not inside their shifted lower bound.

        if  any(jxViolL)

          Info(5)       = 'x';

        % Some variables violate their shifted lower bounds.  Use a
        % penalty term to drive the offending variables inside their
        % shifted lower bounds.

          muA (jxViolL) = muB(jxViolL)./muAfac;

          veL (jxViolL) = zeL(jxViolL);
          ve  (jxViolL) = veL(jxViolL);
          piVL(jxViolL) = piL(jxViolL);
          vL  (jxViolL) = zL (jxViolL);
          v   (jxViolL) = vL (jxViolL);

        % Move the indices of the offending variables from jLo to jLoA.
        % Variables with indices in jLoA have a penalty term only.

          jLoA          = my_union(jLoA  ,jxViolL);
          jLo           = setdiff (jLo   ,jxViolL);
          jxLo          = setdiff (jxLo  ,jxViolL);
          jUp           = setdiff (jUp   ,jxViolL);
          jxUp          = setdiff (jxUp  ,jxViolL);
          jxLorU        = setdiff (jxLorU,jxViolL);

          z(jxViolL)    = nmZeros (jxViolL);
        end
      end
    % ------------
    % Upper bounds
    % ------------
      if  any(jViolU)
        jcViolU         = jViolU(jViolU >  n);
        jxViolU         = jViolU(jViolU <= n);

      % Check for infeasible slacks.

        if  any(jcViolU)

          Info(6)       = 'X';

        % Some slacks violate their shifted upper bounds.
        % Fix the offending slacks at their bounds until c(x)-s
        % lies in the shifted feasible region.

          icViolU       =  jcViolU - n;

          muP(icViolU)  =  muB(jcViolU)./muAfac;

          x  (jcViolU)  =  bU(jcViolU);             xU(jcViolU) = 0;
          xe (jcViolU)  =  max(min(x(jcViolU),bU(jcViolU)),bL(jcViolU));
          ye (icViolU)  = -min(zeU(jcViolU),yMax);
          xeU(jcViolU)  =  muB(jcViolU);
          zU (jcViolU)  =  0;                     % zL(jcViolU) = 0;

        % If the slack upper bound is part of a range, ignore the lower bound.

          jcViolL       =  intersect(jcViolU,jLo);

        % Remove any offending slacks from jLo and jUp. This has the
        % effect of removing them from the barrier term.

          jUp           =  setdiff(jUp   ,jcViolU);
          jLo           =  setdiff(jLo   ,jcViolL);

          icUp          =  setdiff(icUp  ,icViolU);
          icLo          =  setdiff(icLo  ,jcViolL-n);

          jcLorU        =  setdiff(jcLorU,jcViolU); icLorU = setdiff(icLorU,jcViolU-n);
          jcLorU        =  setdiff(jcLorU,jcViolL); icLorU = setdiff(icLorU,jcViolL-n);

        % Update tjfixedcU, the slacks temporarily fixed at their upper bound.
        % Update ifixedc, jfixedc, the lists of temporarily fixed slacks.

          tjfixedcU     =  my_union(tjfixedcU, jcViolU);
          jfixedc       =  my_union(jfixedc  , jcViolU);
          ifixedc       =  jfixedc - n;
          z(jcViolU)    =  zU(jcViolU);    % z(jcViolU) correspond to newly-fixed slacks
        end

      % Check for infeasible variables at their upper bound.

        if  any(jxViolU)

          Info(5)       =  'x';

        % Some variables violate their shifted upper bounds. Use a
        % penalty term to drive the offending variables towards their
        % upper bounds.

          muA (jxViolU) =  muB(jxViolU)/muAfac;

          veU (jxViolU) =  zeU(jxViolU);
          ve  (jxViolU) = -veU(jxViolU);
          piVU(jxViolU) =  piU(jxViolU);
          vU  (jxViolU) =  zU (jxViolU);
          v   (jxViolU) = -vU (jxViolU);

        % Move the offending variables from jUp to jUpA.
        % Variables in jUpA have a penalty term only.

          jUpA          =  my_union(jUpA  ,jxViolU);
          jLo           =  setdiff (jLo   ,jxViolU);
          jxLo          =  setdiff (jxLo  ,jxViolU);
          jUp           =  setdiff (jUp   ,jxViolU);
          jxUp          =  setdiff (jxUp  ,jxViolU);
          jxLorU        =  setdiff (jxLorU,jxViolU);

          z(jxViolU)    = 0;
        end
      end

    % Reassign the barrier multipliers zL, zU and z.

      z(jLo)            = zL(jLo);
      z(jUp)            = 0;                    z(jUp)  = z(jUp)   -  zU(jUp);

      cs                = cx - x(slacks);

      if  Assert
        if  intersect(jLoA,jUpA)
          outcome = 'Variable outside both bounds';
          break
        end
      end
    end

    if  muPused || muPreduced
      muL     = max(muL/2,muP);
      if  muL <= muLMin
        fprintf('Warning: must decrease muL, but minimum value reached.\n\n')
      end
    end

  % -----------------------------------------------------------------------
  % End of the main loop.
  % -----------------------------------------------------------------------
  end

  if  scaleOption == 1   % Unscale
    x(vars) = x(vars).*S(vars);  y = y./S(cons);
  end

  if  status == 0, status = 6; end
  solution.status  = status;
  solution.x       = x(vars);
  solution.y       = y;
  solution.fx      = fx;
  solution.gx      = gx;
  solution.cx      = cx;
  solution.primInf = primInf;
  solution.itn     = itn;
  solution.nf      = nf;
  solution.mods    = mods;

% -------------------------------------------------------------------------
% Collect statistics about this run.
% -------------------------------------------------------------------------
  if  itn > 0
    Stats(jCvEitns) = 100*CvEitns/itn; %   jCvEitns    = 1;
    Stats(jOitns  ) = 100*Oitns  /itn; %   jOitns      = 2;
    Stats(jFitns  ) = 100*Fitns  /itn; %   jFitns      = 3;
  end
  Stats(jMitns)     = Mitns;           %   jMitns      = 4;
  Stats(jnf   )     = nf;              %   jnf         = 5;
  Stats(jitn  )     = itn;             %   jitn        = 6;
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Auxiliary functions.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function        [fMmuL,fMmuP,gMmuP,F,              ...
                 xL,xU,xLinv,xUinv,                ...
                 piL,piU,piZ,piY,piVL,piVU,piV,    ...
                 zML,zMU,zM,yM,vML,vMU,vM,DBL,DBU] ...
      = pdbMerit(muA,muB,muL,muP,x,y,              ...
                 fx,gx,cx,Jx,cs,                   ...
                 jLo,jUp,icLo,icUp,jLoA,jUpA,      ...
                 z,zL,zU,v,vL,vU,                  ...
                 xeL,xeU,zeL,zeU,ye,veL,veU,Prob)

% pdbMerit computes the value and gradient of the merit function.

  m            = Prob.m;
  n            = Prob.n;
  bL           = Prob.bL;
  bU           = Prob.bU;
  vars         = Prob.vars;
  slacks       = Prob.slacks;

  nZeros       = zeros(n,1);                       nmZeros    = zeros(n+m,1);

  ifreec       = union(icLo,icUp);                 ifixedc    = setdiff(1:m,ifreec);
                                                   jfixedc    = n + ifixedc;

  xL           = nmZeros;                          xU         = nmZeros;
  xL(jLo)      = x(jLo) - bL(jLo) + muB(jLo);      xU(jUp)    = bU(jUp) - x(jUp) + muB(jUp);

  xLinv        = nmZeros;                          xUinv      = nmZeros;
  xLinv(jLo)   = 1./xL(jLo);                       xUinv(jUp) = 1./xU(jUp);

  zLinv        = nmZeros;                          zUinv      = nmZeros;
  zLinv(jLo)   = 1./(zL(jLo) + muB(jLo));          zUinv(jUp) = 1./(zU(jUp) + muB(jUp));

  DBL          = nmZeros;                          DBU        = nmZeros;
  DBL(jLo)     = zLinv(jLo).*xL(jLo);              DBU(jUp)   = zUinv(jUp).*xU(jUp);

  piY          = ye  - cs./muP;

  piL          = nmZeros;                          piU        = nmZeros;
  piL(jLo)     = muB(jLo).*(xLinv(jLo).*(zeL(jLo)+xeL(jLo)))-muB(jLo);
  piU(jUp)     = muB(jUp).*(xUinv(jUp).*(zeU(jUp)+xeU(jUp)))-muB(jUp);

  piZ          = nmZeros;
  piZ(jLo)     = piL(jLo);                         piZ(jUp)   = piL(jUp)  -  piU(jUp);
  piZ(jfixedc) = piY(ifixedc);

  piVL         = nZeros;                           piVU       =  nZeros;
  piVL(jLoA)   = veL(jLoA)  - ( x(jLoA) - bL(jLoA))./muA(jLoA);
  piVU(jUpA)   = veU(jUpA)  - (bU(jUpA) -  x(jUpA))./muA(jUpA);
  piV          = piVL       -  piVU;

  vML          = nZeros;                           vMU        =  nZeros;
  vML(jLoA)    = piVL(jLoA) + (piVL(jLoA) - vL(jLoA));
  vMU(jUpA)    = piVU(jUpA) + (piVU(jUpA) - vU(jUpA));
  vM           = vML        - vMU;

  yM           = piY        + (piY        -  y      );

  zML          = nmZeros;                          zMU        = nmZeros;
  zML(jLo)     = piL(jLo)   + (piL(jLo) - zL(jLo));
  zMU(jUp)     = piU(jUp)   + (piU(jUp) - zU(jUp));

  zM           = nmZeros;
  zM(jLo)      = zML(jLo);                         zM(jUp)    = zML(jUp)  - zMU(jUp);
  zM(jfixedc)  = yM(ifixedc);

% -------------------------------------------------------------------------
% Compute the merit function and its gradient;
% -------------------------------------------------------------------------
  muPTerm =    sum(                     cs.^2./muP)                ...
            +  sum( (cs  + (y - ye).*muP ).^2./muP)                ...
            +  sum( ( x(jLoA) -  bL(jLoA)).^2./muA(jLoA))          ...
            +  sum( (bU(jUpA) -   x(jUpA)).^2./muA(jUpA))          ...
            +  sum((( x(jLoA) -  bL(jLoA))./   muA(jLoA)           ...
                  +  vL(jLoA) - veL(jLoA)).^2.*muA(jLoA))          ...
            +  sum(((bU(jUpA) -   x(jUpA))./   muA(jUpA)           ...
                  +  vU(jUpA) - veU(jUpA)).^2.*muA(jUpA));
  muLTerm =    sum(                     cs.^2./muL)                ...
            +  sum( (cs  + (y - ye).*muL ).^2./muL)                ...
            +  sum( ( x(jLoA) -  bL(jLoA)).^2./muA(jLoA))          ...
            +  sum( (bU(jUpA) -   x(jUpA)).^2./muA(jUpA))          ...
            +  sum((( x(jLoA) -  bL(jLoA))./   muA(jLoA)           ...
                  +  vL(jLoA) - veL(jLoA)).^2.*muA(jLoA))          ...
            +  sum(((bU(jUpA) -   x(jUpA))./   muA(jUpA)           ...
                  +  vU(jUpA) - veU(jUpA)).^2.*muA(jUpA));
  muBTerm = -  sum(muB(jLo).*(zeL(jLo)+xeL(jLo)).*log((zL(jLo)+muB(jLo)).*(xL(jLo).^2))) ...
            -  sum(muB(jUp).*(zeU(jUp)+xeU(jUp)).*log((zU(jUp)+muB(jUp)).*(xU(jUp).^2))) ...
            +  sum( zL(jLo).*xL(jLo)) + 2*sum(muB(jLo).*(x(jLo) - bL(jLo)))              ...
            +  sum( zU(jUp).*xU(jUp)) + 2*sum(muB(jUp).*(bU(jUp) - x(jUp)));
  fLag    = fx - cs'*ye                                            ...
            -  sum( ( x(jLoA) -  bL(jLoA)).*veL(jLoA))             ...
            -  sum( (bU(jUpA) -   x(jUpA)).*veU(jUpA));
  fMmuP   = fLag + muPTerm/2 + muBTerm;
  fMmuL   = fLag + muLTerm/2 + muBTerm;

  gMmuP   = [     gx - Jx'*yM - vM  - zM(vars)   ;
                           yM       - zM(slacks) ;
                     muP.*(y        - piY)       ;
               muA(jLoA).*(vL(jLoA) - piVL(jLoA));
               muA(jUpA).*(vU(jUpA) - piVU(jUpA));
                DBL(jLo).*(zL(jLo)  - piL(jLo))  ;
                DBU(jUp).*(zU(jUp)  - piU(jUp)) ];
  F       = [     gx - Jx'*y  - v   - z(vars)    ;
                           y        - z(slacks)  ;
                     muP.*(y        - piY)       ;
               muA(jLoA).*(vL(jLoA) - piVL(jLoA));
               muA(jUpA).*(vU(jUpA) - piVU(jUpA));
                 xL(jLo).*(zL(jLo)  - piL(jLo))  ;
                 xU(jUp).*(zU(jUp)  - piU(jUp)) ];
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [step,fMmuL,fMmuP,gMmuP,                   ...
          x,y,vL,vU,z,zL,zU,                          ...
          fx,gx,cx,Jx,cs,F,                         ...
          xL,xU,xLinv,xUinv,                        ...
          piL,piU,piZ,piY,piVL,piVU,piV,            ...
          zML,zMU,zM,yM,vML,vMU,vM,DBL,DBU,         ...
          jfM,iExit,muPused,maxNormF]               ...
      = pdbqArmijoLS(muA,muB,muL,muP,maxNormF,      ...
                     step0,fMmuL0,fMmuP0,gMmuP0,F0, ...
                     fx0,gx0,cx0,Jx0,cs0,           ...
                     jLo,jUp,icLo,icUp,jLoA,jUpA,   ...
                     x0,y0,vL0,vU0,z0,zL0,zU0,      ...
                     dx,dy,dvL,dvU,dzL,dzU,dzX,     ...
                     scaleOption,S,Sobj,            ...
                     xeL,xeU,ye,veL,veU,zeL,zeU,    ...
                     prob,Prob,parms)
%==========================================================================
%  pdbqArmijoLS finds a (flexible) quasi-Armijo step along the search
%  direction dw for the all-shifted primal-dual penalty-barrier method,
%  such that the function fM is sufficiently reduced, i.e.,
%       fM(w0 + step*dw) <  fM(w0),
%  where w0 = [x0,y0,vL0,vU0,zL0,zU0] is the base point for the search,
%  and   dw = [dx,dy,dvL,dvU,dzL,dzU] is the search direction.
%
%  On entry,
%  step    is the initial estimate of the step.
%  fMmuL0  is the merit function with penalty parameter muL.
%  fMmuP0  is the merit function
%  gMmuP0  is the gradient of the merit function and gradient.
%
%  On exit,
%  step    is the final step length. The final point is w0 + step*dw,
%          with w = [dx,dy,dvL,dvU,dzL,dzU].
%  fMmuL   is the merit function with penalty parameter at w1
%  fMmuP   is the function value at w1.
%  gMmuP   is the gradient  at w1.
%
%  iExit    Result
%  -----    ------
%    0      Repeat the search with smaller stepMax.
%    1      The search is successful.
%    2      The search is successful with the initial step.
%    3      A better point was found but too many functions
%           were needed (no sufficient decrease).
%    7      Too many function calls.
%    8      No descent direction
%==========================================================================
  m           = Prob.m;
  n           = Prob.n;
  bL          = Prob.bL;
  bU          = Prob.bU;
  vars        = Prob.vars;
  slacks      = Prob.slacks;
  jfixedx     = Prob.jfixedx;

  armijoTol   = parms.etaA         ;
  etaF        = parms.etaF         ;  % Line search sufficient decrease for norm(F)
  gammaC      = parms.gammaC       ;  % contraction factor
  boundaryTol = parms.boundaryTol  ;  % fraction-to-the-boundary value
  jMmax       = parms.jMmax        ;  % max functions per line search
  maxMerit    = parms.maxMerit     ;  % Upper bound for merit function in line search

  ifreec      = union(icLo,icUp);       ifixedc    = setdiff(1:m,ifreec);
                                        jfixedc    = n + ifixedc;

  Undefined   = false;
  muPused     = false;

  nZeros      = zeros(n,1);             nmZeros    = zeros(n+m,1);
  zL          = nmZeros;                zU         =  nmZeros;
  z           = nmZeros;

  vL          = nZeros;                 vU         =  nZeros;

  bLbL        = bL;    bLbL(jLo)   = bL(jLo) - muB(jLo);
  bUbU        = bU;    bUbU(jUp)   = bU(jUp) + muB(jUp);

  bLbL(jLo)   = min( x0(jLo)-boundaryTol*(  x0(jLo)-bLbL(jLo)), bL(jLo));
  bUbU(jUp)   = max( x0(jUp)+boundaryTol*(bUbU(jUp)-  x0(jUp)), bU(jUp));

  bLzL        = min(zL0(jLo)-boundaryTol*(zL0(jLo)+muB(jLo)), 0);
  bLzU        = min(zU0(jUp)-boundaryTol*(zU0(jUp)+muB(jUp)), 0);

  fMmuP       = fMmuP0;
  step        = step0;
  normF0      = norm(F0,Inf);
  gMw0        = gMmuP0'*[dx; dy; dvL(jLoA); dvU(jUpA); dzL(jLo); dzU(jUp)];

  jfM    =  1;
  while jfM <= jMmax
    x        = x0        + step*dx;
    y        = y0        + step*dy;
    vL(jLoA) = vL0(jLoA) + step*dvL(jLoA);
    vU(jUpA) = vU0(jUpA) + step*dvU(jUpA);
    zL(jLo)  = zL0(jLo)  + step*dzL(jLo);
    zU(jUp)  = zU0(jUp)  + step*dzU(jUp);

  % Project  x, zL and zU  back into the feasible region.

    x        = max(x, bLbL);        x        = min(x, bUbU);
    zL(jLo)  = max(zL(jLo), bLzL);  zU(jUp)  = max(zU(jUp), bLzU);

    if scaleOption == 1
      x(vars) = x(vars).*S(vars);
    end

    v          = vL          - vU;

    z          = zL          - zU;
    z(jfixedx) = z0(jfixedx) + step*dzX(jfixedx);
    z(jfixedc) = z0(jfixedc) + step*dzX(jfixedc);

    [fx,gx]    = prob.obj(x(vars));
    if  m > 0
      [cx,Jx]  = prob.cons(x(vars));  cs  = cx - x(slacks);
    else
      cx       = zeros(m,1);          cs  = cx;             Jx = zeros(m,n);
    end

    if (sum(isnan([fx; gx])) + sum(isinf([fx; gx]))) > 0
      Undefined = true;
    else

      if scaleOption == 1
        fx      = fx/Sobj;
        gx      = gx.*S(vars)/Sobj;
        Jx      = diag(1./S(slacks))*Jx*diag(S(vars));
        cx      = cx./S(slacks);
        x(vars) = x(vars)./S(vars);
      end

      [fMmuL,fMmuP,gMmuP,F,                            ...
                     xL,xU,xLinv,xUinv,                ...
                     piL,piU,piZ,piY,piVL,piVU,piV,    ...
                     zML,zMU,zM,yM,vML,vMU,vM,DBL,DBU] ...
          = pdbMerit(muA,muB,muL,muP,x,y,              ...
                     fx,gx,cx,Jx,cs,                   ...
                     jLo,jUp,icLo,icUp,jLoA,jUpA,      ...
                     z,zL,zU,v,vL,vU,                  ...
                     xeL,xeU,zeL,zeU,ye,veL,veU,Prob);

    % Flexible quasi-Armijo

      normF      = norm(F    ,Inf);
      normgM     = norm(gMmuP,Inf);

      if  normF <= etaF*min(normF0,maxNormF) && fMmuL < max(fMmuL0,maxMerit) ...
                                             && fMmuP < max(fMmuP0,maxMerit)
        maxNormF = etaF*maxNormF;
        break;
      elseif fMmuL <= fMmuL0 + armijoTol*step*gMw0
        break;
      end

      if     fMmuP <= fMmuP0 + armijoTol*step*gMw0
        muPused  = true;
        break;
      end
    end

    jfM   =  jfM + 1;
    if jfM <=  jMmax
      step  =  gammaC*step;  %     Reduce the step.
    end
  end
% -----------------------------------------------------------------------

  if Undefined
    fprintf('WARNING : Undefined problem functions.\n')
  end

  if  jfM <= jMmax
    if  step == step0
      inform = 2;
    else
      inform = 1;
    end
  else
    if fMmuP < fMmuP0
      inform = 3;
    else
      fx     = fx0;  gx = gx0;
      if  m > 0
        cx   = cx0;      cs    = cs0;    Jx    = Jx0;
      end
      [fMmuL,fMmuP,gMmuP,F,                               ...
                        xL,xU,xLinv,xUinv,                ...
                        piL,piU,piZ,piY,piVL,piVU,piV,    ...
                        zML,zMU,zM,yM,vML,vMU,vM,DBL,DBU] ...
             = pdbMerit(muA,muB,muL,muP,x,y,              ...
                        fx,gx,cx,Jx,cs,                   ...
                        jLo,jUp,icLo,icUp,jLoA,jUpA,      ...
                        z,zL,zU,v,vL,vU,                  ...
                        xeL,xeU,zeL,zeU,ye,veL,veU,Prob);
      x      = x0;
      y      = y0;
      inform = 7;
    end
  end
  iExit = inform;
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [sReset] = pdbslackReset(n,m,x,bL,bU,shat,jcLorU,pdb_parms)
%        [sReset] = pdbslackReset(n,m,x,bL,bU,shat,jcLorU,pdb_parms)
%
% Minimize the merit function with respect to the free slacks.

  parms    = feval(pdb_parms);

  infinity = parms.infinity     ;  % user defined infinity

  sReset   = zeros(m,1);

  for  j = jcLorU'
    i    = j - n;
    s0   = shat(i);
    si   = x (j);
    bLi  = bL(j);
    bUi  = bU(j);

    if      bLi <  -infinity  &&  bUi >  infinity

    % Fixed slack

      sReset(i) = si;

    elseif  bLi >= -infinity  &&  bUi >  infinity

    % Lower bound only

      sReset(i) = max([s0 si]);

    elseif  bLi <  -infinity  &&  bUi <= infinity

    % Upper bound only

      sReset(i) = min([s0 si]);

    else

    % Lower and upper bounds

      sReset(i) = si;
    end
  end
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function pdb_print(parms,outfile)

  solver        = parms.solver       ;
  muB0          = parms.muB          ;  % barrier parameter for the merit function
  muP0          = parms.muP          ;  % penalty parameter
  muL0          = parms.muL          ;  % penalty parameter for the merit function
  muLMin        = parms.muLMin       ;  % minimum value of muL
  maxIterations = parms.maxIterations;  % maximum iterates allowed
  minPosEig     = parms.minPosEig    ;  % smallest allowed pos. eigenvalues of B
  tolPrimFeas   = parms.tolPrimFeas  ;  % primal feasibility tolerance
  tolDualFeas   = parms.tolDualFeas  ;  % dual   feasibility tolerance
  tolStny       = parms.tolStat      ;  % stationarity       tolerance
  tolIsp        = parms.tolIsp       ;  % Infeasible stationary point tolerance
  jMmax         = parms.jMmax        ;  % maximum number of backtracks
  armijoTol     = parms.etaA         ;  % sufficient decrease factor in backtracking
  gammaC        = parms.gammaC       ;  % contraction factor in backtracking
  boundaryTol   = parms.boundaryTol  ;  % fraction-to-the-boundary value
  dxMax         = parms.dxMax        ;  % parameter used to define max step length
  tolM          = parms.tolM         ;  % unconst. opt tolerance for the merit function
  lines         = parms.lines        ;  % lines between printing header
  infinity      = parms.infinity     ;  % user-defined +infinity
  Assert        = parms.Assert       ;  % (0/1) check consistency
  yMax          = parms.yMax         ;  % Largest ye for M-iterate
  unBoundedf    = parms.unBoundedf   ;  % unbounded objective value
  debugItn      = parms.debugItn     ;  % iteration for setting dbstop
  scaleOption   = parms.scaleOption  ;  % (0/1) scaling off/on
  printSol      = parms.printSol     ;  % to print solution to screen
  printLevel    = parms.printLevel   ;  % ==1 for summary line.

  str0 = sprintf('Parameters\n==========');
  fprintf('%s\n', str0);    fprintf(outfile, '%s\n', str0);
% --------------------------------------------------------------------
  str  = sprintf('Solver.................   %8s     ',        solver);
  fprintf('%s\n', str);    fprintf(outfile, '%s\n', str);
% --------------------------------------------------------------------
  str1 = sprintf('Initial Barrier........  %9.2e    ',          muB0);
  %               12345678901234567890123
  str2 = sprintf('Initial Penalty........  %9.2e    ',          muP0);
  str3 = sprintf('Initial Flex Search Pen  %9.2e    ',          muL0);
  str4 = sprintf('Minimum Flex Search Pen  %9.2e    ',        muLMin);
  str  = [str1 str2 str3 str4];
  fprintf('%s\n', str);    fprintf(outfile, '%s\n', str);
% --------------------------------------------------------------------
  str1 = sprintf('Primal feasibility.....  %9.2e    ',   tolPrimFeas);
  str2 = sprintf('Dual feasibility.......  %9.2e    ',   tolDualFeas);
  str3 = sprintf('Stationarity (feas)....  %9.2e    ',       tolStny);
  str4 = sprintf('Stationarity (infeas)..  %9.2e    ',        tolIsp);
  str  = [str1 str2 str3 str4];
  fprintf('%s\n', str);    fprintf(outfile, '%s\n', str);
% --------------------------------------------------------------------
  str1 = sprintf('Armijo tolerance.......  %9.2e    ',     armijoTol);
  str2 = sprintf('Armijo contraction.....  %9.2e    ',        gammaC);
  str3 = sprintf('Backtrack limit........  %9g    '  ,         jMmax);
  str4 = sprintf('Maximum step...........  %9.2e    ',         dxMax);
  str  = [str1 str2 str3 str4];
  fprintf('%s\n', str);    fprintf(outfile, '%s\n', str);
% --------------------------------------------------------------------
  str1 = sprintf('Max iterations.........  %9g    '  , maxIterations);
  str2 = sprintf('Max multiplier est.....  %9.2e    ',          yMax);
  str3 = sprintf('Unbounded objective....  %9.2e    ',    unBoundedf);
  str4 = sprintf('User plus infinity.....  %9.2e    ',      infinity);
  str  = [str1 str2 str3 str4];
  fprintf('%s\n', str);    fprintf(outfile, '%s\n', str);
% --------------------------------------------------------------------
  str1 = sprintf('Fraction to boundary...  %9.2e    ',   boundaryTol);
  str2 = sprintf('Merit minimization tol.  %9.2e    ',          tolM);
  str3 = sprintf('Min positive eig.......  %9.2e    ',     minPosEig);
  str4 = sprintf('Scaling off/on (0/1)...  %9g    '  ,   scaleOption);
  str  = [str1 str2 str3 str4];
  fprintf('%s\n', str);    fprintf(outfile, '%s\n', str);
% --------------------------------------------------------------------
  str1 = sprintf('Print level............  %9g    '  ,    printLevel);
  str2 = sprintf('Lines between headers..  %9g    '  ,         lines);
  str3 = sprintf('Print solution.........  %9g    '  ,      printSol);
  str4 = sprintf('Assert  off/on (0/1)...  %9g    '  ,        Assert);
  str  = [str1 str2 str3 str4];
  fprintf('%s\n', str);    fprintf(outfile, '%s\n', str);
% --------------------------------------------------------------------
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [zL,zU] = infeasMul(muB,muBfac,gx,y,Jx,jLo,jUp,v,zL,zU,Prob)

  m    = Prob.m;
  n    = Prob.n;

  g    = [ gx
           zeros(m,1) ];
  if  m > 0
    J  = [ Jx  -diag(ones(m,1)) ];
  else
    J  = zeros(m,n+m);
  end

  v0        = [v;  zeros(m,1)];

% Check the multipliers for the lower bounds.

  [jInf,~]  = find(zL(jLo) + muB(jLo) <= 0);
  infLo     = jLo(jInf);
  zL(infLo) = zL(infLo)/muBfac;
  zL(infLo) = max(zL(infLo),g(infLo)-J(:,infLo)'*y-v0(infLo)+zU(infLo));

% Check the multipliers for the upper bounds.

  [jInf,~]  = find(zU(jUp) + muB(jUp) <= 0);
  infUp     = jUp(jInf);
  zU(infUp) = zU(infUp)/muBfac;
  zU(infUp) = max(zU(infUp),zL(infUp)-g(infUp)+J(:,infUp)'*y+v0(infUp));
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [primInf,dualInf]=PrimalDualInf(gx,Jx,x,cx,y,zScale,Prob)
%        [primInf,dualInf]=PrimalDualInf(gx,Jx,x,cx,y,zScale,Prob)
% Compute the primal and dual infeasibilities.

  bL      = Prob.bL;
  bU      = Prob.bU;
  slacks  = Prob.slacks;

  z       = [ gx - Jx'*y ; y ]/zScale;
  xL      = (x  - bL)./(1 + abs(bL)); xU     = (bU -  x)./(1 + abs(bU));
  violbL  = min(0,xL);                violbU = min(0,xU);
  viol    = norm(cx-x(slacks),Inf)/max([ 1, norm(x(slacks),Inf) ] );
  zL      = max(0, z.*min(xL,1));     zU     = max(0,-z.*min(xU,1));

  primInf = norm([violbL; violbU; viol],Inf);
  dualInf = norm([zL    ;     zU],Inf);
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [maxComp,maxViol,maxViolBnds,maxStny] ...
          = pdbOptCheck(gx,Jx,cs,x,y,v,z,zL,zU,zScale,Prob)

  bL          = Prob.bL;
  bU          = Prob.bU;
  vars        = Prob.vars;
  slacks      = Prob.slacks;

  xL          = (x  - bL)./(1 + abs(bL));  xU    = (bU -  x)./(1 + abs(bU));
  violL       = min(0,xL);                 violU = min(0,xU);
  compL       = max(zL.*min(abs(xL),1));   compU = max(zU.*min(abs(xU),1));
% [norm(min([ zL xL zU xU ]),Inf), norm(zL.*xL,Inf), norm(zU.*xU,Inf)];
  viol        = norm(cs,Inf)/max([ 1, norm(x(slacks),Inf) ] );

  maxStny     = norm(gx-Jx'*y-v-z(vars),   Inf)/zScale;
  maxComp     = norm([compL; compU],       Inf)/zScale;
  maxViolBnds = norm([violL; violU],       Inf);
  maxViol     = norm([violL; violU; viol], Inf);
  if  maxComp < 1.0e-99, maxComp = 0; end
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [maxComp]=pdbIspCheck(Jx,cs,x,cScale,bL,bU)
%        [maxComp]=pdbIspCheck(Jx,cs,x,cScale,bL,bU)

  xL      = (x  - bL)./(1 + abs(bL));  xU    = (bU -  x)./(1 + abs(bU));
  g       = [ Jx'*cs ; -cs ]/cScale;
  zL      = max(0, g);                 zU    = max(0,-g);
  compL   = max(zL.*min(abs(xL),1));   compU = max(zU.*min(abs(xU),1));

  maxComp = norm([compL; compU], Inf);
  if  maxComp < 1.0e-99, maxComp = 0; end
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [dualInf]=IspDualInf(Jx,cs,x,cScale,bL,bU)
%        [dualInf]=IspDualInf(Jx,cs,x,cScale,bL,bU)

% Compute the dual infeasibility for the locally infeasible problem.

  xL      = (x  - bL)./(1 + abs(bL));  xU    = (bU -  x)./(1 + abs(bU));
  z       = [ Jx'*cs ; -cs ]/cScale;
  zL      = max(0, z.*min(xL,1));      zU    = max(0,-z.*min(xU,1));

  dualInf = norm([ zL ; zU ],Inf);
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [array]=my_union(array1,array2)
%        [array]=my_union(array1,array2)

%        Gives the union of two scalars as a column vector

  [r1,~] = size(array1);
  [r2,~] = size(array2);
  array  = union(array1,array2);

  if r1 == 1 && r2 == 1
    array = array';
  end
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [e,p] = vError(f1,f2,getPrecision)
%        [e,p] = vError(f1,f2,getPrecision)
%
%  Estimates the error  e, and precision  p  of f1 as an
%  approximation to f2 (and vice-versa).
%
%  e = d(f1,f2), where d(f1,f2) is the distance function
%      d(f1,f2) =  min( |f1-f2|, |f1-f2|/(|f1|+|f2|) )
%      with d(0,0) = 0.
%
%  p = -log10(e),  and estimates the number of matching
%      decimal digits in f1 and f1. If |f1|+|f2| < 1,
%      then  p  estimates the matching decimal digits
%      assuming that the leading zeros of the unnormalized
%      values of f1 and f2 are correct digits.

  e    = zeros(size(f1));
  j    = f1 ~= 0 | f2 ~= 0;
  e(j) = min( abs(f1(j)-f2(j)), abs(f1(j)-f2(j))./(abs(f1(j)) + abs(f2(j))) );

  if  nargin == 3
    p    = inf(size(f1));
    p(j) = -log10(e(j));
  end
end
