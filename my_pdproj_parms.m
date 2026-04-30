function [parms] = my_pdproj_parms()
% ------------------------------------------------------------------------
% Default hyperparameter list for pdProj.
%
% Authors : Philip E. Gill and Minxin Zhang
% Purpose : Sets the control parameters for the projected-search
%           interior-point solver pdProj.m
% ------------------------------------------------------------------------
  parms.solver        = 'pdProj';

  parms.muB           = 1.0e-4  ;  % initial barrier parameter for the merit function
  parms.muP           = 1.0e-4  ;  % initial dual regularization parameter
  parms.muL           = 1.0     ;  % initial penalty parameter for the merit function

  parms.printLevel    = 1       ;  % ==1 for summary line.
  parms.muLMin        = 1.0e-14 ;  % minimum value of muP

  parms.maxIterations = 500     ;  % maximum iterates allowed
  parms.tiny          = 1.0e-12 ;  % tiny number
  parms.minPosEig     = 1.0e-8  ;  % smallest allowed positive eigenvalue of B
  parms.tolPrimFeas   = 1.0e-4  ;  % primal feasibility tolerance
  parms.tolDualFeas   = 1.0e-4  ;  % dual   feasibility tolerance
  parms.tolStat       = 1.0e-4  ;  % stationary         tolerance
  parms.tolIsp        = 1.0e-5  ;  % Infeasible stationary point tolerance
  parms.jMmax         = 40      ;  % maximum number of backtracks
  parms.etaA          = 1.0e-3  ;  % Armijo sufficient decrease factor
  parms.gammaC        = 0.5     ;  % contraction factor in backtracking
  parms.etaF          = 0.9;  % Line search sufficient decrease for norm(F)

  parms.TRrelTol      = 1.0e-6  ;  % trust-region relative tolerance.
  parms.TRabsTol      = 0.0     ;  % trust-region absolute tolerance.
  parms.TRetaA        = 1.0e-1  ;  % trust-region sufficient  decrease factor
  parms.TRetaE        = 0.25    ;  % trust-region substantial decrease factor
  parms.TRexpFac      = 2.0     ;  % trust-region radius expand factor
  parms.TRconFac      = 0.5     ;  % trust-region radius contract factor
  parms.TRlog         = 0       ;  % trust-region log (0/1) off/on
  parms.TRitnMax      = 100     ;  % trust-region iteration limit
  parms.Uncondensed   = 0       ;

  parms.boundaryTol   = 0.9     ;  % fraction-to-the-boundary

  parms.dxMax         = 1.0e+18 ;  % parameter used to define max step length

  parms.tolM          = 1.0e-1  ;  % unconstrained opt tolerance for the merit function
  parms.lines         = 25      ;  % lines between printing header
  parms.printSol      = 1       ;  % to print solution to screen
  parms.infinity      = 1.0e+15 ;  % user-defined +infinity
  parms.Assert        = 0       ;  % (0/1) check consistency
  parms.yMax          = 1.0e+6  ;  % Largest yE for M-iterate
  parms.maxNormF      = 1.0e+8  ;  % Upper bound for normF in line search
  parms.maxMerit      = 1.0e+12 ;  % Upper bound for merit function in line search
  parms.show_warnings = false   ;  % Logical variable determines whether warnings are printed

  parms.unBoundedf    =-1.0e+15 ;  % definition of an unbounded objective value

  parms.debugItn      = 35      ;  % iteration for setting dbstop

  parms.scaleOption   = 0       ;  % 0/1 scaling off/on
  parms.Uncondensed   = 1       ;  % 0/1 solve the uncondensed KKT equations

end
