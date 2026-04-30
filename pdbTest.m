function [x,y,status,Stats] = pdbTest(prob,pdb_parms)
%        [x,y,status,Stats] = pdbTest(prob,pdb_parms).
%
%        This version dated 20-Jul-2023.
%        Philip E. Gill, University of California, San Diego.

  global   fx

  format compact

  t0        = cputime;  % start recording cpu time

  % -------------------------------
  % Assign local control parameters
  % -------------------------------
  parms     = feval(pdb_parms);

  solver    = parms.solver;  % Solver

  % -----------------------
  % End: control parameters
  % -----------------------

  % Initialize run statistics

  Stats       = zeros(1,6);
  jCvEitns    = 1;  CvEitns = 0;
  jOitns      = 2;  Oitns   = 0;
  jFitns      = 3;  Fitns   = 0;
  jMitns      = 4;  Mitns   = 0;
  jnf         = 5;  nf      = 0;  % Number of function evaluations
  jitn        = 6;  itn     = 0;  % Iteration counter
  jTime       = 7;

  outfile   = fopen(strcat(prob.name,'.out'),'w');
  % fid       = fopen(strcat('Summary_', solver, '.csv'), 'a+');

  if strcmp(solver,'pdProj')

  % pdb with primal-dual shifts and quasi-Wolfe line search

    [solution,outcome,Stats] = pdProj(prob,outfile,pdb_parms);
    fx      = solution.fx;
    primInf = solution.primInf;
    x       = solution.x;
    y       = solution.y;
    status  = solution.status;
  end

  %--------------------------------------------------------------------------
  %  Collect statistics.
  %    status = 1;  Optimal
  %    status = 2;  Inf Stationary point
  %    status = 3;  Unbounded
  %    status = 4;  Iteration limit
  %    status = 5;  Error
  %--------------------------------------------------------------------------
  Time        = cputime - t0;
  Stats       = [ Stats Time];
  iStat       = fix(Stats);
  if Stats(jCvEitns) == 0
    NonConvex = '';
  else
    NonConvex = 'Nonconvex';
  end

  nf          = iStat(jnf);
  itn         = iStat(jitn);
  if status > 3
    nf   = NaN;  Time = NaN; itn = NaN;
  end

  strP1      = sprintf('\n PROBLEM  %-10s--- %-24s:',      prob.name,outcome);
  if status < 5
    strP2    = sprintf(' Num f = %4g,  Itns   = %4g,',     nf,itn);
  else
    strP2    = sprintf(' Num f = %4g,  Itns   = %4g,',    NaN, NaN);
  end
  strP3      = sprintf(' m      = %4g,  n      = %4g, f = %15.7e,  primInf = %15.7e, Time = %5.1f ', ...
                       prob.m,prob.n,fx,primInf,Time);

  strS1      = sprintf('\n STATS    %-10s    %-24s:',       prob.name,NonConvex);
  strS2      = sprintf(' nEmod = %3i%%%%,  O-itns = %3i%%%%,', iStat(jCvEitns),iStat(jOitns));
  strS3      = sprintf(' F-itns = %3i%%%%,  M-itns =  %3i\n\n',iStat(jFitns),Stats(jMitns));
  strP       = [ strP1 strP2 strP3 ];   strS       = [ strS1 strS2 strS3 ];
  fprintf(strP);   fprintf(outfile, strP);
  fprintf(strS);   fprintf(outfile, strS);

  % fprintf(fid, '%s,%s,%s,%d,%d,%d,%f\n', strtrim(prob.name), solver, outcome, n, itn, nf, Time);

  fclose(outfile);
  % fclose(fid);
end
