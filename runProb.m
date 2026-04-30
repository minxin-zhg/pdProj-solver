function [x,y,status,Stats] =  runProb(prob_name,pdproj_parms)
%        [x,y,status,Stats] =  runProb(prob_name,pdproj_parms)
% If no arguments are provided it will run the custom problem defined
% in getCUSTomProblem.m using whatever solver is already set in the parms file:
% >> runProb();
%
% If one argument is provided it is assumed to be a cutest problem name, and the
% given cutest problem will be run using whatever solver is already set in the
% bfgs_parms file:
% >> runProb(‘AKIVA’);
%

  if nargin > 0 % Solver and problem provided, run solver on CUTEst problem
    [prob] = getCUTEstProblem(prob_name);
    if nargin == 1
      [x,y,status,Stats] = pdbTest(prob,'my_pdproj_parms');
    else
      [x,y,status,Stats] = pdbTest(prob,pdproj_parms);
    end
  else
    [prob] = getCUSTomProblem();
    [x,y,status,Stats] = pdbTest(prob,'my_pdproj_parms');
  end
  cutest_terminate
end
