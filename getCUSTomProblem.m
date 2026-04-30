function  prob  = getCUSTomProblem()
%        [prob] = getCUSTomProblem
  prob          = cutest_setup;
  prob          = struct;

% prob = struct('n',{0},'m',{0},'bl',{0},'bu',{0},'v',{0},'cl',{0},'cu',{0},'equatn',{0},'linear',{0},'obj',{0},'cons',{0},'hess',{0},'x',0,'name',0);

  prob.n      = 1;
  prob.m      = 1;
  prob.bl     = [-Inf];
  prob.bu     = [0];
  prob.cl     = [-Inf];
  prob.cu     = [0];
  prob.equatn = [0];
  prob.linear = [0]
  prob.name   = 'BURKEHAN';
  prob.x      = [ 10 ];
  prob.v      = [ 0  ];

  prob.obj = (@(x)(objective(x)));
  prob.cons = (@(x)(constraint(x)));
  prob.hess = (@(x,y)(hessian(x,y)));
end
%optimal x = 0, y = 0, z = [2;2;0]
function [f,g] = objective(x)
  f      = x(1);
  g(1)   =  1;
end

function [c,J] = constraint(x)
  c(1)   = x(1)^2 + 1;
  J(1,1) = 2*x(1);
end

function [H] = hessian(x,y)
  y      = -y;
  H(1,1) = 2*y(1);
end
