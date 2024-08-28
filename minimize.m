function var = minimize(fun, x0,lb,ub,varargin)
% Release: 1.0.0, August 2024
% Author: Niccolo' Mazzatenta (contact at nmazzatenta@icloud.com)
%
%MINIMIZE non-linear function s.t. box constraints using gradient-based methods.
% x = minimize(fun, x0, lb, ub, varargin)
%
%                     min_{x} L(x)
%                     s.t. lb<x<ub
%
% where L(x) = ||fun(x)||^2 if fun(x) returns a vector, otherwise L(x) =
% fun(x)
%
% Box constraints are handled heuristically. Variables violating the 
% constraints are relocated on the bounds, and gradient componentes 
% pointing outwards are set to zero.
%
% For unbounded problems, set lb=[], ub=[]
%
% Name-Value pair options
%
% 'GradTol'     :   sets the tolerance for the norm-2 of the gradient. The
%                  algorithm stops when norm(grad) < Tol
%                  Default: 1e-10
%
% 'FuncTol'     :   sets the tolerance for the loss decay. The algorithm
%                  stops if, in the last three iterations, 
%                  L(x_{i}) - L(x_{i-1}) < LossTol
%                  Default: 1e-6
%
% 'MaxIter'     :   sets the maximum number of iterations
%                  Default: 200
%
% 'SearchDirection' : algorithm used to compute the search direction p.
%                  Default: 'bfgs'
% >>'gd'        :   Gradient Descent, p = - grad
% >>'cgm'       :   (nonlinear) Conjugate Gradient Method 
%                  p = -grad + beta*p0, where beta is calculated according to
%                  Polak-Ribière approximation.
% >>'bfgs'      :   p = - H*grad where the Hessian H is iteratively
%                  approximated using BFGS algorithm
%
% 'SteppingAlgorithm': set the stepping algorithm where Step0 = -Alfa*p
%                     Default/recommended: 'adaptive'
% >>'fixed'     :   Step = Step0
% >>'adaptive'  :   zero-order method that returns a relaxed estimate of the 
%                  optimal step size Alfa combining a backtracing method with
%                  forwardtracking in case L(x+Step0)<L(x)
% >>'newton'    :   uses a 1st order and/or a 2nd order approximation of
%                  L(x+step) to attain the optimal step size so that
%                  step* = minimize_{step} L(x+step). 
%
% 'Tau'         :   set the step-size decay in the adaptive step
%                  algorithm.
%                  Default: 0.8
%
% 'Alfa'        :   set the intial step-size for SteppingAlgorithm,
%                  i.e. a scalar factor. In case of fixed step size, this
%                  is the effective step size which is taken, once
%                  'ScalingGrad' and 'SearchDirection' are set.
%                  Step = -p*Alfa
%                  Default: 1
%
% 'Jacobian'    :   function handle for the analytic Jacobian. If
%                  supplied, the gradient is computed as
%                              dL/dx = 2*Jac(x)'*f(x)
%                  Default: numerical gradient computation
%                      dL/dx = (L(f(x+dx))-L(f(x-dx)))/(2*dx)
%                  where dx = max(x,1)*eps^(1/3)
%
% 'ConvergencePlot' : Display convergency analytics in a graph.
%                    Active 1, Not Active 0 (default)
% 'Verbose'     :   Display iteration progress in the Command Window. 
%                  Active 1 (default), Not Active 0
%
%% Setting up
% Default values
gradTol=1e-10;
stepTol = 1e-6;
funcTol=1e-6;
maxIter=400;
stepIter = NaN;
tau=0.8;
Alfa=1;
SearchDirection='bfgs';
stepSize='adaptive';
gradMode = 'numerical';
ConvergencePlot = 0;
verbose = 1;
H = diag(ones(size(x0)));
grad=ones(length(x0),1);
isscalarfun = isscalar(fun(x0));
iter=0;
memSet=inf*ones(3,length(x0)+1);
if isempty(lb)
    lb=-Inf*ones(length(x0),1);
end
if isempty(ub)
    ub=Inf*ones(length(x0),1);
end

%% User options
for i=1:2:nargin-5
    switch varargin{i}
        case 'GradTol'
            try
                assert(isfloat(varargin{i+1}))
                gradTol=varargin{i+1};
            catch
                error(join([varargin{i+1}, ' is not a correct value for', varargin{i}]))
            end
            
        case 'FuncTol'
            try
                assert(isfloat(varargin{i+1}))
                funcTol=varargin{i+1};
            catch
                error(join([varargin{i+1}, ' is not a correct value for', varargin{i}]))
            end
            
        case 'StepTol'
            try
                assert(isfloat(varargin{i+1}))
                stepTol=varargin{i+1};
            catch
                error(join([varargin{i+1}, ' is not a correct value for', varargin{i}]))
            end
            
            
        case 'MaxIter'
            try
                assert(isfloat(varargin{i+1}))
                maxIter=varargin{i+1};
            catch
                error(join([varargin{i+1}, ' is not a correct value for', varargin{i}]))
            end
        case 'Tau'
            try
                assert(isfloat(varargin{i+1}))
                tau=varargin{i+1};
            catch
                error(join([varargin{i+1}, ' is not a correct value for', varargin{i}]))
            end
            
        case 'Alfa'
            try
                assert(isfloat(varargin{i+1}))
                Alfa=varargin{i+1};
            catch
                error(join([varargin{i+1}, ' is not a correct value for', varargin{i}]))
            end
            
        case 'SearchDirection'
            switch varargin{i+1}
                case 'gd'
                    SearchDirection='gd';
                case 'cgm'
                    SearchDirection='cgm';
                case 'bfgs'
                    SearchDirection='bfgs';
                otherwise
                    error(join([varargin{i+1}, ' is not a correct value for', varargin{i}]))
            end
            
        case 'Jacobian'
            assert(isa(varargin{i+1},'function_handle'),'"ExplGrad, ** " **is not a function handle for explicit gradient computation')
            Jac = varargin{i+1};
            gradMode = 'explicit';
            
        case 'SteppingAlgorithm'
            switch varargin{i+1}
                case 'adaptive'
                    stepSize='adaptive';
                case 'newton'
                    stepSize='newton';
                case 'fixed'
                    stepSize='fixed';
                otherwise
                    error(join([varargin{i+1}, ' is not a correct value for', varargin{i}]))
            end
            
        case 'ConvergencePlot'
            assert(varargin{i+1}==0 || varargin{i+1}==1)
            ConvergencePlot=varargin{i+1};
            
        case 'Verbose'
            try
                assert(varargin{i+1}==0 || varargin{i+1}==1)
                verbose=varargin{i+1};
            catch
                error(join([varargin{i+1}, ' is not a correct value for', varargin{i}]))
            end
        otherwise
            error(join([varargin{i},' is not an available option for "minimize"!']))
    end
    
    
    
end

format shortG

if ConvergencePlot
    fig = figure();  
    subplot(2,1,1);hold on; grid on;
    xlabel('Iterations')
    ylabel('L(x)')
    subplot(2,1,2); hold on; grid on;
    xlabel('Iterations')
    ylabel('Step Size Iterations')
end

%% Starting point
cost0=objective(x0);

assert(~isnan(cost0),'Cost function evaluation failed, residual is NaN')
assert(length(cost0)==1,'----')
if verbose
    fprintf('%9s%15s%15s\n',["Iteration","L(X)","Steps iter"])
end
if ConvergencePlot
    printPlot()
end
%% Optim iterations
while 1
    
    % perturbation
    dx=max(abs(x0),1)*eps^(1/3);
    
    iter=iter+1;
    if verbose
        fprintf('%9.f%15.2e',[iter,cost0])
    end
    
    switch gradMode
        case 'numerical'
            grad = numGrad(x0,dx);
        case 'explicit'
            if isscalarfun
                grad = Jac(x0);
            else
                grad = 2*Jac(x0)'*(fun(x0));
            end
        otherwise
            grad = numGrad(x0,dx);
    end
    
    %% Setting to zero gradient components of satureted variables (only if
    % their update is in the saturated direction)
    
    grad(x0==lb-grad>0)=0;
    grad(x0==ub-grad<0)=0;
    
    %% Step size evaluation
    % Adaptive step paramter
    
    switch SearchDirection
        case 'gd'
            p=-grad;
        case 'cgm'
            if ~exist('grad0','var')
                p = -grad;
                grad0 = grad;
            else
                betaPR = -grad.'*(-grad+grad0)/(grad0.'*grad0);
                beta = max(0,betaPR);
                p = -grad+beta*p;
                grad0 = grad;
            end
        case 'bfgs'
            if ~exist('grad0','var')
                p = -H*grad;
                grad0 = grad;
            else
                y = grad-grad0;
                if y~=0
                    H = H + (stepOut'*y+y'*H*y)*(stepOut*stepOut')/(stepOut'*y)^2-(H*y*stepOut'+stepOut*y'*H)/(stepOut'*y);
                else
                    H = diag(ones(size(x0)));
                end
                p = -H*grad;
                grad0 = grad;
            end
        otherwise
            error('The specified SearchDirection algorithm is not available. Use "gd","cgm" or "bfgs" instead.')
    end
    
    alfa=Alfa;
    stepIter=0;
    
    % First step
    step = alfa*p;
    
    x = min(max(x0+step,lb),ub);
    
    costStep=objective(x);
    
    % Stepping Algorithm
    switch stepSize
        
        case 'adaptive'
            if costStep>cost0
                costStep0=cost0;
                alfa=alfa*tau;
                while  costStep-costStep0<0 || costStep>cost0
                    
                    costStep0=costStep;
                    
                    step= p*alfa;
                    
                    x=min(max(x0+step,lb),ub);
                    
                    costStep=objective(x);
                    
                    assert(~isnan(costStep),'Cost function evaluation failed, residual is NaN.')
                    
                    stepIter=stepIter+1;
                    stepOut=step;
                    alfa=alfa*tau;
                    
                end
                if verbose
                    fprintf('%15.f\n',stepIter)
                end
            else
                costStep0 = cost0;
                alfa = alfa*(1/tau);
                while costStep-costStep0<0
                    
                    costStep0 = costStep;
                    
                    step0=step;
                    
                    step = p*alfa;
                    
                    x=min(max(x0+step,lb),ub);
                    
                    costStep=objective(x);
                    
                    assert(~isnan(costStep),'Cost function evaluation failed, residual is NaN.')
                    
                    stepIter=stepIter+1;

                    alfa=alfa*(1/tau);
                end
                stepOut = step0;
                if verbose
                    fprintf('%15.f\n',stepIter)
                end
%                stepOut=step;
            end
            
        case 'fixed'
            stepIter=0;
            stepOut=step;
            if verbose
                fprintf('%15.f\n',stepIter)
            end
            
        case 'newton'
            alfa = 0;
            dxLS = eps^(1/3);
            costLS = cost0;
            while stepIter<10000 
                dydx = generalizedNumGrad(@(x) objective(x0+(alfa+x)*p),0,dxLS);
                if abs(dydx)<=gradTol
                    break
                end
                ddyddx = (2*objective(x0+(alfa)*p)-5*objective(x0+(alfa+dxLS)*p)+4*objective(x0+(alfa+2*dxLS)*p)-objective(x0+(alfa+3*dxLS)*p))/dxLS^3;
                if ddyddx>0
                    stepLS=-dydx/ddyddx;
                else
                    stepLS=-objective(x0+alfa*p)/dydx;
                end
                while objective(x0+(alfa+stepLS)*p)>costLS && abs(stepLS)>0
                    stepLS = stepLS*0.5;
                end
                alfa = alfa + stepLS;
                costLS = objective(x0+alfa*p);
                
                stepIter=stepIter+1;

            end
            stepOut = alfa*p;
            
            if verbose
                fprintf('%15.f\n',stepIter)
            end
        otherwise
            error('The specified stepping algorithm is not available. Use "adaptive" (recommended), "fixed" or "newton".')
    end
    
    cost0=costStep;    
    stepOut = min(max(stepOut,lb-x0),ub-x0);
    x0 = x0+stepOut;
    if cost0<=memSet(3,end)
        memSet=[memSet(2:3,:); x0' cost0];
    end
    if ConvergencePlot
        printPlot()
    end
    
    if norm(grad)<gradTol && norm(stepOut)<stepTol
        if verbose
            fprintf('%s\n',"Local minimum possible!");
            printInfo()
        end
        break
    end
    
    if abs(memSet(1,end)-memSet(2,end))<funcTol && abs(memSet(2,end)-memSet(3,end))<funcTol && norm(stepOut)<stepTol
        if verbose
            fprintf('%s%.2e%s\n%s\n',["In the last three iterations, f(x) decay was less than funcTol: ",funcTol," with norm(step)<StepTol, BUT norm(grad)>GradTol","Algorithm stalled. Returning..."]);
            printInfo()
        end
        break
    end
    
    
    if iter>=maxIter
        if verbose
            fprintf('%s\n',"Reached max number of iterations.");
            printInfo()
        end
        break
    end
    
end
var=memSet(3,1:end-1)';

%% Support functions
    function gr = numGrad(x0,dx)
        gr = zeros(length(x0),1);
        for r=1:length(x0) %gradient calculation
            
            dU=x0;
            dL=x0;
            
            dU(r)=min(x0(r)+dx(r),ub(r));
            
            costU=objective(dU);
            
            dL(r)=max(x0(r)-dx(r),lb(r));
            
            costL=objective(dL);
            
            gr(r)=(costL-costU)/(dL(r)-dU(r));
            
        end
    end

    function gr = generalizedNumGrad(f,x0,dx)
        gr = zeros(length(x0),1);
        for r=1:length(x0) %gradient calculation
            
            dU=x0;
            dL=x0;
            
            dU(r)=x0(r)+dx(r);
            
            costU=f(dU);
            
            dL(r)=x0(r)-dx(r);
            
            costL=f(dL);
            
            gr(r)=(costL-costU)/(dL(r)-dU(r));
            
        end
    end

    function cost = objective(x)
        cost=fun(x);
        if isscalarfun
            return
        end
        cost=sum((cost).^2);
    end

    function printInfo()
        fprintf('%-15s',"Values: ");
        fprintf('%15.2f',x0);
        fprintf('\n')
        fprintf('%-15s',"Gradient: ");
        fprintf('%15.2e',grad);
        fprintf('\n')
        fprintf('\n')
        fprintf('%-15s%15.2e\n',["Norm of Grad: ",norm(grad)]);
    end

    function printPlot()
        fig;subplot(2,1,1); hold on; grid on
        scatter(iter, cost0,20,[0 0 1],'filled')
        subplot(2,1,2); hold on; grid on
        scatter(iter,stepIter,20,[0 0 1],'filled')
    end
end



