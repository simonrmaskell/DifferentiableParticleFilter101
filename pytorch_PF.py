import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pylab as plt
import timeit

def runparticlefilter(A,Q,H,R,mu,sigma,y,T,P):
    #run particle filter
    torch.manual_seed(0)

    xp = torch.zeros(P)
    lw = torch.zeros(P)
    lnorm = torch.zeros(P)

    loglikelihood = torch.zeros(T)

    xp[:] = mu+torch.sqrt(sigma)*torch.randn(1,1,P)
    lw[:] = torch.distributions.Normal(H*xp.clone(),R).log_prob(y[1])-torch.log(torch.ones(1,1)*P)

    resampletot = 0
    for t in range(1,T):
        #print(t)
        xp[:] = A*xp.clone()+torch.sqrt(Q)*torch.randn(1,1,P) #assume prior is proposal
        lw[:] = lw.clone()+torch.distributions.Normal(H*xp.clone(),R).log_prob(y[t])
        loglikelihood[t] = torch.logsumexp(lw.clone(),dim=0)
        #print(loglikelihood[t])
        wnorm = torch.exp(lw-loglikelihood[t]) #normalised weights (on a linear scale)
        #print(wnorm)
        neff = 1./torch.sum(wnorm*wnorm)
        xest = torch.sum(wnorm*xp)
        xsqest = torch.sum(wnorm*xp*xp)
        #print('xtrue:',xtrue[t],'y:',y[t],'xest:',xest,'std:',torch.sqrt(xsqest-xest*xest))
        if(neff<P/2):
        #resample
            resampletot = resampletot + 1
            #print('resampling on iteration ',t,' with Neff of',neff)
            #xp[:,t] = systematicresample(wnorm, xp[:,t]).squeeze()
            idx = torch.multinomial(wnorm, P, replacement=True)
            xp[:] = xp[idx]
            lw[:] = loglikelihood[t]-torch.log(torch.ones(1,1)*P)
    print('%resampling = ',resampletot/(T-1))
    return(loglikelihood[T-1])

torch.autograd.set_detect_anomaly(True)

P=750
T=250


A=Variable(torch.ones(1,1)*1,requires_grad=True)
Atrue = torch.ones(1,1)
Q=torch.ones(1,1)*0.01
#H=torch.ones(1,1)*1.1
Htrue = torch.ones(1,1)
R=torch.ones(1,1)

mu=torch.zeros(1,1)
sigma=torch.ones(1,1)

#sim the data
xtrue = torch.zeros(T,1)
xtrue[0] = mu+torch.sqrt(sigma)*torch.randn(1,1)
for t in range(1,T):
    xtrue[t] = Atrue*xtrue[t-1]+torch.sqrt(Q)*torch.randn(1,1)
y = torch.zeros(T,1)
for t in range(0,T):
    y[t] = Htrue*xtrue[t]+torch.sqrt(R)*torch.randn(1,1)

Nvals = 100
Hvals = torch.linspace(0.5, 1.5, Nvals)
likevals = torch.zeros(len(Hvals))
gradvals = torch.zeros(len(Hvals))
for i in range(len(Hvals)):
    thisH = Variable(Hvals[i],requires_grad=True)
    tmp = runparticlefilter(A,Q,thisH,R,mu,sigma,y,T,P)
    tmp.backward()
    likevals[i] = tmp
    gradvals[i] = thisH.grad
    print('iteration:',i,', likelihood:',tmp,', gradient:',thisH.grad)

test = likevals.detach().numpy()
thetas = Hvals.detach().numpy() #np.linspace(0.5, 1.5, Nvals)
plt.figure()
plt.axvline(x=1.0, color='black')
plt.plot(thetas, test)
plt.figure()
plt.plot(thetas, gradvals.detach().numpy())
#SM: performance seems poor when H>Htrue (is the proposal sufficiently heavy-tailed in such a setting?)
plt.show()
