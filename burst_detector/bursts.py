import numpy as np
import pandas as pd

def base_algo(spike_times, s=5, gamma=.3):
    q, a = find_sequence(spike_times, s, gamma)
    bursts = create_output(spike_times, q)
    return bursts, q

def find_bursts(spike_times, s=5, gamma=.3, max_iter=5):
    q, a = find_sequence(spike_times, s, gamma)
    gaps = np.diff(spike_times)
    
    max_level = -1
    ind = 1
    while True:
        q_old = q
        
        # re-estimate a
        k = a.shape[0]
        for i in range(1,k):
            inds = np.nonzero(q_old==i)[0]
            if inds.shape[0] == 0:
                max_level = i
                break
            a_hat = 1/(np.mean(gaps[inds])) # MLE estimator for exponential dist
            a[i] = max(s*a[i-1], a_hat)
        
        # run again
        ind += 1
        q, a = find_sequence(spike_times, s, gamma, a)
        
        if np.array_equal(q, q_old) or ind==max_iter:
            break
    
    bursts = create_output(spike_times, q)
            
    return bursts, q, a[:max_level]
    

def find_sequence(spike_times, s, gamma, a=None):
    # calculate ISIs
    spike_times = np.sort(spike_times)
    gaps = np.diff(spike_times)
    
    # calculate base rate and number of HMM states
    T = np.sum(gaps)
    n = gaps.shape[0]
    a_0 = n/T
    k = int(np.ceil(1+np.emath.logn(s, T)+np.emath.logn(s, 1/gaps.min())))
    
    # lambdas for transition costs and probability density
    tau = lambda i, j: 0 if i >= j else (j-i) * gamma * np.log(n)
    if a is None:
        a = (s ** np.arange(k)) * a_0
    f = lambda j, x: a[j] * np.exp(-a[j]*x)
    
    # Viterbi algorithm for optimal state sequence
    C = np.zeros(k)         # minimum cost for q indexed by ending state 
    q = np.zeros((k,1))      # optimal sequences indexed by ending state
    
    for t in range(n):
        Cprime = np.zeros(k)
        qprime = np.zeros((k, t+1))
        
        for j in range(k):
            # costs of transitioning from possible previous states to j
            cost = list(map(lambda ell: C[ell] + tau(ell, j), np.arange(k)))
            ell = np.argmin(cost)
            
            # Cj = least previous cost - exponential pmf
            with np.errstate(divide='ignore'):
                Cprime[j] = cost[ell] - np.log(f(j, gaps[t]))
            # update state sequences
            if (t == 0):
                qprime[j, 0] = j
            else:
                qprime[j,:t+1] = np.concatenate((q[ell,:], [j]))
            
        C = Cprime
        q = qprime
        
    # get optimal state sequence
    q = q[np.argmin(C),:]
    
    return q, a


def create_output(spike_times, q):
    # calculate output size
    prev_q = -1
    N = 0
    n = np.diff(spike_times).shape[0]
        
    # calculate output size
    for t in range(n - 1):
        if (q[t] > prev_q) and (q[t+1] >= q[t]):
            N += q[t] - prev_q
        prev_q = q[t]
    N = int(N)
    
    if(N == 0):
        bursts = pd.DataFrame({"level":[0], "start":[spike_times[0]], "end":[spike_times[-1]]})
        return bursts
    
    # create output
    level = np.zeros(N, dtype="int")
    start = np.zeros(N, dtype="float64")
    end = np.zeros(N, dtype="float64")
    bursts = pd.DataFrame({"level":level, "start":start, "end":end})    
    
    # populate output
    burst_ind = 0
    prev_q = -1
    stack = np.zeros(N, dtype="int")
    stack_ind = 0
    
    
    for t in range(n-1):
        if (q[t] > prev_q) and (q[t+1] >= q[t]): # if new bursts need to be added
            n_new = int(q[t] - prev_q)
            for i in range(n_new):
                bursts.at[burst_ind, "level"] = prev_q + i + 1
                bursts.at[burst_ind, "start"] = spike_times[t]
                
                stack[stack_ind] = burst_ind
                burst_ind += 1
                stack_ind += 1
        
        elif q[t] < prev_q: # if old bursts need to be closed
            n_close = int(prev_q - q[t])
            for i in range(n_close):
                stack_ind = max(0, stack_ind - 1)
                bursts.at[stack[stack_ind], "end"] = spike_times[t]
                
        prev_q = q[t]
        
    # close bursts that include last spike
    while stack_ind > 0:
        stack_ind -= 1
        bursts.at[stack[stack_ind], "end"] = spike_times[-1]
        
    return bursts