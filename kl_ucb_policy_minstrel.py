import math
from re import S

import numpy as np

def kl_bernoulli(p, q):
    """
    Compute kl-divergence for Bernoulli distributions
    """
    result = p * np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
    return result

def dkl_bernoulli(p, q):
    result = (q-p)/(q*(1.0-q))
    return result

def kl_exponential(p, q):
    """
    Compute kl-divergence for Exponential distributions
    """
    result = (p/q) - 1 - np.log(p/q)
    return result

def dkl_exponential(p, q):
    result = (q-p)/(q**2)
    return result

def klucb_upper_newton(kl_distance, N, S, k, t, precision = 1e-6, max_iterations = 50, dkl = dkl_bernoulli):
    """
    Compute the upper confidence bound for each arm using Newton's iterations method
    """
    delta = 0.1
    logtdt = np.log(t)/N[k]
    p = max(S[k]/N[k], delta)
    if(p>=1):
        return 1

    converged = False
    q = p + delta

    for n in range(max_iterations):
        f  = logtdt - kl_distance(p, q)
        df = - dkl(p, q)

        if(f*f < precision):
            converged = True
            break

    q = min(1 - delta , max(q - f / df, p + delta))

    if(not converged):
        print("KL-UCB algorithm: Newton iteration did not converge!", "p=", p, "logtdt=", logtdt)

    return q

def klucb_upper_bisection(kl_distance, N, S, k, t, precision = 1e-6, max_iterations = 50):
    """
    Compute the upper confidence bound for each arm with bisection method
    """
    upperbound = np.log(t)/N[k]
    reward = S[k]/N[k]

    u = upperbound
    l = reward
    n = 0

    while n < max_iterations and u - l > precision:
        q = (l + u)/2
        if kl_distance(reward, q) > upperbound:
            u = q
        else:
            l = q
        n += 1

    return (l+u)/2

def klucb_upper_bisection_with_l(kl_distance, N, S, k, l, precision=1e-6, max_iterations=50):
    """
    Compute the upper confidence bound for each arm with bisection method
    Change the upper bound as log(l)
    """
    t = np.sum(N)
    if l[k]==0:
        upperbound = np.log(t) / N[k]
    else:
        upperbound = np.log(l[k]) / N[k]
    reward = S[k] / N[k]

    up = upperbound
    low = reward
    n = 0

    while n < max_iterations and up - low > precision:
        q = (low + up) / 2
        if kl_distance(reward, q) > upperbound:
            up = q
        else:
            low = q
        n += 1

    return (low + up) / 2

class KLUCBPolicy :
    """
    KL-UCB algorithm
    """
    def __init__(self, K, rate, S, N, klucb_upper = klucb_upper_bisection, kl_distance = kl_bernoulli, precision = 1e-6, max_iterations = 50):
        self.K = K
        self.rate = rate
        self.kl_distance = kl_distance
        self.klucb_upper = klucb_upper
        self.precision = precision
        self.max_iterations = max_iterations
        self.S = S
        self.N = N

    def reset(self):
        self.N = np.zeros(self.K) #Count for each arm to be selected
        self.S = np.zeros(self.K) #Count for success of each arm

    def select_next_arm(self):
        t = np.sum(self.N)
        indices = np.zeros(self.K)
        for k in range(self.K):
            if(self.N[k]==0):
                return k

            #KL-UCB index
            indices[k] = self.klucb_upper(self.kl_distance, self.N, self.S, k, t, self.precision, self.max_iterations)*self.rate[k]
        # print('indices are: ', indices)
        selected_arm = np.argmax(indices)
        return selected_arm

    def update_state(self, S, N):
        self.S = S
        self.N = N

class GORS(KLUCBPolicy):
    """
    Graphical OptimalRate Sampling
    """
    def reset(self):
        # super(GORS, self).reset()
        self.L = 0 #Leading arm
        self.l = np.zeros(self.K) #Leading counts for each arm

    def select_next_arm(self, gamma=10):
        t = np.sum(self.N)
        indices = np.zeros(self.K)
        for k in range(self.K): #For the first K time slots
            if (self.N[k] == 0):
                return k
        for k in range(self.K): #After the first K time slots
            if (self.l[self.L]-1)%gamma==0:
                return self.L
            indices[k] = self.klucb_upper(self.kl_distance, self.N, self.S, k, self.l, self.precision,
                                          self.max_iterations)*self.rate[k]
        selected_arm = np.argmax(indices)
        # print('indices are: ',indices)
        return selected_arm

    def update_state(self, S, N):
        self.S = S
        self.N = N
        tp_estimate = np.multiply(self.rate, np.true_divide(self.S, self.N))  # Estimated throughput
        for i in range(len(tp_estimate)):
            if math.isnan(tp_estimate[i]):
                tp_estimate[i] = 0
        self.L = np.argmax(tp_estimate)  # Leading arm
        self.l[self.L] += 1

class KLUCB_EWMA(KLUCBPolicy):
    """
    KL-UCB with Exponentially Weighted Moving Averages
    """
    def __init__(self, K, rate, alpha, tau):
        super().__init__(K, rate, klucb_upper=klucb_upper_bisection, kl_distance=kl_bernoulli, precision=1e-6,
                 max_iterations=50)
        self.alpha = alpha #EWMA discount
        self.tau = tau #Window size

    def reset(self):
        super().reset()
        self.t = 0 #Time

    def update_state(self, k, r):
        if self.t < self.tau:
            super().update_state(k,r)
            return
        self.N[k] = (1 - self.alpha) * self.N[k] + self.alpha * 1
        self.S[k] = (1 - self.alpha) * self.S[k] + self.alpha * r

class KLUCB_SW(KLUCBPolicy):
    """
    KL-UCB with Sliding Window
    """
    def __init__(self, K, rate, tau):
        super().__init__(K, rate, klucb_upper=klucb_upper_bisection, kl_distance=kl_bernoulli, precision=1e-6,
                 max_iterations=50)
        self.tau = tau #Window size

    def reset(self):
        self.N_window = [[] for i in range(self.K)]
        self.S_window = [[] for i in range(self.K)]
        self.N = np.zeros(self.K)
        self.S = np.zeros(self.K)

    def update_state(self, k, r):
        if len(self.N_window[k]) == self.tau:
            self.N_window[k].pop(0)
            self.N_window[k].append(1)
            self.S_window[k].pop(0)
            self.S_window[k].append(r)
        else:
            self.N_window[k].append(1)
            self.S_window[k].append(r)
        self.N[k] = sum(self.N_window[k])
        self.S[k] = sum(self.S_window[k])

class GORS_EWMA(GORS):
    """
    G-ORS with EWMA
    """
    def __init__(self, K, rate, tau, alpha):
        super().__init__(K, rate, klucb_upper=klucb_upper_bisection_with_l)
        self.alpha = alpha #EWMA discount
        self.tau = tau#Window size

    def reset(self):
        super().reset()
        self.t = 0

    def update_state(self, k, r):
        if self.t < self.tau:
            super().update_state(k, r)
            return

        self.N[k] = (1 - self.alpha) * self.N[k] + self.alpha * 1
        self.S[k] = (1 - self.alpha) * self.S[k] + self.alpha * r
        tp_estimate = np.multiply(self.rate, np.true_divide(self.S, self.N))  # Estimated throughput
        for i in range(len(tp_estimate)):
            if math.isnan(tp_estimate[i]):
                tp_estimate[i] = 0
        self.L = np.argmax(tp_estimate)  # Leading arm
        self.l[self.L] = (1 - self.alpha) * self.l[self.L] + self.alpha * 1

class GORS_SW(GORS):
    """
    G-ORS with Sliding Window
    """
    def __init__(self, K, rate, tau):
        super().__init__(K, rate, klucb_upper=klucb_upper_bisection_with_l)
        self.tau = tau #EWMA discount

    def reset(self):
        super().reset()
        self.N_window = [[] for i in range(self.K)]
        self.S_window = [[] for i in range(self.K)]
        self.l_window = [[] for i in range(self.K)]

    def update_state(self, k, r):
        if len(self.N_window[k]) == self.tau:
            self.N_window[k].pop(0)
            self.S_window[k].pop(0)

        self.N_window[k].append(1)
        self.S_window[k].append(r)

        self.N[k] = sum(self.N_window[k])
        self.S[k] = sum(self.S_window[k])

        tp_estimate = np.multiply(self.rate, np.true_divide(self.S, self.N))  # Estimated throughput
        for i in range(len(tp_estimate)):
            if math.isnan(tp_estimate[i]):
                tp_estimate[i] = 0
        self.L = np.argmax(tp_estimate)  # Leading arm

        if len(self.l_window[k]) == self.tau:
            self.l_window[self.L].pop(0)

        self.l_window[self.L].append(1)
        self.l[self.L] = sum(self.l_window[self.L])