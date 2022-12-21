import numpy as np
from scipy.stats import beta

class TSBanditPolicy :
    """
    Thompson Sampling Bandit
    """
    def __init__(self, K, rate, Delta):
        self.K = K
        self.rate = rate
        self.delta = Delta # packet error rate (PER) threshold
        self.reset()

    def reset(self):
        self.a = np.ones(self.K) # success number for each rate
        self.b = np.ones(self.K) # failure number for each rate

    def select_next_arm(self):
        p = beta.rvs(self.a, self.b) # estimated PER
        p_avail = p.copy()*(p>self.delta) # available rates with PER larger than delta
        if sum(p_avail) != 0:
            selected_arm = np.argmax(np.multiply(p_avail,self.rate))
        else:
            selected_arm = np.argmax(p)
        # print('indices are: ', indices)
        return selected_arm

    def update_state(self, k, r):
        self.a[k] += r # update success number for selected rate k
        self.b[k] += (1-r) # update failure number for selected rate k

class TSBandit_SW(TSBanditPolicy):
    """
    TS_Bandit with Sliding Window
    """
    def __init__(self, K, rate, Delta, tau):
        super().__init__(K, rate, Delta)
        self.tau = tau # window size

    def reset(self):
        self.a_window = [[] for i in range(self.K)] # sliding window for success number of each rate, [K,tau] matrix
        self.b_window = [[] for i in range(self.K)] # sliding window for failure number of each rate, [K, tau] matrix
        self.a = np.zeros(self.K)
        self.b = np.zeros(self.K)

    def update_state(self, k, r):
        if len(self.a_window[k]) == self.tau:
            self.a_window[k].pop(0)
            self.a_window[k].append(r)
            self.b_window[k].pop(0)
            self.b_window[k].append(1-r)
        else:
            self.a_window[k].append(r)
            self.b_window[k].append(1-r)
        self.a[k] = sum(self.a_window[k]) # alpha is the sum of success numer in the window
        self.b[k] = sum(self.b_window[k]) # beta is the sum of failure number in the window


class CBanditPolicy(TSBanditPolicy):
    """
    Correlated Bandit based on Thompson Sampling
    """
    def __init__(self, K, rate, s, delta):
        super().__init__(K, rate, delta)
        self.s = s # conditional upper bound matrix [[K,K],[K,K]]
        self.reset()

    def reset(self):
        super().reset()
        self.A = np.zeros(self.K)
        self.Phi = np.zeros([self.K,self.K]) # expected pseudo-rewards

    def reduce_set(self):
        n = self.a + self.b - 2 # selected times for each arm
        t = sum(n) # total time
        self.Sig_S = np.zeros(self.K)
        self.Sig_S += (n>=(t//self.K)) # a boolean array, 1 indicating significant rate, 0 indicating insignificant rate
        r_emp = (beta.rvs(self.a, self.b)*self.rate)*self.Sig_S
        # r_emp = beta.rvs(self.a, self.b)*self.rate
        lead = np.argmax(r_emp) # leading arm in significant set
        mu_lead = r_emp[lead]

        # self.A = np.zeros(self.K) # competitive set, a boolean array. 1 indicating competitive rate, 0 indicating non-competitive rate
        for k in range(self.K):
            phi_tmp =self.Sig_S*self.Phi[k,:]
            if np.min(phi_tmp) >= mu_lead:
                self.A[k] = 1
        self.A[lead] = 1

        for k in range(self.K):
            min = np.max(self.Phi[k,:])
            min_idx = -1
            for l in range(self.K):
                if self.Sig_S[l] == 1 and self.Phi[k,l] <= min:
                    min = self.Phi[k,l]
                    min_idx = l
            if min_idx == -1:
                continue
            elif min >= mu_lead:
                self.A[min_idx] = 1

    def select_next_arm(self):
        t = sum(self.a+self.b) - 2*self.K
        if t <= self.K-1:
            return int(t)
        p = beta.rvs(self.a, self.b) # estimated PER
        p_comp = p.copy()*self.A # competitive rates
        selected_arm = np.argmax(np.multiply(p_comp,self.rate))
        return selected_arm

    def update_state(self, k, r):
        # print('the reward is', r)
        self.a[k] += r # update success number for selected rate k
        self.b[k] += (1-r) # update failure number for selected rate k
        n = self.a[k] + self.b[k] - 2
        for l in range(self.K):
            self.Phi[l,k] = ((n-1)*self.Phi[l,k]+self.s[r][l,k])/n # update pseudo-rewards
        for k in range(self.K):
            self.Phi[k,k] = beta.rvs(self.a[k],self.b[k])*self.rate[k]

def construct_s(rate, K):
    s = []
    s.append(np.zeros([K,K]))
    s.append(np.zeros([K,K]))
    for k in range(K):
        for l in range(K):
            if rate[l] < rate[k]:
                s[0][l,k] = rate[l]
            else:
                s[0][l,k] = 0
    for k in range(K):
        for l in range(K):
            s[1][l,k] = rate[l]
    return s



