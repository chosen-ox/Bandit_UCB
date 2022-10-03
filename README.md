# bandits_kl-ucb
### ENSAE - Online learning and aggregation project
Based on the article "[The KL-UCB algortihm for bounded stochastic bandits and beyond](https://arxiv.org/abs/1102.2490)" by **Aurélien Garivier** and **Olivier Cappé**.  
Code developped by **Thomas Levy** and **Zakarya Ali**, inspired by [SMPyBandits](https://perso.crans.org/besson/phd/SMPyBandits/index.html) and [MultiplayBanditLib](https://github.com/jkomiyama/multiplaybanditlib). 

**Objective : Demonstrate KL-UCB (Kullback Leibler Upper Confidence Bound) algorithm advantages over UCB  in stochastic bandits problems.**

## Structure
- [kl_ucb_policy.py](https://github.com/zakaryaxali/bandits_kl-ucb/blob/master/kl_ucb_policy.py) : Code for KL-UCB algorithm
- 3 Notebooks :
  - [scenario_1.ipynb](https://github.com/zakaryaxali/bandits_kl-ucb/blob/master/scenario_1.ipynb) : Follow the article first scenario, two arms with Bernoulli rewards.   
  - [scenario_2.ipynb](https://github.com/zakaryaxali/bandits_kl-ucb/blob/master/scenario_2.ipynb) : 10 arms with low Bernoulli rewards.
  - [scenario_3.ipynb](https://github.com/zakaryaxali/bandits_kl-ucb/blob/master/scenario_3.ipynb) : 2 arms with truncated exponential rewards.
  
# Methology

We implement two parts: user space part, kernel space part.

We use Python and C respectively

## Environment

## User Space Part: Implement KL_UCB

```python
def main():

    K = 16
    T = 5000
    his_group_succ = 16*[0]
    his_group_total = 16*[0]
    his_group_succ_array = np.array(his_group_succ)
    his_group_total_array = np.array(his_group_total)
    check_init(his_group_succ, his_group_total)
    rate = np.array([5.6, 10.6, 14.9, 18.8, 25.4, 30.7,33.0, 35.1, 6.2, 11.6, 16.3, 20.4, 27.2, 32.8, 35.1, 37.3])
    klucb = kl_ucb_policy_minstrel.KLUCBPolicy(K, rate, his_group_succ_array, his_group_total_array) #Original KL UCB
    # total_rewards_list_klucb = np.zeros((runs, T))
    actions_list_klucb = []
    start_time = time.time()
    t = 0
    actions_klucb = np.zeros((K, T), dtype=int)
    while (True):
        while (check_update(his_group_succ, his_group_total)):
            print("time is",t)
            arm_klucb = klucb.select_next_arm()
            print("select arm is ", arm_klucb)
            echo_switch(arm_klucb)
            actions_klucb[arm_klucb, t] = 1
            klucb.update_state(np.array(his_group_succ), np.array(his_group_total))
            t += 1
```

### How can we get the success and history numbers of ACK for every rates?

 [rc_stats](https://wireless.wiki.kernel.org/en/developers/documentation/mac80211/ratecontrol/minstrel), debug file generated by [rc80211_minstrel_ht_debugfs.c](https://github.com/chosen-ox/Mac80211_Comment/blob/master/mac80211/rc80211_minstrel_ht_debugfs.c) （kernel space）

![Untitled](https://github.com/chosen-ox/Bandit_UCB/blob/main/minstrel_wiki.png)

We can use **open** function to view *rc_stats* directly

```python
        f = open("/sys/kernel/debug/ieee80211/phy0/netdev:wlp1s0/stations/dc:a6:32:a7:ca:17/rc_stats", 'r')
        f.readline() 
        f.readline() 
        f.readline()
        for i in range(16):
            group = f.readline()
            group = group.split()
            cur_group_succ[i] = int (group[-2])
            cur_group_total[i] = int (group[-1])
        f.close()
```

### How can we send our decision to kernel space?

[/proc](https://man7.org/linux/man-pages/man5/proc.5.html), (The procfilesystem is a pseudo-filesystem which provides an interface to kernel data structures), User space write the *selected_arm* to the /proc and Kernel space can read it immediately.

## Kernel Space

In kernel space, we receive the message from the user space, then kernel space choose corresponding rate to send packets.

```python
minstrel_ht_update_rates(struct minstrel_priv *mp, struct minstrel_ht_sta *mi)
{
	struct ieee80211_sta_rates *rates;
	int i;
	
	printk(KERN_INFO "mac80211->kernelbuffer:%s\n", kernel_buf);

	int first_char = kernel_buf[0] - '0';
	int second_char = kernel_buf[1] - '0';
	int idx = 10 * first_char + second_char;
	if (idx > 7) {
		idx += 56;
	}	
	printk(KERN_INFO "mac80211->kernelbuffer:%d\n", idx);
	if ((idx >= 0 && idx < 8) || (idx > 63 && idx < 72)) {
		idx++;
		idx--;
	}
	else {
		idx = mi->max_tp_rate[0];
	}

	rates = kzalloc(sizeof(*rates), GFP_ATOMIC);
	if (!rates)
		return;

	/* Start with max_tp_rate[0] */
	//minstrel_ht_set_rate(mp, mi, rates, i++, mi->max_tp_rate[0]);
	minstrel_ht_set_rate(mp, mi, rates, i++, idx);

	if (mp->hw->max_rates >= 3) {
		/* At least 3 tx rates supported, use max_tp_rate[1] next */
		minstrel_ht_set_rate(mp, mi, rates, i++, idx);
	}

	if (mp->hw->max_rates >= 2) {
		minstrel_ht_set_rate(mp, mi, rates, i++, idx);
	}

	mi->sta->max_rc_amsdu_len = minstrel_ht_get_max_amsdu_len(mi);
	rates->rate[i].idx = -1;
	rate_control_set_rates(mp->hw, mi->sta, rates);
}
```

## Interaction between two spaces

![Untitled](https://github.com/chosen-ox/Bandit_UCB/blob/main/Interaction_between_two_spaces.png)

## Real-Time Problem

### Though two spaces can communicate  with each other. How can the user space know that the rc_stats has been updated?

No good method. Just use a listener. User space check the rc_stats over and over again. If it was updated, then user space read the data and calculate and then send the selected_arm to the /proc.

```python
def check_update(his_group_succ, his_group_total):
    while (True):
        time.sleep(1)
        cur_group_succ = 16*[0]
        cur_group_total = 16*[0]
        f = open("/sys/kernel/debug/ieee80211/phy0/netdev:wlp1s0/stations/dc:a6:32:a7:ca:17/rc_stats", 'r')
        f.readline() 
        f.readline() 
        f.readline()
        for i in range(16):
            group = f.readline()
            group = group.split()
            cur_group_succ[i] = int (group[-2])
            cur_group_total[i] = int (group[-1])
        f.close()
        if (compareTwoList(cur_group_total, his_group_total)):
            print("choose a rate....")
            his_group_succ[:] = cur_group_succ
            his_group_total[:] = cur_group_total
            break
    return True
```
