import time
import numpy as np
import math
import pickle
import kl_ucb_policy_minstrel
def compareTwoList(list1, list2):
    for i in range(16):
        if (list1[i] != list2[i]):
            return True
    return False


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
            # print(cur_group_total)
            # print(his_group_total)
            print("wrong!")
            his_group_succ = cur_group_succ[:]
            his_group_total = cur_group_total[:]
            break
    return True

def check_init(his_group_succ, his_group_total):
    is_init = True
    f = open("/sys/kernel/debug/ieee80211/phy0/netdev:wlp1s0/stations/dc:a6:32:a7:ca:17/rc_stats", 'r')
    f.readline() 
    f.readline() 
    f.readline()
    for i in range(16):
        group = f.readline()
        group = group.split()
        if (group[-1] != 0):
            is_init = False
        his_group_succ[i] = int (group[-2])
        his_group_total[i] = int (group[-1])
        print(group[-2]+ " " + group[-1])
    f.close()
    if (is_init):
        return



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
            print(t)
            arm_klucb = klucb.select_next_arm()
            print("select arm is ", arm_klucb)
            actions_klucb[arm_klucb, t] = 1
            klucb.update_state(np.array(his_group_succ), np.array(his_group_total))
            t += 1



main()