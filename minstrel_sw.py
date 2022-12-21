import time
import numpy as np
import math
import pickle
import kl_ucb_policy_minstrel
import os



def echo_switch(idx):
    if (idx == 0):
        os.system("echo '00' > /proc/mytest")
    elif (idx == 1):
        os.system("echo '01' > /proc/mytest")
    elif (idx == 2):
        os.system("echo '02' > /proc/mytest")
    elif (idx == 3):
        os.system("echo '03' > /proc/mytest")
    elif (idx == 4):
        os.system("echo '04' > /proc/mytest")
    elif (idx == 5):
        os.system("echo '05' > /proc/mytest")
    elif (idx == 6):
        os.system("echo '06' > /proc/mytest")
    elif (idx == 7):
        os.system("echo '07' > /proc/mytest")
    elif (idx == 8):
        os.system("echo '08' > /proc/mytest")
    elif (idx == 9):
        os.system("echo '09' > /proc/mytest")
    elif (idx == 10):
        os.system("echo '10' > /proc/mytest")
    elif (idx == 11):
        os.system("echo '11' > /proc/mytest")
    elif (idx == 12):
        os.system("echo '12' > /proc/mytest")
    elif (idx == 13):
        os.system("echo '13' > /proc/mytest")
    elif (idx == 14):
        os.system("echo '14' > /proc/mytest")
    elif (idx == 15):
        os.system("echo '15' > /proc/mytest")

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
            print("choose a rate....")
            his_group_succ[:] = cur_group_succ
            his_group_total[:] = cur_group_total
            break
    return True

def check_init(his_group_succ, his_group_total):
    is_init = True
    idx = 0
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
        while (check_update(his_group_succ, his_group_total) and idx < 8):
            print("time is",t)
            arm_klucb = klucb.select_next_arm()
            print("select arm is ", arm_klucb)
            echo_switch(arm_klucb)
            klucb.update_state(np.array(his_group_succ), np.array(his_group_total))
            idx += 1




def main():

    K = 16
    T = 5000
    his_group_succ = 16*[0]
    his_group_total = 16*[0]
    _his_group_succ = 16*[0]
    _his_group_total = 16*[0]
    his_group_succ_array = np.array(his_group_succ)
    his_group_total_array = np.array(his_group_total)
    check_init(his_group_succ, his_group_total)
    _his_group_succ[:] = his_group_succ
    _his_group_total[:] = his_group_total
    rate = np.array([5.6, 10.6, 14.9, 18.8, 25.4, 30.7,33.0, 35.1, 6.2, 11.6, 16.3, 20.4, 27.2, 32.8, 35.1, 37.3])
    # klucb = kl_ucb_policy_minstrel.KLUCBPolicy(K, rate, his_group_succ_array, his_group_total_array) #Original KL UCB
    klucb = kl_ucb_policy_minstrel.GORS_SW(K, rate, 10)
    # total_rewards_list_klucb = np.zeros((runs, T))
    actions_list_klucb = []
    start_time = time.time()
    t = 0
    actions_klucb = np.zeros((K, T), dtype=int)
    arm_klucb = 0
    _arm_klucb = 0
    
    while (True):
        while (check_update(his_group_succ, his_group_total)):
            print("time is",t)
            # print(f"succ is{his_group_succ[arm_klucb]-_his_group_succ[arm_klucb]} att is {his_group_total[arm_klucb]-_his_group_total[arm_klucb]}")
            for i in range(16):
                S = his_group_succ[i]-_his_group_succ[i]
                N = his_group_total[i]-_his_group_total[i]
                if N != 0:
                    klucb.update_state(i, N, S)
                
            arm_klucb = klucb.select_next_arm()
            print("select arm is ", arm_klucb)
            echo_switch(arm_klucb)
            actions_klucb[arm_klucb, t] = 1
            # klucb.update_state(arm_klucb, N, S)
            t += 1
            _his_group_succ[:] = his_group_succ
            _his_group_total[:] = his_group_total
            # _arm_klucb = arm_klucb



main()
