import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
k=5
df=pd.read_csv('simulation_summary_k=5.csv')
df.info()
df.describe()
level1_mean_waiting_time=df['ctas1_mean_waiting'].tolist()
level2_mean_waiting_time=df['ctas2_mean_waiting'].tolist()
level3_mean_waiting_time=df['ctas3_mean_waiting'].tolist()
level4_mean_waiting_time=df['ctas4_mean_waiting'].tolist()
level5_mean_waiting_time=df['ctas5_mean_waiting'].tolist()

level1_mean_nurse=df['ctas1_mean_nurse'].tolist()
level2_mean_nurse=df['ctas2_mean_nurse'].tolist()
level3_mean_nurse=df['ctas3_mean_nurse'].tolist()
level4_mean_nurse=df['ctas4_mean_nurse'].tolist()
level5_mean_nurse=df['ctas5_mean_nurse'].tolist()

level1_service_time=df['ctas1_mean_service'].tolist()
level2_service_time=df['ctas2_mean_service'].tolist()
level3_service_time=df['ctas3_mean_service'].tolist()
level4_service_time=df['ctas4_mean_service'].tolist()
level5_service_time=df['ctas5_mean_service'].tolist()

level1_whole=df['ctas1_mean_whole'].tolist()
level2_whole=df['ctas2_mean_whole'].tolist()
level3_whole=df['ctas3_mean_whole'].tolist()
level4_whole=df['ctas4_mean_whole'].tolist()
level5_whole=df['ctas5_mean_whole'].tolist()


level1_whole_wait=[c+d for c,d in zip(level1_mean_waiting_time, level1_mean_nurse)]
level2_whole_wait=[c+d for c,d in zip(level2_mean_waiting_time, level2_mean_nurse)]
level3_whole_wait=[c+d for c,d in zip(level3_mean_waiting_time, level2_mean_nurse)]
level4_whole_wait=[c+d for c,d in zip(level4_mean_waiting_time, level2_mean_nurse)]
level5_whole_wait=[c+d for c,d in zip(level5_mean_waiting_time, level2_mean_nurse)]


def ci_and_pi(lis,level,k):
    data = [x for x in lis if not np.isnan(x)]  # 你的 list
    mean = np.mean(data)
    ci_low, ci_up = st.t.interval(0.95, len(data) - 1, loc=mean, scale=st.sem(data))
    print(f"{level} waiting time (include nurse time)Mean ={mean}, k={k}")
    print("95% CI: (", ci_low, ",", ci_up, ")")

    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)

    t_value = st.t.ppf(0.975, n - 1)  # 95% PI
    pi_low = mean - t_value * std * np.sqrt(1 + 1 / n)
    pi_up = mean + t_value * std * np.sqrt(1 + 1 / n)
    print("Prediction Interval (95%):", pi_low, pi_up)
    return ci_up, ci_low, pi_up, pi_low

ci_and_pi(level1_whole_wait,'ctas1',5)
ci_and_pi(level2_whole_wait,'ctas2',5)
ci_and_pi(level3_whole_wait,'ctas3',5)
ci_and_pi(level4_whole_wait,'ctas4',5)
ci_and_pi(level5_whole_wait,'ctas5',5)


def plot(CI_up, CI_down, PI_up, PI_down,xLabel, yLabel,title):

    x = np.arange(len(CI_up))

    CI_mean = [(up + down) / 2 for up, down in zip(CI_up, CI_down)]
    PI_mean = [(up + down) / 2 for up, down in zip(PI_up, PI_down)]

    CI_err = [(up - down) / 2 for up, down in zip(CI_up, CI_down)]
    PI_err = [(up - down) / 2 for up, down in zip(PI_up, PI_down)]

    plt.figure(figsize=(8, 5))

    plt.errorbar(x, CI_mean, yerr=CI_err, fmt='o', capsize=5, label="95% CI")

    plt.errorbar(x, PI_mean, yerr=PI_err, fmt='o', capsize=5, label="95% PI", alpha=0.5)
    labels = ["CTAS1", "CTAS2", "CTAS3", "CTAS4", "CTAS5"]
    plt.xticks(x, labels)

    plt.xlabel("CTAS Level")
    plt.ylabel(f"{yLabel}")
    plt.title(f"CI & PI {title} by CTAS Level k={k}")
    plt.legend()
    plt.show()

def interation(title):
    CI_up, CI_down, PI_up, PI_down = [], [], [], []

    level_lists = [level1_service_time,level2_service_time,level3_service_time,level4_service_time,level5_service_time]

    for i, level in enumerate(level_lists, start=1):
        ci_up, ci_down, pi_up, pi_down = ci_and_pi(level, f"ctas{i}", 5)
        CI_up.append(ci_up)
        CI_down.append(ci_down)
        PI_up.append(pi_up)
        PI_down.append(pi_down)

    plot(CI_up, CI_down, PI_up, PI_down,
         xLabel="CTAS Level", yLabel="Service time", title=title)



interation("Service time")


df2 = pd.read_csv('simulation_summary_k=2.csv')
doctor_iter2 = (df2["overall_average_waiting_time"]+df2["nurse_time"]).mean()

df3= pd.read_csv('simulation_summary_k=3.csv')
doctor_iter3 = (df3["overall_average_waiting_time"]+df3["nurse_time"]).mean()

df4 = pd.read_csv('simulation_summary_k=4.csv')
doctor_iter4=(df4["overall_average_waiting_time"]+df4["nurse_time"]).mean()

df5 = pd.read_csv('simulation_summary_k=5.csv')
doctor_iter5=(df5["overall_average_waiting_time"]+df5["nurse_time"]).mean()

k_list = [2, 3, 4, 5]
util_mean = [doctor_iter2, doctor_iter3, doctor_iter4, doctor_iter5]


plt.figure()
plt.plot(k_list, util_mean, marker='o')
plt.xlabel("Number of Doctors (k)")
plt.ylabel("Overall average waiting time")
plt.title("Average Overall average waiting time vs k")
plt.grid(True)
plt.show()





