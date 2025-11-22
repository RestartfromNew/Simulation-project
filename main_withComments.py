from socket import send_fds

import simpy
import random
import heapq
import numpy as np
import pandas as pd
k=5
from scipy.stats import gamma
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)
rng_np = np.random.default_rng(GLOBAL_SEED)
DOCTOR_NUM=k
SIMULATION_TIME=14*24*60
DAY_MINUTES=7*24*60
WARM_UP_TIME=2*24*60
MEAN_PATIENT_PER_HOUR=  [
    # average number of patients per hour of the day
    3.0, 2.8, 2.5, 2.2, 2.0, 2.3,
    3.0, 5.0, 7.5, 9.5, 10.5, 10.0,
    9.8, 9.2, 8.8, 7.8, 7.2, 6.6,
    6.0, 5.5, 4.8, 4.0, 3.5, 3.2
]

rates = np.array(MEAN_PATIENT_PER_HOUR)

# 估计 gamma shape(k) 和 scale(θ)
shape, loc, scale = gamma.fit(rates, floc=0)
print("shape k =", shape, "scale θ =", scale)
CTAS={"ctas1":0,"ctas2":15,"ctas3":30,"ctas4":60,"ctas5":120}
# maximum waiting time (minutes) for each CTAS level
CTAS_distribution={"ctas1":0.01,"ctas2":0.16,"ctas3":0.56,"ctas4":0.25,"ctas5":0.02}
CTAS_nurse={"ctas1":3.2,"ctas2":7.1,"ctas3":20.4,"ctas4":39.7,"ctas5":32.1}
ADMISSION_PROB = {
    "ctas1": 0.89,  # 例如 I: 8/9
    "ctas2": 0.65,  # II: 36/55
    "ctas3": 0.35,  # III:104/297
    "ctas4": 0.13,  # IV: 43/327
    "ctas5": 0.05   # V: 11/206
}
BED_WAIT_MEAN = {
    "ctas1": 60.0,
    "ctas2": 120.0,
    "ctas3": 180.0,
    "ctas4": 60.0,
    "ctas5": 30.0
}
# probability distribution of CTAS levels
CTAS1_queue=[]
CTAS2_queue=[]
CTAS3_queue=[]
departure_list=[]
patient_id=0

class Doctor:
    def __init__(self, env, id):
        self.proc=None
        self.env = env
        self.id = id
        self.status = 'idle'
        self.patient_level=None
        self.busy_time=0
        self.seen_number=0
        self.current_patient_id=None

    def start_service(self):
        return
    def end_service(self):
        return

class Patient:
    # Patient record state
    def __init__(self, env,doctors, id,arrive_time,ctas_level,rest_waiting_time,nurse_process_time):
        self.env = env
        self.id = id
        self.status = 'arrival'
        self.nurse_process_time = nurse_process_time
        # patient state: arrival, waiting, being served, depart
        self.arrival_time=arrive_time
        self.waiting_time=0
        self.total_time=0
        self.service_time=0
        self.departure_time=None
        self.bed_time=0
        self.ctas_level=ctas_level
        self.queue_rank=0
        self.doctors=doctors
        # queue_rank is based on the ratio of waiting_time/rest_waiting_time, higher is served earlier
        self.rest_waiting_time=rest_waiting_time
        self.proc = env.process(self.run())
    def print_required_parameters(self):
        print("rest_waiting_time:",self.rest_waiting_time)

    def __lt__(self, other):
        """heapq uses < for comparison"""
        return self.queue_rank > other.queue_rank

    def run(self):
        # Nurse stage (first processing stage)
        yield self.env.timeout(self.nurse_process_time)
        # After this moment, patient is ready to join the doctor queue
        self.status = "waiting"

        renew_waiting_rank(self)  # update queue rank after nurse time
        print(f"{self.id}: 进入队列的时间 {self.env.now}，到达时间{self.arrival_time}")
        insert_queue(self.ctas_level,self)
        if(self.ctas_level=="ctas1"):
            disruption(self.env.now,self,self.ctas_level,self.doctors)
        renew_queue(self.env.now)

ARRIVAL_SCV=1.708812612
#SCV, Squared Coefficient of Variation （表示到达的波动）
def gamma_params_with_SCV(mean_val,scv):
    # scv(quared Coefficient of Variation)= Var(A)/(E[A])^2
    scv = max(scv, 1e-9)
    alpha = 1.0 / scv
    #alpha:shape, E(x)=alpha * theta
    #theta:scale, Var(x)=alpha* power(theta,2)
    theta = mean_val / alpha
    return alpha, theta
    #two parameters of Gamma
def sample_interarrival(env_time):
    # 1. sample current arrival rate λ ~ Gamma(k, θ)
    mean_arrival = rng_np.gamma(shape=shape, scale=scale)
    mean_arrival = max(mean_arrival, 1e-6)

    # 2. 转换为平均 interarrival time (分钟)
    mean_interarrival = 60.0 / mean_arrival

    # 3. 如果需要带 SCV，用 gamma 生成真实间隔
    alpha, theta = gamma_params_with_SCV(mean_interarrival, ARRIVAL_SCV)
    interarrival_gap = rng_np.gamma(shape=alpha, scale=theta)

    return float(max(interarrival_gap, 1e-6))


ARRIVAL_RATE_PER_HOUR = 10
def sample_interarrival_constant(env_time):
    mean_interarrival = 60 / ARRIVAL_RATE_PER_HOUR  # 平均间隔 6 分钟
    alpha, theta = gamma_params_with_SCV(mean_interarrival, ARRIVAL_SCV)

    return float(max(rng_np.gamma(shape=alpha, scale=theta), 1e-6))
def sample_nurse_visit_time(env_time,patient_level):
    mean = CTAS_nurse[patient_level]
    lam = 1.0 / mean  # lambda = 1/mean
    return random.expovariate(lam)

def sample_level():
    # determine CTAS level using cumulative probability
    r = random.random()
    cum = 0
    for level, p in CTAS_distribution.items():
        cum += p
        if (r <= cum):
            return level
    return "ctas5"
def renew_queue(current_time):
    # update the rank of all patients when an event occurs
    for q in (CTAS1_queue, CTAS2_queue, CTAS3_queue):
        for p in q:
            p.waiting_time = current_time - p.arrival_time
            renew_waiting_rank(p)
        # re-heapify because ranks have changed
        heapq.heapify(q)

def renew_waiting_rank(patient):
    if patient.rest_waiting_time <= 0:
        # CTAS1 patients: assign a very large rank to always be prioritized
        patient.queue_rank = 1e9 + patient.waiting_time
    else:
        patient.queue_rank = patient.waiting_time / patient.rest_waiting_time
def insert_queue(level, patient):
    if level == "ctas1":
        heapq.heappush(CTAS1_queue, patient)
    elif level == "ctas2":
        heapq.heappush(CTAS2_queue,patient)
    else:
        heapq.heappush(CTAS3_queue, patient)
    print("patient arrived, patient id:", patient_id, " arrive time", patient.arrival_time)
def disruption(env_time,patient,level,doctors):
    if level == "ctas1":
        # for CTAS1, check if a doctor can be preempted
        candidate = None
        for d in doctors:
            if d.status == 'busy' and d.patient_level != "ctas1":
                if candidate is None:
                    candidate = d
                else:
                    # prefer to preempt a doctor treating CTAS3, then CTAS2
                    if d.patient_level > candidate.patient_level:
                        candidate = d

        if candidate is not None:
            print(f"⚠️ CTAS1 arrived, preempt Doctor{candidate.id} who is treating {candidate.patient_level}")
            # interrupt the doctor process
            candidate.proc.interrupt()

def generate_arrival(env,doctors):
    # generate a new patient arrival
    global patient_id
    last_report = 0
    hourly_count = 0
    while True:
        # generate interarrival time and move to the next arrival
        interarrival=sample_interarrival(env.now)
        #sample_interarrival通过每个小时的平均人数生成gamma的参数
        # interarrival = sample_interarrival_constant(env.now)
        #sample_interarrival_constant 的每小时平均人数是恒定的，通过svc的值来控制病人来的间隔
        yield env.timeout(interarrival)
        # generate random patient info
        patient_id=patient_id+1

        level=sample_level()
        nurse_process_time=sample_nurse_visit_time(env.now,level)
        rest_waiting_time=CTAS[level]
        current_time = env.now
        new_patient=Patient(env,doctors,patient_id,current_time,level,rest_waiting_time,nurse_process_time)
        # update queue ranks
        renew_queue(current_time)
        #
        # # put the patient into the queue
        # if level=="ctas1":
        #     heapq.heappush(CTAS1_queue,new_patient)
        # elif level=="ctas2":
        #     heapq.heappush(CTAS2_queue,new_patient)
        # else:
        #     heapq.heappush(CTAS3_queue,new_patient)
        # print("patient arrived, patient id:",patient_id," arrive time",new_patient.arrival_time)
        #
        # if level=="ctas1":
        #     # for CTAS1, check if a doctor can be preempted
        #     candidate = None
        #     for d in doctors:
        #         if d.status == 'busy' and d.patient_level != "ctas1":
        #             if candidate is None:
        #                 candidate = d
        #             else:
        #                 # prefer to preempt a doctor treating CTAS3, then CTAS2
        #                 if d.patient_level > candidate.patient_level:
        #                     candidate = d
        #
        #     if candidate is not None:
        #         print(f"⚠️ CTAS1 arrived, preempt Doctor{candidate.id} who is treating {candidate.patient_level}")
        #         # interrupt the doctor process
        #         candidate.proc.interrupt()

        if env.now - last_report >= 60:
            print(f"[t={env.now / 60:.1f}h] total arrivals: {patient_id}, "
                  f"CTAS1={len(CTAS1_queue)}, CTAS2={len(CTAS2_queue)}, CTAS3={len(CTAS3_queue)}")
            last_report = env.now
            hourly_count = 0

def sample_service_time(patient):
    # generate service time as exponential distribution
    base_time = {
        "ctas1": 73.6,
        "ctas2": 38.9,
        "ctas3": 26.3,
        "ctas4": 15.0,
        "ctas5": 10.9
    }[patient.ctas_level]
    service_scv_by_ctas = {
        "ctas1": 1.2,  # 高波动（重症情况差异大）
        "ctas2": 1.0,
        "ctas3": 0.8,
        "ctas4": 0.6,
        "ctas5": 0.4  # 简单病例较稳定
    }
    scv = service_scv_by_ctas[patient.ctas_level]
    # alpha = 2.0
    # beta = base_time / alpha
    alpha = 1.0 / scv  # shape 参数
    theta = base_time / alpha  # scale 参数
    service_time = random.gammavariate(alpha, theta)
    return max(service_time, 1e-6)




def find_next_patient() -> 'Patient | None':
    # select the next patient based on highest rank
    if(len(CTAS1_queue) > 0):
        patient=heapq.heappop(CTAS1_queue)
        print(f"Pick CTAS1 patient {patient.id}, rank={patient.queue_rank:.2f}, wait={patient.waiting_time:.1f}")
    elif(len(CTAS2_queue) > 0):
        patient=heapq.heappop(CTAS2_queue)
        print(f"Pick CTAS2 patient {patient.id}, rank={patient.queue_rank:.2f}, wait={patient.waiting_time:.1f}")
    elif(len(CTAS3_queue) > 0):
        patient=heapq.heappop(CTAS3_queue)
        print(f"Pick CTAS3 patient {patient.id}, rank={patient.queue_rank:.2f}, wait={patient.waiting_time:.1f}")
    # preemption logic not updated yet
    else:
        patient=None
    return patient

def doctor_process(env, doctor):
    while True:
        patient: 'Patient' = find_next_patient()
        if not patient:
            # no patients, wait 1 minute and check again
            yield env.timeout(1)
            continue
        # patient selected
        doctor.status='busy'
        doctor.current_patient_id=patient.id
        doctor.patient_level=patient.ctas_level
        patient.waiting_time=env.now-patient.arrival_time
        service_time = sample_service_time(patient)
        patient.status='being serviced'
        print(f"Doctor{doctor.id}, {doctor.current_patient_id} is now serviced, service_time:{service_time},start_time:{env.now}")
        start_time=env.now
        try:
            # normal completion
            yield env.timeout(service_time)
            # update doctor status
            base_departure_time = env.now
            current_time = env.now
            doctor.status='idle'
            doctor.busy_time+=service_time
            doctor.seen_number+=1
            doctor.current_patient_id=None
            doctor.patient_level = None
            patient.service_time = service_time

            # 2) 这里不再让系统时间前进，而是“逻辑上”加一个等床时间
            admitted = (random.random() < ADMISSION_PROB[patient.ctas_level])
            if admitted:
                print(f"{patient.ctas_level} is admitted")
                mean_wait = BED_WAIT_MEAN[patient.ctas_level]
                bed_wait = random.expovariate(1.0 / mean_wait)
            else:
                bed_wait = 0.0

            patient.bed_time = bed_wait

            # 3) 用 base_departure_time + bed_wait 来定义“逻辑上的离开时间”
            patient.departure_time = base_departure_time + bed_wait
            patient.total_time = patient.departure_time - patient.arrival_time
            patient.status = 'depart'

            departure_list.append(patient)

            print(f"finished doctor{doctor.id}, patient {patient.id}, "
                  f"decision_time={base_departure_time}, logical_departure={patient.departure_time}")

            # 如果你 renew_queue 想用真实 env.now，就用 base_departure_time 就可以
            renew_queue(base_departure_time)

            # update ranks
        except simpy.Interrupt:
            # preemption logic
            current_time = env.now
            served = env.now - start_time
            print(f"Interruption occurred, Doctor{doctor.id} interrupted after serving {served:.2f} minutes "
                  f"for patient {patient.id} (level {patient.ctas_level})")
            doctor.busy_time += served  # 医生确实忙了这一段
            patient.service_time += served
            renew_queue(current_time)
            patient.status = 'waiting'
            patient.service_time += served  # partial service time recorded
            renew_waiting_rank(patient)
            # reward interrupted patient with a bonus rank
            patient.queue_rank+=1
            # put the patient back into its CTAS queue
            if patient.ctas_level == "ctas1":
                heapq.heappush(CTAS1_queue, patient)
            elif patient.ctas_level == "ctas2":
                heapq.heappush(CTAS2_queue, patient)
            else:
                heapq.heappush(CTAS3_queue, patient)

            doctor.status = 'idle'
            doctor.current_patient_id = None
            doctor.patient_level = None
            # next loop will prioritize CTAS1 patients


def run_simulation(seeds):
    global GLOBAL_SEED, rng_np
    GLOBAL_SEED = seeds
    random.seed(GLOBAL_SEED)
    rng_np = np.random.default_rng(GLOBAL_SEED)
    env = simpy.Environment()
    doctors=[Doctor(env,f'Doctor{i}') for i in range(0,DOCTOR_NUM)]
    env.process(generate_arrival(env, doctors))
    for doctor in doctors:
        # create k doctors running in parallel
        doctor.proc = env.process(doctor_process(env, doctor))
    print(f"---- Warm-up for {WARM_UP_TIME} minutes ----")
    env.run(until=WARM_UP_TIME)

    # 预热完：保留当前队列和医生状态，只清统计量
    departure_list.clear()
    for d in doctors:
        d.busy_time = 0.0
        d.seen_number = 0

    print("---- Start official simulation period ----")

    # ========= 5. 正式统计阶段 =========
    # 注意：这里跑到 warm_up + SIMULATION_TIME
    env.run(until=WARM_UP_TIME + SIMULATION_TIME)

    for i in departure_list:
        print(f"id{i.id},arrive time{i.arrival_time},departure_time:{i.departure_time},service_time:{i.service_time},waiting_time:{i.waiting_time}")


    data = [{
        "id": p.id,
        "arrival_time": p.arrival_time,
        "service_time": p.service_time,
        "waiting_time": p.waiting_time,
        "departure_time": p.departure_time,
        "ctas_level": p.ctas_level,
        "nurse_time": p.nurse_process_time,
    } for p in departure_list]

    df = pd.DataFrame(data)
    print("Average service time:", df["service_time"].mean())
    print("Average waiting time:", df["waiting_time"].mean())
    print("Average total system time:", df["departure_time"].sub(df["arrival_time"]).mean())
    Patient_number=len(departure_list)
    print(f"总计{Patient_number}个病人")

    data = [{
        "id": p.id,
        "arrival_time": p.arrival_time,
        "service_time": p.service_time,
        "waiting_time": p.waiting_time,
        "departure_time": p.departure_time,
        "bed_time":p.bed_time,
        "ctas_level": p.ctas_level,
        "whole_time":p.departure_time-p.arrival_time,
        "nurse_time":p.nurse_process_time,
    } for p in departure_list]

    data_doctor=[{
    "id": d.id,
        "busy time": d.busy_time,
        "utilization":d.busy_time/SIMULATION_TIME
    }for d in doctors]
    ddf=pd.DataFrame(data_doctor)
    df = pd.DataFrame(data)
    print("到达率：",Patient_number/SIMULATION_TIME)
    print("平均服务时间：", df["service_time"].mean())
    print("平均等待时间：", df["waiting_time"].mean())
    print("平均总停留时间：", df["departure_time"].sub(df["arrival_time"]).mean())

    for d in doctors:
        print(f"{d.id}的服务总时间是{d.busy_time},利用率为{d.busy_time/SIMULATION_TIME}")

    # average times grouped by CTAS level
    print(df.groupby("ctas_level")[["waiting_time", "service_time","whole_time","nurse_time","bed_time"]].mean())

    simulation_resut={
    "seed": GLOBAL_SEED,
    "average_service_time": float(df["service_time"].mean()),
    "average_waiting_time": float(df["waiting_time"].mean()),
    "average_system_time": float(df["departure_time"].sub(df["arrival_time"]).mean()),
    "doctor_mean_service_time": float(ddf["busy time"].mean()),
    "doctor_utilization": float(ddf["utilization"].mean())
}





    import matplotlib.pyplot as plt

    # 1. distribution of service time
    plt.hist(df["service_time"], bins=30)
    plt.xlabel("Service Time (minutes)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Service Time")
    plt.show()

    # 2. distribution of waiting time
    plt.hist(df["waiting_time"], bins=30)
    plt.xlabel("Waiting Time (minutes)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Waiting Time")
    plt.show()

    # 3. scatter plot of arrival time vs service time
    plt.scatter(df["arrival_time"], df["service_time"], alpha=0.6)
    plt.xlabel("Arrival Time (minutes)")
    plt.ylabel("Service Time (minutes)")
    plt.title("Arrival Time vs Service Time")
    plt.show()

    # 4. rolling average waiting time over time (to observe congestion)
    df_sorted = df.sort_values("arrival_time")
    window = 50
    rolling_mean = df_sorted["waiting_time"].rolling(window=window).mean()
    plt.plot(df_sorted["arrival_time"], rolling_mean)
    plt.xlabel("Arrival Time (minutes)")
    plt.ylabel(f"Rolling Mean of Waiting Time (window={window})")
    plt.title("Trend of Waiting Time over Time")
    plt.show()

    # 5. Arrivals per hour plot
    arrivals = df["arrival_time"].copy()
    hour = (arrivals // 60).astype(int)
    arrivals_by_hour = hour.value_counts().sort_index()

    plt.figure()
    arrivals_by_hour.plot(kind="bar")
    plt.xlabel("Hour of Day")
    plt.ylabel("Arrivals")
    plt.title("Arrivals per Hour")
    plt.tight_layout(); plt.show()

    # 6. Average witing time with 95% CI interval
    def mean_ci(x, alpha=0.05):
        x = np.asarray(x.dropna())
        m = x.mean()
        se = x.std(ddof=1)/np.sqrt(len(x))
        z = 1.96
        return m, m - z*se, m + z*se

    g = df.groupby("ctas_level")["waiting_time"].apply(mean_ci)
    means  = g.apply(lambda t: t[0])
    lower  = g.apply(lambda t: t[1])
    upper  = g.apply(lambda t: t[2])
    err    = means - lower

    order = ["ctas1","ctas2","ctas3","ctas4","ctas5"]
    means = means.reindex(order); err = err.reindex(order)

    plt.figure()
    plt.bar(means.index, means.values, yerr=err.values, capsize=4)
    plt.xlabel("CTAS Level"); plt.ylabel("Average Waiting (min)")
    plt.title("Average Waiting by CTAS (95% CI)")
    plt.tight_layout(); plt.show()
    return simulation_resut

run_simulation(141)