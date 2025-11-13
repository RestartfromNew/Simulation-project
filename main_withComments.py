import simpy
import random
import heapq
import numpy as np
import pandas as pd
k=5
GLOBAL_SEED = 141
random.seed(GLOBAL_SEED)
rng_np = np.random.default_rng(GLOBAL_SEED)
DOCTOR_NUM=k
SIMULATION_TIME=7*24*60
DAY_MINUTES=24*60
MEAN_PATIENT_PER_HOUR=  [
    # average number of patients per hour of the day
    2, 1, 1, 1,   # 0-3
    2, 3, 5, 7,   # 4-7
    10, 12, 14, 15,  # 8-11
    15, 14, 12, 10,  # 12-15
    8, 6, 4, 3,      # 16-19
    3, 2, 2, 2
]
CTAS={"ctas1":0,"ctas2":15,"ctas3":30,"ctas4":60,"ctas5":120}
# maximum waiting time (minutes) for each CTAS level
CTAS_distribution={"ctas1":0.035,"ctas2":0.177,"ctas3":0.418,"ctas4":0.339,"ctas5":0.031}
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
    def __init__(self, env, id,arrive_time,ctas_level,rest_waiting_time):
        self.env = env
        self.id = id
        self.status = 'arrival'
        # patient state: arrival, waiting, being served, depart
        self.arrival_time=arrive_time
        self.waiting_time=0
        self.total_time=0
        self.service_time=0
        self.departure_time=None
        self.ctas_level=ctas_level
        self.queue_rank=0
        # queue_rank is based on the ratio of waiting_time/rest_waiting_time, higher is served earlier
        self.rest_waiting_time=rest_waiting_time
    def print_required_parameters(self):
        print("rest_waiting_time:",self.rest_waiting_time)

    def __lt__(self, other):
        """heapq uses < for comparison"""
        return self.queue_rank > other.queue_rank

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
    # generate the interarrival time of the next patient
    t_in_day = env_time % DAY_MINUTES
    hour = int(t_in_day // 60)
    mean_arrival = MEAN_PATIENT_PER_HOUR[hour]
    if mean_arrival <= 0:
        return 60

    mean_interarrival = 60 / mean_arrival
    alpha, theta = gamma_params_with_SCV(mean_interarrival, ARRIVAL_SCV)

    interarrival_gap=rng_np.gamma(shape=alpha, scale=theta)
    # return random.expovariate(1.0 / mean_interarrival)
    return float(max(interarrival_gap, 1e-6))

ARRIVAL_RATE_PER_HOUR = 10
def sample_interarrival_constant(env_time):
    mean_interarrival = 60 / ARRIVAL_RATE_PER_HOUR  # 平均间隔 6 分钟
    alpha, theta = gamma_params_with_SCV(mean_interarrival, ARRIVAL_SCV)

    return float(max(rng_np.gamma(shape=alpha, scale=theta), 1e-6))

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
        current_time = env.now
        level=sample_level()
        rest_waiting_time=CTAS[level]
        new_patient=Patient(env,patient_id,current_time,level,rest_waiting_time)
        # update queue ranks
        renew_queue(current_time)

        # put the patient into the queue
        if level=="ctas1":
            heapq.heappush(CTAS1_queue,new_patient)
        elif level=="ctas2":
            heapq.heappush(CTAS2_queue,new_patient)
        else:
            heapq.heappush(CTAS3_queue,new_patient)
        print("patient arrived, patient id:",patient_id," arrive time",new_patient.arrival_time)

        if level=="ctas1":
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

        if env.now - last_report >= 60:
            print(f"[t={env.now / 60:.1f}h] total arrivals: {patient_id}, "
                  f"CTAS1={len(CTAS1_queue)}, CTAS2={len(CTAS2_queue)}, CTAS3={len(CTAS3_queue)}")
            last_report = env.now
            hourly_count = 0

def sample_service_time(patient):
    # generate service time as exponential distribution
    base_time = {
        "ctas1": 60,
        "ctas2": 45,
        "ctas3": 30,
        "ctas4": 20,
        "ctas5": 15
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
            current_time = env.now
            doctor.status='idle'
            doctor.busy_time+=service_time
            doctor.seen_number+=1
            doctor.current_patient_id=None
            doctor.patient_level = None
            # update patient status
            patient.service_time=service_time
            patient.total_time=env.now-patient.arrival_time
            patient.departure_time=current_time
            patient.status='depart'
            # record patient info for statistics
            departure_list.append(patient)
            print(f"finished doctor{doctor.id}, patient departure_time:{patient.departure_time}")
            renew_queue(current_time)
            # update ranks
        except simpy.Interrupt:
            # preemption logic
            current_time = env.now
            served = env.now - start_time
            print(f"Interruption occurred, Doctor{doctor.id} interrupted after serving {served:.2f} minutes "
                  f"for patient {patient.id} (level {patient.ctas_level})")

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

env=simpy.Environment()
doctors=[Doctor(env,f'Doctor{i}') for i in range(0,DOCTOR_NUM)]
env.process(generate_arrival(env, doctors))
for doctor in doctors:
    # create k doctors running in parallel
    doctor.proc = env.process(doctor_process(env, doctor))
env.run(until=SIMULATION_TIME)

for i in departure_list:
    print(f"id{i.id},arrive time{i.arrival_time},departure_time:{i.departure_time},service_time:{i.service_time},waiting_time:{i.waiting_time}")


data = [{
    "id": p.id,
    "arrival_time": p.arrival_time,
    "service_time": p.service_time,
    "waiting_time": p.waiting_time,
    "departure_time": p.departure_time,
    "ctas_level": p.ctas_level
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
    "ctas_level": p.ctas_level
} for p in departure_list]

data_doctor=[{
"id": d.id,
    "busy time": d.busy_time,
}for d in doctors]

df = pd.DataFrame(data)
print("到达率：",Patient_number/SIMULATION_TIME)
print("平均服务时间：", df["service_time"].mean())
print("平均等待时间：", df["waiting_time"].mean())
print("平均总停留时间：", df["departure_time"].sub(df["arrival_time"]).mean())

for d in doctors:
    print(f"{d.id}的服务总时间是{d.busy_time},利用率为{d.busy_time/SIMULATION_TIME}")

# average times grouped by CTAS level
print(df.groupby("ctas_level")[["waiting_time", "service_time"]].mean())
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
