import simpy
import random
import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt

# ================== 全局参数 ==================
k = 2  # 医生数量
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)
rng_np = np.random.default_rng(GLOBAL_SEED)

DOCTOR_NUM = k
SIMULATION_TIME = 14 * 24 * 60   # 正式统计阶段长度（分钟）
WARM_UP_TIME = 2 * 24 * 60       # 预热长度（分钟）

# 原来的每小时平均到达数，用来拟合 Gamma 分布，保留
MEAN_PATIENT_PER_HOUR = [
    3.0, 2.8, 2.5, 2.2, 2.0, 2.3,
    3.0, 5.0, 7.5, 9.5, 10.5, 10.0,
    9.8, 9.2, 8.8, 7.8, 7.2, 6.6,
    6.0, 5.5, 4.8, 4.0, 3.5, 3.2
]

rates = np.array(MEAN_PATIENT_PER_HOUR)
# 拟合 Γ 分布：用于生成“每小时平均到达率 λ”的随机波动
shape, loc, scale = gamma.fit(rates, floc=0)
print("shape k =", shape, "scale θ =", scale)

# -------- 这里开始就不再用任何“按等级取参数”的字典 --------

# 等级分布：只用来随机生成标签 (ctas1~ctas5)，不影响机制
LEVEL_DISTRIBUTION = {
    "ctas1": 0.01,
    "ctas2": 0.16,
    "ctas3": 0.56,
    "ctas4": 0.25,
    "ctas5": 0.02
}

# 把原来不同等级的平均值做一个统一的“整体平均”，你也可以自己改成想要的值：
# 这里做一个简单加权平均（一次性算出几个全局常数）
_ctas_nurse = {"ctas1": 3.2, "ctas2": 7.1, "ctas3": 20.4, "ctas4": 39.7, "ctas5": 32.1}
_ctas_service = {"ctas1": 73.6, "ctas2": 38.9, "ctas3": 26.3, "ctas4": 15.0, "ctas5": 10.9}
_ctas_adm = {"ctas1": 0.89, "ctas2": 0.65, "ctas3": 0.35, "ctas4": 0.13, "ctas5": 0.05}
_ctas_bed = {"ctas1": 60.0, "ctas2": 120.0, "ctas3": 180.0, "ctas4": 60.0, "ctas5": 30.0}

def weighted_avg(table):
    return sum(LEVEL_DISTRIBUTION[l] * table[l] for l in LEVEL_DISTRIBUTION.keys())

NURSE_MEAN = weighted_avg(_ctas_nurse)      # 统一的 nurse 平均时间
SERVICE_MEAN = weighted_avg(_ctas_service)  # 统一的 service 平均时间
ADMISSION_MEAN_PROB = weighted_avg(_ctas_adm)  # 统一入院概率
BED_WAIT_MEAN = weighted_avg(_ctas_bed)     # 统一床位等待时间均值

print(f"统一 NURSE_MEAN={NURSE_MEAN:.2f}, SERVICE_MEAN={SERVICE_MEAN:.2f}, "
      f"ADMISSION_PROB={ADMISSION_MEAN_PROB:.2f}, BED_WAIT_MEAN={BED_WAIT_MEAN:.2f}")

# === 队列和统计 ===
QUEUE = []          # FCFS 队列
departure_list = [] # 存储已经“离开系统”的患者（含床等时间）
patient_id = 0

# ================== 类定义 ==================
class Doctor:
    def __init__(self, env, id):
        self.proc = None
        self.env = env
        self.id = id
        self.status = 'idle'
        self.busy_time = 0.0
        self.seen_number = 0
        self.current_patient_id = None


class Patient:
    # Patient record state
    def __init__(self, env, id, arrive_time, level_label, nurse_process_time):
        self.env = env
        self.id = id
        self.status = 'arrival'

        self.level = level_label      # 只是一个标签，不参与逻辑
        self.nurse_process_time = nurse_process_time

        self.arrival_time = arrive_time
        self.waiting_time = 0.0
        self.total_time = 0.0
        self.service_time = 0.0
        self.departure_time = None
        self.bed_time = 0.0

        # 启动自己的过程：先 nurse，再入队
        self.proc = env.process(self.run())

    def run(self):
        # 1. triage / nurse 时间
        yield self.env.timeout(self.nurse_process_time)

        # 2. triage 完成，进入医生队列（FCFS）
        self.status = "waiting"
        print(f"Patient {self.id}: triage done at {self.env.now:.2f}, arrival at {self.arrival_time:.2f}, level={self.level}")
        QUEUE.append(self)


# ================== 随机函数 ==================
ARRIVAL_SCV = 1.708812612  # 到达间隔 SCV

def gamma_params_with_SCV(mean_val, scv):
    scv = max(scv, 1e-9)
    alpha = 1.0 / scv
    theta = mean_val / alpha
    return alpha, theta

def sample_interarrival(env_time):
    # 先随机一个“本小时平均患者率 λ” ~ Gamma(shape, scale)
    mean_arrival = rng_np.gamma(shape=shape, scale=scale)
    mean_arrival = max(mean_arrival, 1e-6)

    mean_interarrival = 60.0 / mean_arrival  # 分钟
    alpha, theta = gamma_params_with_SCV(mean_interarrival, ARRIVAL_SCV)
    interarrival_gap = rng_np.gamma(shape=alpha, scale=theta)
    return float(max(interarrival_gap, 1e-6))

def sample_level():
    # 按 LEVEL_DISTRIBUTION 的概率随机给一个标签，但不影响行为
    r = random.random()
    cum = 0.0
    for level, p in LEVEL_DISTRIBUTION.items():
        cum += p
        if r <= cum:
            return level
    return "ctas5"

def sample_nurse_visit_time():
    # 所有等级使用同一分布：指数，均值 NURSE_MEAN
    lam = 1.0 / NURSE_MEAN
    return random.expovariate(lam)

def sample_service_time():
    # 所有等级使用同一分布：Gamma，均值 SERVICE_MEAN
    scv = 0.8  # 给一个统一 SCV，随便定一个合理值
    alpha = 1.0 / scv
    theta = SERVICE_MEAN / alpha
    service_time = random.gammavariate(alpha, theta)
    return max(service_time, 1e-6)


# ================== 过程函数 ==================
def generate_arrival(env):
    global patient_id
    last_report = 0.0

    while True:
        interarrival = sample_interarrival(env.now)
        yield env.timeout(interarrival)

        patient_id += 1
        level = sample_level()
        nurse_process_time = sample_nurse_visit_time()
        current_time = env.now

        Patient(env, patient_id, current_time, level, nurse_process_time)

        if env.now - last_report >= 60:
            print(f"[t={env.now / 60:.1f}h] total arrivals: {patient_id}, "
                  f"queue length={len(QUEUE)}")
            last_report = env.now

def find_next_patient() -> 'Patient | None':
    # FCFS：从队列头部弹出
    if len(QUEUE) > 0:
        patient = QUEUE.pop(0)
        print(f"Pick patient {patient.id}, level={patient.level}, "
              f"wait so far={patient.waiting_time:.1f}")
        return patient
    else:
        return None

def doctor_process(env, doctor):
    while True:
        patient = find_next_patient()
        if not patient:
            # 没病人，医生等 1 分钟再看
            yield env.timeout(1.0)
            continue

        doctor.status = 'busy'
        doctor.current_patient_id = patient.id

        # 记录 waiting time = 开始服务时刻 - arrival_time
        patient.waiting_time = env.now - patient.arrival_time

        service_time = sample_service_time()
        patient.status = 'being serviced'
        print(f"Doctor {doctor.id}: start servicing patient {patient.id}, "
              f"service_time={service_time:.2f}, start_time={env.now:.2f}, level={patient.level}")

        yield env.timeout(service_time)

        base_departure_time = env.now

        # 更新医生状态
        doctor.status = 'idle'
        doctor.busy_time += service_time
        doctor.seen_number += 1
        doctor.current_patient_id = None

        # 记录医生服务时间
        patient.service_time = service_time

        # 模拟入院和床位等待：所有人使用同一个概率 & 同一个 bed wait 分布
        admitted = (random.random() < ADMISSION_MEAN_PROB)
        if admitted:
            print(f"Patient {patient.id} (level={patient.level}) is admitted")
            bed_wait = random.expovariate(1.0 / BED_WAIT_MEAN)
        else:
            bed_wait = 0.0

        patient.bed_time = bed_wait
        patient.departure_time = base_departure_time + bed_wait
        patient.total_time = patient.departure_time - patient.arrival_time
        patient.status = 'depart'

        departure_list.append(patient)

        print(f"Doctor {doctor.id}: finished patient {patient.id}, "
              f"decision_time={base_departure_time:.2f}, "
              f"logical_departure={patient.departure_time:.2f}, level={patient.level}")


# ================== 主模拟函数 ==================
def run_simulation(seeds):
    global GLOBAL_SEED, rng_np, QUEUE, departure_list, patient_id

    GLOBAL_SEED = seeds
    random.seed(GLOBAL_SEED)
    rng_np = np.random.default_rng(GLOBAL_SEED)

    # 初始化队列和统计
    QUEUE = []
    departure_list = []
    patient_id = 0

    env = simpy.Environment()
    doctors = [Doctor(env, f'Doctor{i}') for i in range(DOCTOR_NUM)]

    env.process(generate_arrival(env))
    for doctor in doctors:
        doctor.proc = env.process(doctor_process(env, doctor))

    # ---------- Warm-up ----------
    print(f"---- Warm-up for {WARM_UP_TIME} minutes ----")
    env.run(until=WARM_UP_TIME)

    # 清空统计（但不清空队列和系统状态）
    departure_list.clear()
    for d in doctors:
        d.busy_time = 0.0
        d.seen_number = 0

    print("---- Start official simulation period ----")
    env.run(until=WARM_UP_TIME + SIMULATION_TIME)

    # ========== 结果统计 ==========
    for p in departure_list:
        print(f"id{p.id}, arrive={p.arrival_time:.2f}, "
              f"depart={p.departure_time:.2f}, "
              f"service={p.service_time:.2f}, "
              f"wait={p.waiting_time:.2f}, level={p.level}")

    data = [{
        "id": p.id,
        "arrival_time": p.arrival_time,
        "service_time": p.service_time,
        "waiting_time": p.waiting_time,
        "departure_time": p.departure_time,
        "bed_time": p.bed_time,
        "whole_time": p.departure_time - p.arrival_time,
        "nurse_time": p.nurse_process_time,
        "level": p.level,   # 只是标签
    } for p in departure_list]

    df = pd.DataFrame(data)
    Patient_number = len(departure_list)
    print(f"总计 {Patient_number} 个病人")

    # 医生统计
    data_doctor = [{
        "id": d.id,
        "busy time": d.busy_time,
        "utilization": d.busy_time / SIMULATION_TIME
    } for d in doctors]
    ddf = pd.DataFrame(data_doctor)

    print("到达率：", Patient_number / SIMULATION_TIME)
    print("平均服务时间：", df["service_time"].mean())
    print("平均等待时间：", df["waiting_time"].mean())
    print("平均总停留时间：", df["departure_time"].sub(df["arrival_time"]).mean())

    for d in doctors:
        print(f"{d.id} 的服务总时间是 {d.busy_time:.2f}, 利用率为 {d.busy_time / SIMULATION_TIME:.3f}")

    # 不再按等级分组，只看整体均值
    print("整体平均：")
    print(df[["waiting_time", "service_time", "whole_time", "nurse_time", "bed_time"]].mean())

    print(df.groupby("level")[["waiting_time", "service_time", "whole_time", "nurse_time", "bed_time"]].mean())

    simulation_resut = {
        "seed": GLOBAL_SEED,
        "average_service_time": float(df["service_time"].mean()),
        "average_waiting_time": float(df["waiting_time"].mean()),
        "average_system_time": float(df["departure_time"].sub(df["arrival_time"]).mean()),
        "doctor_mean_service_time": float(ddf["busy time"].mean()),
        "doctor_utilization": float(ddf["utilization"].mean())
    }

    # ========== 画图 ==========
    # 1. Service time 分布
    plt.hist(df["service_time"], bins=30)
    plt.xlabel("Service Time (minutes)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Service Time")
    plt.show()

    # 2. Waiting time 分布
    plt.hist(df["waiting_time"], bins=30)
    plt.xlabel("Waiting Time (minutes)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Waiting Time")
    plt.show()

    # 3. arrival vs service
    plt.scatter(df["arrival_time"], df["service_time"], alpha=0.6)
    plt.xlabel("Arrival Time (minutes)")
    plt.ylabel("Service Time (minutes)")
    plt.title("Arrival Time vs Service Time")
    plt.show()

    # 4. rolling average waiting time
    df_sorted = df.sort_values("arrival_time")
    window = 50
    rolling_mean = df_sorted["waiting_time"].rolling(window=window).mean()
    plt.plot(df_sorted["arrival_time"], rolling_mean)
    plt.xlabel("Arrival Time (minutes)")
    plt.ylabel(f"Rolling Mean of Waiting Time (window={window})")
    plt.title("Trend of Waiting Time over Time")
    plt.show()

    # 5. Arrivals per hour (approx)
    arrivals = df["arrival_time"].copy()
    hour = (arrivals // 60).astype(int)
    arrivals_by_hour = hour.value_counts().sort_index()

    plt.figure()
    arrivals_by_hour.plot(kind="bar")
    plt.xlabel("Hour Index (since warm-up)")
    plt.ylabel("Arrivals")
    plt.title("Arrivals per Hour (simulated)")
    plt.tight_layout()
    plt.show()

    return simulation_resut


# 运行一次
run_simulation(141)
