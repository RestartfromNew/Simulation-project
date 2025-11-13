
import simpy
import random
import heapq
import pandas as pd
k=5

DOCTOR_NUM=k
SIMULATION_TIME=7*24*60
DAY_MINUTES=24*60
MEAN_PATIENT_PER_HOUR=  [
    #每天每个时间段的平均病人数量
    2, 1, 1, 1,   # 0-3
    2, 3, 5, 7,   # 4-7
    10, 12, 14, 15,  # 8-11
    15, 14, 12, 10,  # 12-15
    8, 6, 4, 3,      # 16-19
    3, 2, 2, 2
]
CTAS={"ctas1":0,"ctas2":15,"ctas3":30,"ctas4":60,"ctas5":120}
#每个等级病人最长等待时间
CTAS_distribution={"ctas1":0.035,"ctas2":0.177,"ctas3":0.418,"ctas4":0.339,"ctas5":0.031}
#每个等级病人的概率
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
    #Patient记录状态
    def __init__(self, env, id,arrive_time,ctas_level,rest_waiting_time):
        self.env = env
        self.id = id
        self.status = 'arrival'
        #patient state: arrival, waiting, being served,depart
        self.arrival_time=arrive_time
        self.waiting_time=0
        self.total_time=0
        self.service_time=0
        self.departure_time=None
        self.ctas_level=ctas_level
        self.queue_rank=0
        #queue_rank按照waiting time/ rest waiting time 的比例进行排队，越大的越会被先选到
        self.rest_waiting_time=rest_waiting_time
    def print_required_parameters(self):
        print("rest_waiting_time:",self.rest_waiting_time)

    def __lt__(self, other):
        """heapq 需要用 < 比较"""
        return self.queue_rank > other.queue_rank

def sample_interarrival(env_time):
    #产生下一个病人的到达间隔
    t_in_day = env_time % DAY_MINUTES
    hour = int(t_in_day // 60)
    mean_arrival = MEAN_PATIENT_PER_HOUR[hour]
    if mean_arrival <= 0:
        return 60
    mean_interarrival = 60 / mean_arrival
    return random.expovariate(1.0 / mean_interarrival)
def sample_level():
    #用累计法确定他的等级
    r = random.random()
    cum = 0
    for level, p in CTAS_distribution.items():
        cum += p
        if (r <= cum):
            return level
    return "ctas5"
def renew_queue(current_time):
    #在事件发生的时候，更新所有病人的rank
    for q in (CTAS1_queue, CTAS2_queue, CTAS3_queue):
        for p in q:
            p.waiting_time = current_time - p.arrival_time
            renew_waiting_rank(p)
        # 因为内部的 rank 变了，需要重新堆化
        heapq.heapify(q)

def renew_waiting_rank(patient):
    if patient.rest_waiting_time <= 0:
        # 1级病人：rank 直接给一个很大的值，永远优先
        patient.queue_rank = 1e9 + patient.waiting_time
    else:
        patient.queue_rank = patient.waiting_time / patient.rest_waiting_time

def generate_arrival(env,doctors):
    #产生一个新病人
    global patient_id
    last_report = 0
    hourly_count = 0
    while True:
        #产生一个间隔时间，推进到下一个arrive
        interarrival=sample_interarrival(env.now)
        yield env.timeout(interarrival)
        #获取随机生成的病人信息
        patient_id=patient_id+1
        current_time = env.now
        level=sample_level()
        rest_waiting_time=CTAS[level]
        new_patient=Patient(env,patient_id,current_time,level,rest_waiting_time)
        #更新队列rank
        renew_queue(current_time)

        #把病人放入队列

        if level=="ctas1":
            heapq.heappush(CTAS1_queue,new_patient)
        elif level=="ctas2":
            heapq.heappush(CTAS2_queue,new_patient)
        else:
            heapq.heappush(CTAS3_queue,new_patient)
        print("patient arrived, patient id:",patient_id," arrive time",new_patient.arrival_time)

        if level=="ctas1":
            #如果是高等级病人，立刻查询医生的服务病人等级
            candidate = None
            for d in doctors:
                if d.status == 'busy' and d.patient_level != "ctas1":
                    if candidate is None:
                        candidate = d
                    else:
                        # 选正在看 CTAS3 的优先，其次 CTAS2
                        if d.patient_level > candidate.patient_level:
                            candidate = d

            if candidate is not None:
                print(f"⚠️ CTAS1 arrived, preempt Doctor{candidate.id} who is treating {candidate.patient_level}")
                #发出打断，抢占医生的服务
                candidate.proc.interrupt()

        if env.now - last_report >= 60:
            print(f"[t={env.now / 60:.1f}h] total arrivals: {patient_id}, "
                  f"CTAS1={len(CTAS1_queue)}, CTAS2={len(CTAS2_queue)}, CTAS3={len(CTAS3_queue)}")
            last_report = env.now
            hourly_count = 0

def sample_service_time(patient):
    #产生服务时间，是指数分布
    base_time = {
        "ctas1": 60,
        "ctas2": 45,
        "ctas3": 30,
        "ctas4": 20,
        "ctas5": 15
    }[patient.ctas_level]
    return random.expovariate(1.0 / base_time)




def find_next_patient() -> 'Patient | None':
    #选取下一个病人
    #按照rank高的病人选取
    if(len(CTAS1_queue) > 0):
        patient=heapq.heappop(CTAS1_queue)
        print(f"Pick CTAS1 patient {patient.id}, rank={patient.queue_rank:.2f}, wait={patient.waiting_time:.1f}")
    elif(len(CTAS2_queue) > 0):
        patient=heapq.heappop(CTAS2_queue)
        print(f"Pick CTAS2 patient {patient.id}, rank={patient.queue_rank:.2f}, wait={patient.waiting_time:.1f}")
    elif(len(CTAS3_queue) > 0):
        patient=heapq.heappop(CTAS3_queue)
        print(f"Pick CTAS3 patient {patient.id}, rank={patient.queue_rank:.2f}, wait={patient.waiting_time:.1f}")
    #暂时没有更新抢占逻辑
    else:
        patient=None
    return patient

def doctor_process(env, doctor):
    while True:
        patient: 'Patient' = find_next_patient()
        if not patient:
            # 没病人就等 1 分钟再看
            yield env.timeout(1)
            continue
        #选取病人
        doctor.status='busy'
        doctor.current_patient_id=patient.id
        doctor.patient_level=patient.ctas_level
        patient.waiting_time=env.now-patient.arrival_time
        service_time = sample_service_time(patient)
        patient.status='being serviced'
        print(f"Doctor{doctor.id}, {doctor.current_patient_id} is now serviced, service_time:{service_time},start_time:{env.now}")
        start_time=env.now
        try:
            #如果服务正常完成
            yield env.timeout(service_time)
            #更新医生状态
            current_time = env.now
            doctor.status='idle'
            doctor.busy_time+=service_time
            doctor.seen_number+=1
            doctor.current_patient_id=None
            doctor.patient_level = None
            #更新病人状态
            patient.service_time=service_time
            patient.total_time=env.now-patient.arrival_time
            patient.departure_time=current_time
            patient.status='depart'
            #病人信息放入depart list方便统计
            departure_list.append(patient)
            print(f"finished doctor{doctor.id}, patient departure_time:{patient.departure_time}")
            renew_queue(current_time)
            #更新rank
        except simpy.Interrupt:
            #抢占逻辑
            current_time = env.now
            served = env.now - start_time
            print(f"打断发生，Doctor{doctor.id} interrupted after serving {served:.2f} minutes "
                  f"for patient {patient.id} (level {patient.ctas_level})")

            renew_queue(current_time)
            patient.status = 'waiting'
            patient.service_time += served  # 如果想记一下已经做了多少
            renew_waiting_rank(patient)
            #给被打断病人补偿
            patient.queue_rank+=1
            # 丢回它原来的 CTAS 队列
            if patient.ctas_level == "ctas1":
                heapq.heappush(CTAS1_queue, patient)
            elif patient.ctas_level == "ctas2":
                heapq.heappush(CTAS2_queue, patient)
            else:
                heapq.heappush(CTAS3_queue, patient)

            doctor.status = 'idle'
            doctor.current_patient_id = None
            doctor.patient_level = None
            #完成释放逻辑后，下一轮循环优先选择1等级病人

env=simpy.Environment()
doctors=[Doctor(env,f'Doctor{i}') for i in range(0,DOCTOR_NUM)]
env.process(generate_arrival(env, doctors))
for doctor in doctors:
    #生成k个医生并行运行
    doctor.proc = env.process(doctor_process(env, doctor))
env.run(until=SIMULATION_TIME)

for i in departure_list:
    print(f"id{i.id},arrive time{i.arrival_time},departure_time:{i.departure_time},service_time:{i.service_time},waiting_time:{i.waiting_time}")

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

# 按 CTAS 等级分组的平均时间
print(df.groupby("ctas_level")[["waiting_time", "service_time"]].mean())
import matplotlib.pyplot as plt

# 1. 服务时间分布
plt.hist(df["service_time"], bins=30)
plt.xlabel("Service Time (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Service Time")
plt.show()

# 2. 等待时间分布
plt.hist(df["waiting_time"], bins=30)
plt.xlabel("Waiting Time (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Waiting Time")
plt.show()

# 3. 到达时间 vs 服务时间的散点图
plt.scatter(df["arrival_time"], df["service_time"], alpha=0.6)
plt.xlabel("Arrival Time (minutes)")
plt.ylabel("Service Time (minutes)")
plt.title("Arrival Time vs Service Time")
plt.show()

# 4. 平均等待时间随到达时间变化（看系统是否拥堵）
df_sorted = df.sort_values("arrival_time")
window = 50
rolling_mean = df_sorted["waiting_time"].rolling(window=window).mean()
plt.plot(df_sorted["arrival_time"], rolling_mean)
plt.xlabel("Arrival Time (minutes)")
plt.ylabel(f"Rolling Mean of Waiting Time (window={window})")
plt.title("Trend of Waiting Time over Time")
plt.show()

