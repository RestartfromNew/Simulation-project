"""
Discrete-event simulation of an Emergency Department (ED) queueing system.

- SimPy-based DES with k doctors (DOCTOR_NUM) and CTAS 1–5 triage levels.
- Two-stage flow: nurse assessment -> doctor service, with possible admission and bed waiting.
- Arrival process: non-homogeneous gamma-based interarrival times with controlled SCV.
- Service times: CTAS-dependent gamma-distributed durations.
- CTAS1 patients can preempt doctors treating lower-priority patients.
- The simulation runs with a warm-up period and then a main observation period.
- For each run, per-patient and per-doctor statistics are collected, and a summary dict is returned.
"""

from socket import send_fds  # currently unused, can be removed if not needed

import simpy
import random
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma

k = 2
GLOBAL_SEED = None
all_patients = []
all_doctor = []
random.seed(GLOBAL_SEED)
rng_np = np.random.default_rng(GLOBAL_SEED)

DOCTOR_NUM = k
SIMULATION_TIME = 7 * 24 * 60
DAY_MINUTES = 24 * 60
WARM_UP_TIME = 7 * 24 * 60

# average number of patients per hour of the day
MEAN_PATIENT_PER_HOUR = [
    3.0, 2.8, 2.5, 2.2, 2.0, 2.3,
    3.0, 5.0, 7.5, 9.5, 10.5, 10.0,
    9.8, 9.2, 8.8, 7.8, 7.2, 6.6,
    6.0, 5.5, 4.8, 4.0, 3.5, 3.2
]

rates = np.array(MEAN_PATIENT_PER_HOUR)

# fit gamma distribution for hourly arrival rates
shape, loc, scale = gamma.fit(rates, floc=0)
print("shape k =", shape, "scale θ =", scale)

CTAS = {"ctas1": 0, "ctas2": 15, "ctas3": 30, "ctas4": 60, "ctas5": 120}

# probability distribution of CTAS levels
CTAS_distribution = {"ctas1": 0.01, "ctas2": 0.16, "ctas3": 0.56, "ctas4": 0.25, "ctas5": 0.02}

# mean nurse-processing times by CTAS (minutes)
CTAS_nurse = {"ctas1": 3.2, "ctas2": 7.1, "ctas3": 20.4, "ctas4": 39.7, "ctas5": 32.1}

ADMISSION_PROB = {
    "ctas1": 0.89,
    "ctas2": 0.65,
    "ctas3": 0.35,
    "ctas4": 0.13,
    "ctas5": 0.05
}

BED_WAIT_MEAN = {
    "ctas1": 60.0,
    "ctas2": 120.0,
    "ctas3": 180.0,
    "ctas4": 60.0,
    "ctas5": 30.0
}

CTAS1_queue = []
CTAS2_queue = []
CTAS3_queue = []
departure_list = []
patient_id = 0

ARRIVAL_SCV = 1.708812612  # SCV: squared coefficient of variation for arrival process


class Doctor:
    def __init__(self, env, id):
        self.proc = None
        self.env = env
        self.id = id
        self.status = 'idle'
        self.patient_level = None
        self.busy_time = 0
        self.seen_number = 0
        self.current_patient_id = None

    def start_service(self):
        return

    def end_service(self):
        return


class Patient:
    def __init__(self, env, doctors, id, arrive_time, ctas_level, rest_waiting_time, nurse_process_time):
        self.env = env
        self.id = id
        self.status = 'arrival'
        self.nurse_process_time = nurse_process_time

        self.arrival_time = arrive_time
        self.waiting_time = 0
        self.total_time = 0
        self.service_time = 0
        self.departure_time = None
        self.bed_time = 0
        self.ctas_level = ctas_level
        self.queue_rank = 0
        self.doctors = doctors

        # queue_rank is based on waiting_time/rest_waiting_time, higher is served earlier
        self.rest_waiting_time = rest_waiting_time
        self.proc = env.process(self.run())

    def print_required_parameters(self):
        print("rest_waiting_time:", self.rest_waiting_time)

    def __lt__(self, other):
        """heapq uses < for comparison"""
        return self.queue_rank > other.queue_rank

    def run(self):
        # nurse stage
        yield self.env.timeout(self.nurse_process_time)
        # after this moment, patient is ready to join the doctor queue
        self.status = "waiting"

        renew_waiting_rank(self)  # update queue rank after nurse time
        insert_queue(self.ctas_level, self)

        if self.ctas_level == "ctas1":
            disruption(self.env.now, self, self.ctas_level, self.doctors)

        renew_queue(self.env.now)


def gamma_params_with_SCV(mean_val, scv):
    """
    Compute shape (alpha) and scale (theta) for a gamma distribution
    given mean and squared coefficient of variation (SCV).
    SCV = Var(X) / (E[X])^2
    """
    scv = max(scv, 1e-9)
    alpha = 1.0 / scv
    theta = mean_val / alpha
    return alpha, theta


def sample_interarrival(env_time):
    """
    Sample interarrival time (minutes) using:
    - gamma-distributed hourly arrival rate
    - then gamma with target SCV for interarrival times
    """
    mean_arrival = rng_np.gamma(shape=shape, scale=scale)
    mean_arrival = max(mean_arrival, 1e-6)

    mean_interarrival = 60.0 / mean_arrival

    alpha, theta = gamma_params_with_SCV(mean_interarrival, ARRIVAL_SCV)
    interarrival_gap = rng_np.gamma(shape=alpha, scale=theta)

    return float(max(interarrival_gap, 1e-6))


ARRIVAL_RATE_PER_HOUR = 10


def sample_interarrival_constant(env_time):
    """
    Alternative interarrival generator with constant hourly rate
    and SCV control via gamma distribution.
    """
    mean_interarrival = 60 / ARRIVAL_RATE_PER_HOUR
    alpha, theta = gamma_params_with_SCV(mean_interarrival, ARRIVAL_SCV)
    return float(max(rng_np.gamma(shape=alpha, scale=theta), 1e-6))


def sample_nurse_visit_time(env_time, patient_level):
    mean = CTAS_nurse[patient_level]
    lam = 1.0 / mean
    return random.expovariate(lam)


def sample_level():
    """Sample CTAS level from the categorical distribution."""
    r = random.random()
    cum = 0
    for level, p in CTAS_distribution.items():
        cum += p
        if r <= cum:
            return level
    return "ctas5"


def renew_queue(current_time):
    """Update queue ranks when an event occurs."""
    for q in (CTAS1_queue, CTAS2_queue, CTAS3_queue):
        for p in q:
            p.waiting_time = current_time - p.arrival_time
            renew_waiting_rank(p)
        heapq.heapify(q)


def renew_waiting_rank(patient):
    """
    Update a patient's rank.
    CTAS1 patients with zero remaining waiting time get a very large rank
    so that they are always prioritized.
    """
    if patient.rest_waiting_time <= 0:
        patient.queue_rank = 1e9 + patient.waiting_time
    else:
        patient.queue_rank = patient.waiting_time / patient.rest_waiting_time


def insert_queue(level, patient):
    """Push patient into the appropriate priority queue."""
    if level == "ctas1":
        heapq.heappush(CTAS1_queue, patient)
    elif level == "ctas2":
        heapq.heappush(CTAS2_queue, patient)
    else:
        heapq.heappush(CTAS3_queue, patient)
    print("patient arrived, patient id:", patient_id, " arrive time", patient.arrival_time)


def disruption(env_time, patient, level, doctors):
    """
    For CTAS1, check if a doctor can be preempted.
    Prefer to preempt a doctor treating CTAS3, then CTAS2.
    """
    if level == "ctas1":
        candidate = None
        for d in doctors:
            if d.status == 'busy' and d.patient_level != "ctas1":
                if candidate is None:
                    candidate = d
                else:
                    if d.patient_level > candidate.patient_level:
                        candidate = d

        if candidate is not None:
            print(f"CTAS1 arrival: preempt Doctor{candidate.id} treating {candidate.patient_level}")
            candidate.proc.interrupt()


def generate_arrival(env, doctors):
    """Generate arriving patients over time."""
    global patient_id
    last_report = 0
    hourly_count = 0
    while True:
        interarrival = sample_interarrival(env.now)
        # alternative:
        # interarrival = sample_interarrival_constant(env.now)

        yield env.timeout(interarrival)
        patient_id += 1

        level = sample_level()
        nurse_process_time = sample_nurse_visit_time(env.now, level)
        rest_waiting_time = CTAS[level]
        current_time = env.now
        Patient(env, doctors, patient_id, current_time, level, rest_waiting_time, nurse_process_time)

        renew_queue(current_time)

        if env.now - last_report >= 60:
            print(
                f"[t={env.now / 60:.1f}h] total arrivals: {patient_id}, "
                f"CTAS1={len(CTAS1_queue)}, CTAS2={len(CTAS2_queue)}, CTAS3={len(CTAS3_queue)}"
            )
            last_report = env.now
            hourly_count = 0


def sample_service_time(patient):
    """Sample CTAS-dependent service time using a gamma distribution."""
    base_time = {
        "ctas1": 73.6,
        "ctas2": 38.9,
        "ctas3": 26.3,
        "ctas4": 15.0,
        "ctas5": 10.9
    }[patient.ctas_level]

    service_scv_by_ctas = {
        "ctas1": 1.2,
        "ctas2": 1.0,
        "ctas3": 0.8,
        "ctas4": 0.6,
        "ctas5": 0.4
    }
    scv = service_scv_by_ctas[patient.ctas_level]
    alpha = 1.0 / scv
    theta = base_time / alpha
    service_time = random.gammavariate(alpha, theta)
    return max(service_time, 1e-6)


def find_next_patient() -> 'Patient | None':
    """Select the next patient based on highest-rank among CTAS queues."""
    if len(CTAS1_queue) > 0:
        patient = heapq.heappop(CTAS1_queue)
        print(f"Pick CTAS1 patient {patient.id}, rank={patient.queue_rank:.2f}, wait={patient.waiting_time:.1f}")
    elif len(CTAS2_queue) > 0:
        patient = heapq.heappop(CTAS2_queue)
        print(f"Pick CTAS2 patient {patient.id}, rank={patient.queue_rank:.2f}, wait={patient.waiting_time:.1f}")
    elif len(CTAS3_queue) > 0:
        patient = heapq.heappop(CTAS3_queue)
        print(f"Pick CTAS3 patient {patient.id}, rank={patient.queue_rank:.2f}, wait={patient.waiting_time:.1f}")
    else:
        patient = None
    return patient


def doctor_process(env, doctor):
    """Doctor process: repeatedly picks and serves patients, with preemption."""
    while True:
        patient: 'Patient' = find_next_patient()
        if not patient:
            # no patients, wait and re-check
            yield env.timeout(1)
            continue

        doctor.status = 'busy'
        doctor.current_patient_id = patient.id
        doctor.patient_level = patient.ctas_level
        patient.waiting_time = env.now - patient.arrival_time
        service_time = sample_service_time(patient)
        patient.status = 'being serviced'
        print(
            f"Doctor{doctor.id}, patient {doctor.current_patient_id} is now serviced, "
            f"service_time={service_time:.2f}, start_time={env.now:.2f}"
        )
        start_time = env.now
        try:
            # normal completion
            yield env.timeout(service_time)
            base_departure_time = env.now
            doctor.status = 'idle'
            doctor.busy_time += service_time
            doctor.seen_number += 1
            doctor.current_patient_id = None
            doctor.patient_level = None
            patient.service_time = service_time

            admitted = (random.random() < ADMISSION_PROB[patient.ctas_level])
            if admitted:
                print(f"{patient.ctas_level} is admitted")
                mean_wait = BED_WAIT_MEAN[patient.ctas_level]
                bed_wait = random.expovariate(1.0 / mean_wait)
            else:
                bed_wait = 0.0

            patient.bed_time = bed_wait
            patient.departure_time = base_departure_time + bed_wait
            patient.total_time = patient.departure_time - patient.arrival_time
            patient.status = 'depart'

            departure_list.append(patient)

            print(
                f"Finished Doctor{doctor.id}, patient {patient.id}, "
                f"decision_time={base_departure_time:.2f}, logical_departure={patient.departure_time:.2f}"
            )

            renew_queue(base_departure_time)

        except simpy.Interrupt:
            # preemption
            current_time = env.now
            served = env.now - start_time
            print(
                f"Interruption: Doctor{doctor.id} interrupted after {served:.2f} minutes "
                f"for patient {patient.id} (level {patient.ctas_level})"
            )
            doctor.busy_time += served
            patient.service_time += served
            renew_queue(current_time)
            patient.status = 'waiting'
            renew_waiting_rank(patient)
            patient.queue_rank += 1

            if patient.ctas_level == "ctas1":
                heapq.heappush(CTAS1_queue, patient)
            elif patient.ctas_level == "ctas2":
                heapq.heappush(CTAS2_queue, patient)
            else:
                heapq.heappush(CTAS3_queue, patient)

            doctor.status = 'idle'
            doctor.current_patient_id = None
            doctor.patient_level = None


def run_simulation(seeds):
    global GLOBAL_SEED, rng_np
    GLOBAL_SEED = seeds
    random.seed(GLOBAL_SEED)
    rng_np = np.random.default_rng(GLOBAL_SEED)

    env = simpy.Environment()
    doctors = [Doctor(env, f'Doctor{i}') for i in range(0, DOCTOR_NUM)]
    env.process(generate_arrival(env, doctors))
    for doctor in doctors:
        doctor.proc = env.process(doctor_process(env, doctor))

    print(f"---- Warm-up for {WARM_UP_TIME} minutes ----")
    env.run(until=WARM_UP_TIME)

    # after warm-up, keep current queues and doctor states, but reset statistics
    departure_list.clear()
    for d in doctors:
        d.busy_time = 0.0
        d.seen_number = 0

    print("---- Start official simulation period ----")
    env.run(until=WARM_UP_TIME + SIMULATION_TIME)

    all_patients.extend(departure_list)
    all_doctor.extend(doctors)

    for p in departure_list:
        print(
            f"id={p.id}, arrive={p.arrival_time:.2f}, "
            f"depart={p.departure_time:.2f}, "
            f"service={p.service_time:.2f}, "
            f"wait={p.waiting_time:.2f}, ctas={p.ctas_level}"
        )

    data = [{
        "id": p.id,
        "arrival_time": p.arrival_time,
        "service_time": p.service_time,
        "waiting_time": p.waiting_time,
        "departure_time": p.departure_time,
        "bed_time": p.bed_time,
        "ctas_level": p.ctas_level,
        "whole_time": p.departure_time - p.arrival_time,
        "nurse_time": p.nurse_process_time,
    } for p in departure_list]

    df = pd.DataFrame(data)
    Patient_number = len(departure_list)
    print(f"Total patients in this run: {Patient_number}")

    data_doctor = [{
        "id": d.id,
        "busy time": d.busy_time,
        "utilization": d.busy_time / SIMULATION_TIME
    } for d in doctors]
    ddf = pd.DataFrame(data_doctor)

    print("Arrival rate:", Patient_number / SIMULATION_TIME)
    print("Average service time:", df["service_time"].mean())
    print("Average waiting time:", df["waiting_time"].mean())
    print("Average total time in system:", df["departure_time"].sub(df["arrival_time"]).mean())

    for d in doctors:
        print(
            f"{d.id} total busy time = {d.busy_time:.2f}, "
            f"utilization = {d.busy_time / SIMULATION_TIME:.3f}"
        )

    ctas_counts = df["ctas_level"].value_counts().to_dict()
    print("Number of patients by CTAS:", ctas_counts)

    by_ctas = df.groupby("ctas_level").agg(
        patient_count=("id", "count"),
        mean_waiting=("waiting_time", "mean"),
        mean_service=("service_time", "mean"),
        mean_whole=("whole_time", "mean"),
        mean_nurse=("nurse_time", "mean"),
        mean_bed=("bed_time", "mean"),
    )

    simulation_resut = {
        "seed": GLOBAL_SEED,
        "total_patients": int(Patient_number),
        "overall_average_service_time": float(df["service_time"].mean()),
        "overall_average_waiting_time": float(df["waiting_time"].mean()),
        "overall_average_system_time": float(df["departure_time"].sub(df["arrival_time"]).mean()),
        "doctor_mean_service_time": float(ddf["busy time"].mean()),
        "doctor_utilization": float(ddf["utilization"].mean()),
        "nurse_time": float(df["nurse_time"].mean()),
    }

    levels = ["ctas1", "ctas2", "ctas3", "ctas4", "ctas5"]
    for lvl in levels:
        if lvl in by_ctas.index:
            row = by_ctas.loc[lvl]
            simulation_resut[f"{lvl}_patient_count"] = int(row["patient_count"])
            simulation_resut[f"{lvl}_mean_waiting"] = float(row["mean_waiting"])
            simulation_resut[f"{lvl}_mean_service"] = float(row["mean_service"])
            simulation_resut[f"{lvl}_mean_whole"] = float(row["mean_whole"])
            simulation_resut[f"{lvl}_mean_nurse"] = float(row["mean_nurse"])
            simulation_resut[f"{lvl}_mean_bed"] = float(row["mean_bed"])
        else:
            simulation_resut[f"{lvl}_patient_count"] = 0
            simulation_resut[f"{lvl}_mean_waiting"] = np.nan
            simulation_resut[f"{lvl}_mean_service"] = np.nan
            simulation_resut[f"{lvl}_mean_whole"] = np.nan
            simulation_resut[f"{lvl}_mean_nurse"] = np.nan
            simulation_resut[f"{lvl}_mean_bed"] = np.nan

    return simulation_resut


for i in range(1, 100):
    CTAS1_queue = []
    CTAS2_queue = []
    CTAS3_queue = []
    departure_list = []
    doctors = []
    run_simulation(i)

data = [{
    "id": p.id,
    "arrival_time": p.arrival_time,
    "service_time": p.service_time,
    "waiting_time": p.waiting_time + p.bed_time + p.nurse_process_time,
    "waiting_time_before_process": p.waiting_time + p.bed_time,
    "departure_time": p.departure_time,
    "bed_time": p.bed_time,
    "ctas_level": p.ctas_level,
    "whole_time": p.departure_time - p.arrival_time,
    "nurse_time": p.nurse_process_time,
} for p in all_patients]

df = pd.DataFrame(data)

plt.hist(df["service_time"], bins=30)
plt.xlabel("Service Time")
plt.ylabel("Frequency")
plt.title(f"Distribution of Service Time (k={k})")
plt.show()

print(f"Total number of patients across all runs: {len(all_patients)}")
print("shape k =", shape, "scale θ =", scale)
