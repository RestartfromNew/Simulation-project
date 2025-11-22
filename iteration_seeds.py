import main_withComments
import pandas as pd
import os
df=pd.DataFrame()

for seed in range(1, 100):
    result=main_withComments.run_simulation(seed)
    row_df = pd.DataFrame([result])
    df = pd.concat([df, row_df], ignore_index=True)
    # file_path="/Users/cin/工程文件/Python/Simulation-project/simulation_summary.csv"
    # if not os.path.exists(file_path):
    #     result.to_csv(file_path, index=False)
    # else:
    #     result.to_csv(file_path, mode='a', index=False, header=False)
df = df.applymap(lambda x: x.item() if hasattr(x, "item") else x)

print("平均服务时间",df["average_service_time"].mean())
print("平均等待时间",df["average_waiting_time"].mean())
print("平均系统时间",df["average_system_time"].mean())
print("医生平均服务时间",df["doctor_mean_service_time"].mean())
print("医生平均利用率",df["doctor_utilization"].mean())