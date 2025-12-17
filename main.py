# @Author  : Yu Bai
# @Time    : 2024/6/16 20:31
# @File    : main.py
# @Description
import subprocess


def start_training():
    # 更新命令，只定向PID输出，其它输出默认写入nohup.out
    command = "nohup /root/miniconda3/bin/python3.8 train.py & echo $! > pid.txt"
    # 使用subprocess执行命令
    subprocess.run(command, shell=True)#.call和.run的区别是.run不会等待命令执行完毕，而是立即返回
    print("train.py has been started in background. PID stored in pid.txt.")

def stop_training():
    subprocess.run(["pkill", "-f", "train.py"])

if __name__ == '__main__':
    # start_training()
    stop_training()