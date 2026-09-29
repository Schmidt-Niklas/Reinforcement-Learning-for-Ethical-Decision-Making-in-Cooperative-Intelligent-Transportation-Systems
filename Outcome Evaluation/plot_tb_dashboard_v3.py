import subprocess

logdir = (
    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
    r"\00_InTas_New - RL\Pyscripts\logs"
)

subprocess.Popen(
    [
        "tensorboard",
        "--logdir",
        logdir,
        "--port",
        "6006"
    ],
    shell=True
)

print("TensorBoard gestartet")
print("http://localhost:6006")