import subprocess
import time


KERNEL = "aryansingh21fd/securehealthiot-disease-trainer-v1"


def get_status() -> str:
    cp = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL],
        capture_output=True,
        text=True,
        check=True,
    )
    out = cp.stdout.strip()
    if "status" in out:
        return out.split("status", 1)[-1].strip().strip('"')
    return out


def main():
    while True:
        status = get_status()
        print(status)
        if "COMPLETE" in status.upper() or "ERROR" in status.upper():
            break
        time.sleep(20)


if __name__ == "__main__":
    main()

