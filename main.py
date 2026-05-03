import subprocess
import sys
import os

# -----------------------------------
# Config: App folders + ports
# -----------------------------------
APPS = [
    ("Wodden_allot_forecast", 5000),
    ("Wodden_dehired_forecast", 5001),
    ("plastic_allot_forecast", 5002),
    ("plastic_dehired_forecast", 5003),
]

# -----------------------------------
# Start apps
# -----------------------------------
# def start_app(folder, port):
#     print(f"Starting {folder} on port {port}...")
#
#     env = os.environ.copy()
#     env["FLASK_APP"] = "app.py"
#     env["FLASK_RUN_PORT"] = str(port)
#
#     return subprocess.Popen(
#         [sys.executable, "app.py"],
#         cwd=folder,
#         env=env
#     )

def start_app(folder, port):
    print(f"Starting {folder} on port {port}...")

    env = os.environ.copy()
    env["PORT"] = str(port)

    return subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=folder,
        env=env
    )


def main():
    processes = []

    try:
        for folder, port in APPS:
            p = start_app(folder, port)
            processes.append(p)

        print("\nAll apps are running:")
        for folder, port in APPS:
            print(f"http://127.0.0.1:{port} -> {folder}")

        # Keep running
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\nStopping all apps...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()