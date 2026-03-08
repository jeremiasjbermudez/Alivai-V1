import requests, time
while True:
    try:
        data = requests.get("http://localhost:8000/v1/hff/status").json()
        print(f"Pulse: Zeta={data['zeta']:.6f} | State={data['final_state']:.2e}")
    except:
        print("Pulse Offline...")
    time.sleep(2)
