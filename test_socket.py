import socket
import json
import time
import random

def send_data():
    host = 'localhost'
    port = 65432
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        
        # Start a new batch
        print("Sending NEW_BATCH command...")
        new_batch_msg = json.dumps({
            "command": "NEW_BATCH",
            "batch_id": f"test_batch_{time.time_ns()}"
        }) + "\n"
        s.sendall(new_batch_msg.encode('utf-8'))
        time.sleep(1)
        
        # Send 25 readings
        print("Sending 25 readings...")
        base_cold_flow = random.uniform(5.0, 20.0)
        base_inlet_temp = random.uniform(60.0, 120.0)
        base_fouling = random.uniform(0.0, 30.0)
        base_ambient = random.uniform(10.0, 35.0)
        for i in range(1, 26):
            cold_water_flow = base_cold_flow + random.uniform(-2.0, 2.0)
            inlet_temp = base_inlet_temp + random.uniform(-5.0, 5.0)
            fouling_factor = base_fouling + random.uniform(-3.0, 3.0)
            ambient_temp = base_ambient + random.uniform(-2.0, 2.0)
            noise = random.uniform(-1.5, 1.5)
            outlet_temp = (
                0.8 * inlet_temp
                - 0.3 * cold_water_flow
                - 0.2 * fouling_factor
                + 0.5 * ambient_temp
                + noise
            )
            data = {
                "timestamp": time.time(),
                "cold_water_flow": cold_water_flow,
                "inlet_temp": inlet_temp,
                "fouling_factor": fouling_factor,
                "ambient_temp": ambient_temp,
                "outlet_temp": outlet_temp
            }
            msg = json.dumps(data) + "\n"
            s.sendall(msg.encode('utf-8'))
            time.sleep(0.1) # small delay
            
        print("Done!")

if __name__ == "__main__":
    send_data()