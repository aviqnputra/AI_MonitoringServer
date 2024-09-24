import zmq
import json
import time
import Adafruit_DHT
import numpy as np
import random

def randomVoltage():
    return round(random.uniform(0,10),1)
def randomCurrent():
    return round(random.uniform(0,10),1)

def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect("tcp://127.0.0.1:5003")
    try:
        while True:
            sensor = Adafruit_DHT.DHT22
            pin = 4
            tegangan = randomVoltage()
            arus = randomCurrent()
            #client_id = "client1"
            kelembapan, suhu = Adafruit_DHT.read_retry(sensor, pin)
            if kelembapan is not None and suhu is not None and suhu > 15 and kelembapan < 100:
                message = f"{suhu:.1f},{kelembapan:.1f},{tegangan},{arus}"
                socket.send_string(message)
                print("Data sent - Temperature:", suhu, "Humidity:", kelembapan, "volt:", tegangan, "arus:", arus)
            else:
                print("Failed")
	    
            time.sleep(1)
    except KeyboardInterrupt:

        print("Close Client 1")
if __name__ == "__main__":
    main()
    

