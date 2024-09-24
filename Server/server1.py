import zmq
import psycopg2
import time

def to_database(cur, suhu, kelembapan, tegangan, arus):
 cur.execute("INSERT INTO datadht_client1 (temp, hum, voltage, current) VALUES (%s, %s, %s, %s)", (suhu, kelembapan, tegangan, arus))


def main():
 context = zmq.Context()
 socket = context.socket(zmq.SUB)
 socket.bind("tcp://127.0.0.1:8005")
 socket.setsockopt_string(zmq.SUBSCRIBE, '')

 conn = psycopg2.connect(
	dbname="postgres",
	user="postgres",
	password="1",
	host="localhost"
 )

 cur = conn.cursor()

 try:

  while True:
   message = socket.recv_string()
   print(f"Received: {message}")
   suhu, kelembapan, tegangan, arus = message.split(',')
   to_database(cur, float(suhu), float(kelembapan), float(tegangan), float(arus))
   conn.commit()
   time.sleep(2)
 except KeyboardInterrupt:
  print("Server ditutup")
  conn.close()
  cur.close()

if __name__ == "__main__":
 main()
