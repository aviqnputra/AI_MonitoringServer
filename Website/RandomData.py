import time
import random
import psycopg2

def to_database(conn, cur, suhu, kelembapan, tegangan, arus):
    try:
        cur.execute("INSERT INTO datadht_client1 (temp, hum, voltage, current) VALUES (%s, %s, %s, %s)", (suhu, kelembapan, tegangan, arus))
        conn.commit()
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()
def to_databasepred(conn, cur, ran, ran2):
    try:
        cur.execute("INSERT INTO datadhtpred_client1 (pred_kelembapan, pred_suhu) VALUES (%s, %s)", (ran, ran2))
        conn.commit()
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()  

def to_database2(conn, cur, suhu2, kelembapan2, tegangan2, arus2):
    try:
        cur.execute("INSERT INTO datadht_client2 (temp, hum, voltage, current) VALUES (%s, %s, %s, %s)", (suhu2, kelembapan2, tegangan2, arus2))
        conn.commit()
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback() 
def to_databasepred2(conn, cur, ran, ran2):
    try:
        cur.execute("INSERT INTO datadhtpred_client2 (pred_kelembapan, pred_suhu) VALUES (%s, %s)", (ran, ran2))
        conn.commit()
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()  

def to_database3(conn, cur, suhu3, kelembapan3, tegangan3, arus3):
    try:
        cur.execute("INSERT INTO datadht_client3 (temp, hum, voltage, current) VALUES (%s, %s, %s, %s)", (suhu3, kelembapan3, tegangan3, arus3))
        conn.commit()
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()     
def to_databasepred3(conn, cur, ran2, ran3):
    try:
        cur.execute("INSERT INTO datadhtpred_client3 (pred_kelembapan, pred_suhu) VALUES (%s, %s)", (ran2, ran3))
        conn.commit()
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()  

      

def randomVoltage():
    return round(random.uniform(220, 230), 1)

def randomCurrent():
    return round(random.uniform(0, 10), 1)

def randomTemp():
    return round(random.uniform(0, 100), 1)

def randomHum():
    return round(random.uniform(0, 100), 1)

def pred():
    return round(random.uniform(0, 100), 1)

def randomVoltage2():
    return round(random.uniform(220, 230), 1)

def randomCurrent2():
    return round(random.uniform(0, 10), 1)

def randomTemp2():
    return round(random.uniform(0, 100), 1)

def randomHum2():
    return round(random.uniform(0, 100), 1)

def pred2():
    return round(random.uniform(0, 100), 1)

def randomVoltage3():
    return round(random.uniform(220, 230), 1)

def randomCurrent3():
    return round(random.uniform(0, 10), 1)

def randomTemp3():
    return round(random.uniform(0, 100), 1)

def randomHum3():
    return round(random.uniform(0, 100), 1)

def pred3():
    return round(random.uniform(0, 100), 1)

def main():
    try:
        with psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="1",
            host="localhost",
            port = 5432
        ) as conn:
            with conn.cursor() as cur:
                while True:
                    suhu = randomTemp()
                    kelembapan = randomHum()
                    tegangan = randomVoltage()
                    arus = randomCurrent()
                    ran = pred()

                    suhu2 = randomTemp2()
                    kelembapan2 = randomHum2()
                    tegangan2 = randomVoltage2()
                    arus2 = randomCurrent2()
                    ran2 = pred2()

                    suhu3 = randomTemp3()
                    kelembapan3 = randomHum3()
                    tegangan3 = randomVoltage3()
                    arus3 = randomCurrent3()
                    ran3 = pred3()

                    message = f"{suhu},{suhu2},{suhu3},{arus}"
                    print(message)

                    to_database(conn, cur, float(suhu), float(kelembapan), float(tegangan), float(arus))
                    to_database2(conn, cur, float(suhu2), float(kelembapan2), float(tegangan2), float(arus2))
                    to_database3(conn, cur, float(suhu3), float(kelembapan3), float(tegangan3), float(arus3))
                    to_databasepred(conn, cur, float(ran), float(ran))
                    to_databasepred2(conn, cur, float(ran2), float(ran2))
                    to_databasepred3(conn, cur, float(ran3), float(ran3))
                    
                    time.sleep(1)

    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
