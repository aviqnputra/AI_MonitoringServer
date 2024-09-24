from flask import Flask, render_template, jsonify, send_from_directory
import psycopg2
from flask_cors import CORS
import os

app = Flask(__name__, static_url_path='/static')
CORS(app)

def get_db_connection():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="1",
        host="localhost",
        port=5432
    )
    return conn

@app.route('/datadht1')
def datadht1():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT ROUND(hum::numeric,1), ROUND(temp::numeric,1), ROUND(voltage::numeric,1), ROUND(current::numeric,1) FROM datadht_client1;')
    datadht1 = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(datadht1)

@app.route('/dataPredDHT1')
def dataPredDHT1():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT pred_kelembapan, pred_suhu FROM datadhtpred_client1;')
    dataPredDHT1 = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(dataPredDHT1)


@app.route('/dataorang1')
def dataorang1():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT ROUND(orang::numeric,1) FROM dataorang_client1;')  # Adjust the query to select relevant columns
    datadht1 = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(datadht1)

@app.route('/datadht2')
def datadht2():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT ROUND(hum::numeric,1), ROUND(temp::numeric,1), ROUND(voltage::numeric,1), ROUND(current::numeric,1) FROM datadht_client2;')
    datadht2 = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(datadht2)

@app.route('/dataPredDHT2')
def dataPredDHT2():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT pred_kelembapan, pred_suhu FROM datadhtpred_client2;')
    dataPredDHT2 = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(dataPredDHT2)


@app.route('/dataorang2')
def dataorang2():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT ROUND(orang::numeric,1) FROM dataorang_client2;')  # Adjust the query to select relevant columns
    datadht2 = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(datadht2)

@app.route('/datadht3')
def datadht3():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT ROUND(hum::numeric,1), ROUND(temp::numeric,1), ROUND(voltage::numeric,1), ROUND(current::numeric,1) FROM datadht_client3;')
    datadht3 = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(datadht3)

@app.route('/dataPredDHT3')
def dataPredDHT3():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT pred_kelembapan, pred_suhu FROM datadhtpred_client3;')
    dataPredDHT3 = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(dataPredDHT3)

@app.route('/dataorang3')
def dataorang3():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT ROUND(orang::numeric,1) FROM dataorang_client3;')  # Adjust the query to select relevant columns
    datadht3 = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(datadht3)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/node1')
def node1():
    return render_template('node1.html')

@app.route('/node2')
def node2():
    return render_template('node2.html')

@app.route('/node3')
def node3():
    return render_template('node3.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8003)
