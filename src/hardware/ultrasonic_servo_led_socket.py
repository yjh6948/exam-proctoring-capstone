import socket
import pigpio
import time
import os
import subprocess
import RPi.GPIO as GPIO
from gpiozero import LED

# 핀 설정
servo_pin = 18
TRIG = 23
ECHO = 24
led_pin = 17

# LED, GPIO 초기화
led = LED(led_pin)
pi = pigpio.pi()

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)


#초음파 거리 센서 함수
def measure_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO) == 0:
        start_time = time.time()

    while GPIO.input(ECHO) == 1:
        stop_time = time.time()

    duration = stop_time - start_time
    distance = duration * 17150  # Distance in cm

    return distance

#실시간 카메라 함수
def start_camera():
    subprocess.Popen(["libcamera-hello", "-t", "0"])
    print("Camera activated.")


 #초음파 센서로 거리 측정
	      distance = measure_distance()
	      print(f"Distance: {distance:.2f} cm")
	
	      # If the distance is less than or equal to 90 cm, activate the camera
	      if distance <= 110:
	          print("Distance is within 90 cm. Activating camera.")
	          start_camera()
	          
	      else: 
			      print("Come Closer to Activate the camera.")

#소켓 통신 연결 시
        while True:
            # Wait for client connection
            client_socket, addr = server_socket.accept()
            with client_socket:
                print("Connected by", addr)

                # client로부터 data 받기
                data = client_socket.recv(1024).decode()
                if data == '1': #data = 1 수신 받으면 
                    print("Door Opening")
                    set_servo_angle(135)  # 문 열림
                    time.sleep(5)  
                    print("Door Closing")
                    set_servo_angle(0)  # 문 닫힘
                    time.sleep(5)  # Wait before accepting next request
                elif data == '0':  #data = 0 수신 받으면
                    print("You are Non-owner. The door will not open.")
                    led.on()  #led 켜짐
                    time.sleep(10)  
                    led.off()
                    os.system("libcamera-still -o non-owner.jpg")  
                    # 외부인인 경우 사진 찍어서 저장
                else:
                    print(f"Received invalid data: {data}")

if __name__ == "__main__":
    start_server()
