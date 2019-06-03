#!/usr/bin/env python3
import time
import RPi.GPIO as GPIO
"""
Note due to odd GPIO hardware issue with sentry script, 
firing logic uses input/output workaround instead of high/low logic to control turret operation
"""

ammo = 6 # v959 plastic dart turret ammo count
channel = 21 # 3V relay GPIO address

# GPIO setup
def gpio_setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(channel, GPIO.IN)


def fire_semi_auto():
    global ammo
    if ammo > 0:
        print('firing')
        GPIO.setup(channel, GPIO.OUT)
        time.sleep(0.05)
        GPIO.setup(channel, GPIO.IN)
        time.sleep(0.05)
        ammo -= 1
    else:
        print('out of ammo')


def fire_full_auto():
    global ammo
    ammo_count = ammo
    if ammo <= 0:
        print('out of ammo')
        return
    print('supressing')
    for i in range(ammo_count):
        fire_semi_auto()


def reload():
   global ammo
   ammo = 6
   print('turret reloaded')


def cease_fire():
    GPIO.setup(channel, GPIO.IN)
    time.sleep(0.05)

