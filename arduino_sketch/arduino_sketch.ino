#include <Servo.h>

Servo servo_left;
Servo servo_right;

int input_pinx = A7;
int input_piny = A4;
String strx = "X:";
String stry = "Y:";


int left_limit_down = 0;
int left_limit_up = 60;
int right_limit_down = 60;
int right_limit_up = 120;

int left_current_angle = left_limit_down;
int right_current_angle = right_limit_up;
int angle_change = 180;


enum command {
  NO_DATA = -1,
  LEFT_UP,
  LEFT_DOWN,
  RIGHT_UP,
  RIGHT_DOWN
};

void setup() {
  Serial.begin(9600);
  pinMode(input_pinx, INPUT);
  pinMode(input_piny, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  servo_left.attach(7);
  servo_right.attach(6);
  servo_left.write(left_limit_up);
  servo_right.write(right_limit_down);
  delay(3000);
}

void loop() {
  //Serial.println(analogRead(input_pinx));
  //Serial.println(stry+analogRead(input_piny));
  servo_left.write(left_current_angle);
  servo_right.write(right_current_angle);
  byte data = Serial.read();
  if (data == NO_DATA) {
    Serial.println("no_data");
  }
  if (data == LEFT_DOWN) {
    //Serial.println("left");
    left_current_angle = max(left_limit_down, left_current_angle - angle_change);
  }
  if (data == LEFT_UP) {
    //Serial.println("right");
    left_current_angle = min(left_limit_up, left_current_angle + angle_change);
  }
  if (data == RIGHT_UP) {
    //Serial.println("right");
    right_current_angle = max(right_limit_down, right_current_angle - angle_change);
  }
  if (data == RIGHT_DOWN) {
    //Serial.println("right");
    right_current_angle = min(right_limit_up, right_current_angle + angle_change);
  }
  //Serial.println(current_angle);
  //delay(500);
}
