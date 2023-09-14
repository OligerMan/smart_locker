#include <Servo.h>

Servo servo;

int input_pinx = A7;
int input_piny = A4;
String strx = "X:";
String stry = "Y:";
int current_angle = 90;
int angle_change = 30;


enum command {
  NO_DATA = -1,
  LEFT,
  RIGHT
};

void setup() {
  Serial.begin(9600);
  pinMode(input_pinx, INPUT);
  pinMode(input_piny, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  servo.attach(7);
}

void loop() {
  //Serial.println(analogRead(input_pinx));
  //Serial.println(stry+analogRead(input_piny));
  servo.write(current_angle);
  byte data = Serial.read();
  if (data == NO_DATA) {
    Serial.println("no_data");
  }
  if (data == LEFT) {
    Serial.println("left");
    current_angle = max(0, current_angle - angle_change);
  }
  if (data == RIGHT) {
    Serial.println("right");
    current_angle = min(180, current_angle + angle_change);
  }
  Serial.println(current_angle);
  delay(500);
}
