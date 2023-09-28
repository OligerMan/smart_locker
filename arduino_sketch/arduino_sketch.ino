#include <Servo.h>

int angle_change = 180;

// NEW INIT

struct Cell {
  int limit_down;
  int limit_up;
  int current_angle;
  int lock_direction; // 0 or 1, referred is right and left turn
  int state; // closed = 0, opened = 1
  int servo_port;
  int green_led_port;
  int green_led_port_type; // 0 for analog, 1 for digital
  int yellow_led_port;
  int yellow_led_port_type; // 0 for analog, 1 for digital
  int red_led_port;
  int red_led_port_type; // 0 for analog, 1 for digital
  Servo servo_driver;
};

#define CELLS_COUNT 5
struct Cell cells[CELLS_COUNT];

// NEW INIT END

enum command {
  NO_DATA = -1,
  LEFT_UP,
  LEFT_DOWN,
  RIGHT_UP,
  RIGHT_DOWN,
  SELECT_LEFT_UP,
  SELECT_LEFT_DOWN,
  SELECT_RIGHT_UP,
  SELECT_RIGHT_DOWN,
  OPEN_CELL,
  CLOSE_CELL,
  OPEN_CELL_NO_PERMISSION,
  CLOSE_CELL_NO_PERMISSION
};

void setup() {
  Serial.begin(9600);

  cells[0].lock_direction = 1;
  cells[1].lock_direction = 1;
  cells[2].lock_direction = 0;
  cells[3].lock_direction = 0;

  cells[0].limit_down = 60;
  cells[0].limit_up = 120;
  cells[1].limit_down = 60;
  cells[1].limit_up = 120;
  cells[2].limit_down = 60;
  cells[2].limit_up = 120;
  cells[3].limit_down = 60;
  cells[3].limit_up = 120;

  cells[0].green_led_port = 3;
  cells[1].green_led_port = 2;
  cells[2].green_led_port = 4;
  cells[3].green_led_port = 10;
  cells[0].yellow_led_port = A7;
  cells[1].yellow_led_port = 12;
  cells[2].yellow_led_port = A1;
  cells[3].yellow_led_port = A2;
  cells[0].red_led_port = A3;
  cells[1].red_led_port = A4;
  cells[2].red_led_port = A5;
  cells[3].red_led_port = A6;
 
  cells[0].green_led_port_type = 1;
  cells[1].green_led_port_type = 1;
  cells[2].green_led_port_type = 1;
  cells[3].green_led_port_type = 1;
  cells[0].yellow_led_port_type = 0;
  cells[1].yellow_led_port_type = 1;
  cells[2].yellow_led_port_type = 0;
  cells[3].yellow_led_port_type = 0;
  cells[0].red_led_port_type = 0;
  cells[1].red_led_port_type = 0;
  cells[2].red_led_port_type = 0;
    cells[3].red_led_port_type = 0;
  
  cells[0].servo_port = 11;
  cells[1].servo_port = 6;
  cells[2].servo_port = 7;
  cells[3].servo_port = 9;
  
  for (int i = 0; i < CELLS_COUNT; i++) {
    cells[i].state = 0;
    pinMode(cells[i].green_led_port, OUTPUT);
    pinMode(cells[i].yellow_led_port, OUTPUT);
    pinMode(cells[i].red_led_port, OUTPUT);
    cells[i].servo_driver.attach(cells[i].servo_port);
    if (cells[i].lock_direction == 0) {
      cells[i].current_angle = cells[i].limit_down;
    }
    else {
      cells[i].current_angle = cells[i].limit_up;
    }
  }
}

int current_cell = 0;
int blink_delay = 100;

void loop() {
  for (int i = 0; i < CELLS_COUNT; i++) {
    cells[i].servo_driver.write(cells[i].current_angle);
    if (i == current_cell) {
      if (cells[current_cell].yellow_led_port_type == 0)
        analogWrite(cells[current_cell].yellow_led_port, 1023);
      if (cells[current_cell].yellow_led_port_type == 1)
        digitalWrite(cells[current_cell].yellow_led_port, HIGH);
    }
    else {
      if (cells[current_cell].yellow_led_port_type == 0)
        analogWrite(cells[current_cell].yellow_led_port, 0);
      if (cells[current_cell].yellow_led_port_type == 1)
        digitalWrite(cells[current_cell].yellow_led_port, LOW);
    }
  }
  byte data = Serial.read();
  if (data == NO_DATA) {
    Serial.println("no_data");
  }
  /*servo_left.write(left_current_angle);
  servo_right.write(right_current_angle);
  
  if (data == LEFT_DOWN)
    left_current_angle = max(left_limit_down, left_current_angle - angle_change);
  if (data == LEFT_UP)
    left_current_angle = min(left_limit_up, left_current_angle + angle_change);
  if (data == RIGHT_UP)
    right_current_angle = max(right_limit_down, right_current_angle - angle_change);
  if (data == RIGHT_DOWN)
    right_current_angle = min(right_limit_up, right_current_angle + angle_change);*/
  if (data == SELECT_LEFT_UP)
    current_cell = 0;
  if (data == SELECT_LEFT_DOWN)
    current_cell = 1;
  if (data == SELECT_RIGHT_UP)
    current_cell = 2;
  if (data == SELECT_RIGHT_DOWN)
    current_cell = 3;
  if (data == OPEN_CELL) {
    cells[current_cell].state = 1;
    if (cells[current_cell].lock_direction == 0) {
      cells[current_cell].current_angle = max(cells[current_cell].limit_down, cells[current_cell].current_angle - angle_change);
    }
    else {
      cells[current_cell].current_angle = min(cells[current_cell].limit_up, cells[current_cell].current_angle + angle_change);
    }
    
    if (cells[current_cell].green_led_port_type == 0){
        analogWrite(cells[current_cell].green_led_port, 1023);
        delay(blink_delay);
        analogWrite(cells[current_cell].green_led_port, 0);
        delay(blink_delay);
        analogWrite(cells[current_cell].green_led_port, 1023);
        delay(blink_delay);
        analogWrite(cells[current_cell].green_led_port, 0);
    }
    if (cells[current_cell].green_led_port_type == 1){
        digitalWrite(cells[current_cell].green_led_port, HIGH);
        delay(blink_delay);
        digitalWrite(cells[current_cell].green_led_port, LOW);
        delay(blink_delay);
        digitalWrite(cells[current_cell].green_led_port, HIGH);
        delay(blink_delay);
        digitalWrite(cells[current_cell].green_led_port, LOW);
    }
  }
  if (data == CLOSE_CELL) {
    cells[current_cell].state = 0;
    if (cells[current_cell].lock_direction == 0) {
      cells[current_cell].current_angle = min(cells[current_cell].limit_up, cells[current_cell].current_angle + angle_change);
    }
    else {
      cells[current_cell].current_angle = max(cells[current_cell].limit_down, cells[current_cell].current_angle - angle_change);
    }
    
    if (cells[current_cell].green_led_port_type == 0){
        analogWrite(cells[current_cell].green_led_port, 1023);
        delay(blink_delay);
        analogWrite(cells[current_cell].green_led_port, 0);
        delay(blink_delay);
        analogWrite(cells[current_cell].green_led_port, 1023);
        delay(blink_delay);
        analogWrite(cells[current_cell].green_led_port, 0);
    }
    if (cells[current_cell].green_led_port_type == 1){
        digitalWrite(cells[current_cell].green_led_port, HIGH);
        delay(blink_delay);
        digitalWrite(cells[current_cell].green_led_port, LOW);
        delay(blink_delay);
        digitalWrite(cells[current_cell].green_led_port, HIGH);
        delay(blink_delay);
        digitalWrite(cells[current_cell].green_led_port, LOW);
    }
  }
  if (data == OPEN_CELL_NO_PERMISSION) {
    if (cells[current_cell].red_led_port_type == 0){
        analogWrite(cells[current_cell].red_led_port, 1023);
        delay(blink_delay);
        analogWrite(cells[current_cell].red_led_port, 0);
        delay(blink_delay);
        analogWrite(cells[current_cell].red_led_port, 1023);
        delay(blink_delay);
        analogWrite(cells[current_cell].red_led_port, 0);
    }
    if (cells[current_cell].red_led_port_type == 1){
        digitalWrite(cells[current_cell].red_led_port, HIGH);
        delay(blink_delay);
        digitalWrite(cells[current_cell].red_led_port, LOW);
        delay(blink_delay);
        digitalWrite(cells[current_cell].red_led_port, HIGH);
        delay(blink_delay);
        digitalWrite(cells[current_cell].red_led_port, LOW);
    }
  }
  if (data == CLOSE_CELL_NO_PERMISSION) {
    if (cells[current_cell].red_led_port_type == 0){
        analogWrite(cells[current_cell].red_led_port, 1023);
        delay(blink_delay);
        analogWrite(cells[current_cell].red_led_port, 0);
        delay(blink_delay);
        analogWrite(cells[current_cell].red_led_port, 1023);
        delay(blink_delay);
        analogWrite(cells[current_cell].red_led_port, 0);
    }
    if (cells[current_cell].red_led_port_type == 1){
        digitalWrite(cells[current_cell].red_led_port, HIGH);
        delay(blink_delay);
        digitalWrite(cells[current_cell].red_led_port, LOW);
        delay(blink_delay);
        digitalWrite(cells[current_cell].red_led_port, HIGH);
        delay(blink_delay);
        digitalWrite(cells[current_cell].red_led_port, LOW);
    }
  }
}
