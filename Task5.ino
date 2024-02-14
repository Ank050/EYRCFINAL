#include <WiFi.h>

const char* ssid = "ILAA HOME";                    
const char* password =  "#angelchenjeri";               
const uint16_t port = 8002;
const uint16_t port2 = 8080;
const char * host = "192.168.0.108"; 
// const char * host = "192.168.0.108";
// const char * host = "192.168.0.108";
// const char * host = "10.42.0.1";
                

char incomingPacket[80];
WiFiClient client;
WiFiClient client2;

String msg = "0";
String msg2 = "0";
String ms = "0";
String event_msg = "0";
String node_path = "0";
String node_list = "0";
int wait_for_path_flag = 0;
int wait_for_node_8 = 0;
int event_node_flag = 0;
int nc = 0;


int high = 95;
int low = 93;
int u_turn_reverse_speed = 133;
int u_turn_forward_speed = 123;
int sharp_turn_low = 97; //node 13,16,high speed turn part

int counter = 1;
int counter_n = 0;
int counter2 =  0;
int n = 0;

// Motor A
const int enablePinA = 26;
const int in1PinA = 25;
const int in2PinA = 33;

const int enablePinB = 13;
const int in1PinB = 12;
const int in2PinB = 14;   

int s1 = 4;  // Leftmost sensor
int s2 = 21;  // Left sensor
int s3 = 19;  // Center sensor
int s4 = 18;  // Right sensor
int s5 = 5;  // Rightmost sensor

const int pwm_channel_A = 0;
const int pwm_channel_B = 1;
const int frequency = 500;
const int resolution = 8;

int buzzerPin = 27;
int ledred = 22;
int ledgreen = 23; 
int flag = 0;
int s1Val = 0;
int s2Val = 0;
int s3Val = 0;
int s4Val = 0;
int s5Val = 0;

int illa = 1;

void setup(){
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
  }
 
  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());

  pinMode(enablePinA, OUTPUT);
  pinMode(in1PinA, OUTPUT);
  pinMode(in2PinA, OUTPUT);

  pinMode(enablePinB, OUTPUT);
  pinMode(in1PinB, OUTPUT);
  pinMode(in2PinB, OUTPUT);

  pinMode(s1, INPUT);
  pinMode(s2, INPUT);
  pinMode(s3, INPUT);
  pinMode(s4, INPUT);
  pinMode(s5, INPUT);

  ledcSetup(pwm_channel_A, frequency, resolution);
  ledcAttachPin(enablePinA, pwm_channel_A);

  ledcSetup(pwm_channel_B, frequency, resolution);
  ledcAttachPin(enablePinB, pwm_channel_B);

  pinMode(buzzerPin, OUTPUT);
  pinMode(ledred, OUTPUT);
  pinMode(ledgreen, OUTPUT);
  digitalWrite(buzzerPin, HIGH);
  digitalWrite(ledred, LOW);
  digitalWrite(ledgreen, LOW);
}

void forward(int speed) {
  digitalWrite(in1PinA, HIGH);
  digitalWrite(in2PinA, LOW);
  ledcWrite(pwm_channel_A, speed);
  Serial.println("");
  digitalWrite(in1PinB, LOW);  // Reverse the direction for motor B
  digitalWrite(in2PinB, HIGH); // Reverse the direction for motor B
  ledcWrite(pwm_channel_B, speed-5);
}

void left(int speedA, int speedB) {
  digitalWrite(in1PinA, HIGH);
  digitalWrite(in2PinA, LOW);
  ledcWrite(pwm_channel_A, speedA);
  Serial.println("");
  digitalWrite(in1PinB, LOW);
  digitalWrite(in2PinB, LOW);
  ledcWrite(pwm_channel_B, speedB);
}

void left2(int speedA, int speedB) {
  digitalWrite(in1PinA, HIGH);
  digitalWrite(in2PinA, LOW);
  ledcWrite(pwm_channel_A, speedA);
  Serial.println("");
  digitalWrite(in1PinB, HIGH);
  digitalWrite(in2PinB, LOW);
  ledcWrite(pwm_channel_B, speedB);
}

void right2(int speedA, int speedB) {
  digitalWrite(in1PinA, LOW);
  digitalWrite(in2PinA, HIGH);
  ledcWrite(pwm_channel_A, speedA);
  Serial.println("");
  digitalWrite(in1PinB, LOW);
  digitalWrite(in2PinB, HIGH);
  ledcWrite(pwm_channel_B, speedB);
}

void right(int speedA, int speedB) {
  digitalWrite(in1PinA, LOW);
  digitalWrite(in2PinA, LOW);
  ledcWrite(pwm_channel_A, speedA);
  Serial.println("");
  digitalWrite(in1PinB, LOW);
  digitalWrite(in2PinB, HIGH);
  ledcWrite(pwm_channel_B, speedB);
}

void forwardu(int speed) {
  digitalWrite(in1PinA, HIGH);
  digitalWrite(in2PinA, LOW);
  ledcWrite(pwm_channel_A, speed);
  Serial.println("");
  digitalWrite(in1PinB, LOW);  // Reverse the direction for motor B
  digitalWrite(in2PinB, HIGH); // Reverse the direction for motor B
  ledcWrite(pwm_channel_B, 0 > speed-5 ? 0 : speed-5);
}



void adjustLeft() {
  // left(0, high); 
  left(high, 0); 
  Serial.println("");// Left motor speed is 200, right motor speed is 255
}

void adjustLeft2() {
  left2(u_turn_forward_speed, u_turn_reverse_speed); 
  // right2(u_turn_reverse_speed,u_turn_forward_speed);
  Serial.println("");// Left motor speed is 200, right motor speed is 255
}

void adjustRight() {
  // right(high, 0);
  right(0, high);
  Serial.println("");  // Left motor speed is 255, right motor speed is 200
}

void adjustRight2() {
  right2(u_turn_reverse_speed,u_turn_forward_speed);
  Serial.println("");  // Left motor speed is 255, right motor speed is 200
}

void run_start(){
  digitalWrite(ledred,HIGH);
  ledcWrite(buzzerPin, HIGH);
  Serial.println("");
  delay(1000);
  digitalWrite(ledred, LOW);
  digitalWrite(buzzerPin, HIGH);
}

void run_stop(){                       
  digitalWrite(ledred,HIGH);
  digitalWrite(buzzerPin, LOW);
  Serial.println("");
  delay(5000);
  digitalWrite(ledred, LOW);
  digitalWrite(buzzerPin, HIGH);
}

void stop() {
  ledcWrite(pwm_channel_A, 0);
  Serial.println("");
  ledcWrite(pwm_channel_B, 0);
}

void beep(){
  digitalWrite(buzzerPin, LOW);
  Serial.println("");
  delay(1000);
  digitalWrite(buzzerPin ,HIGH);  
}

void read_all(){
  Serial.println("");
  s1Val = digitalRead(s1);
  s2Val = digitalRead(s2);
  s3Val = digitalRead(s3);
  s4Val = digitalRead(s4);
  s5Val = digitalRead(s5);
}

void right_till_detect(){
  read_all();
  Serial.println("");
  forward(high);
  while(s5Val == 0){
    Serial.println("right_turn");
    read_all();
     adjustLeft();
  }
  while(s4Val == 0){
    Serial.println("right_turn");
    read_all();
     adjustLeft();
  }
  while(s3Val == 0){
    Serial.println("x_right_turn");
    read_all();
    adjustLeft();
  }
}

void right_till_detect2(){
  read_all();
  Serial.println("");
  forward(high);
  while(s4Val == 0){
    Serial.println("right_turn");
    read_all();
     adjustLeft2();
  }
  while(s3Val == 0){
    Serial.println("x_right_turn");
    read_all();
    adjustLeft2();
  }
}

void left_till_detect3(){
  read_all();
  forward(1);
  Serial.println("");
  while(s1Val == 0){
    Serial.println("left_turn");
    read_all();
    adjustRight2();
  }
  while(s2Val == 0){
    Serial.println("left_turn");
    read_all();
    adjustRight2();
  }
  while(s3Val == 0){
    Serial.println("x_left_turn");
    read_all();
    adjustRight2();
  }
}

int next(String a){
    int n = a.length();
    int x = counter_n;
    while(counter_n < n && a[counter_n] != ' ')
        counter_n++;
    if(counter_n - x == 1){
        int t = a[counter_n-1] - '0';
        counter_n++;
        return t;
    }

    int t1 = a[counter_n-1] - '0';
    int t2 = a[counter_n-2] - '0';
    counter_n++;
    return t1*1 + t2*10;
}

int get(String a){
    int n = a.length();
    int x = counter_n;
    int y = counter_n;
    while(y < n && a[y] != ' ')
        y++;
    if(y - x == 1){
        int t = a[y-1] - '0';
        return t;
    }
    int t1 = a[y-1] - '0';
    int t2 = a[y-2] - '0';
    return t1*1 + t2*10;
}

int get_next(String a)
{
    int saved_counter_n = counter_n;
    int result = next(a);
    result = next(a);
    counter_n = saved_counter_n;
    return result;
}

int get_prev(String a)
{
    int y = counter_n;
    if (counter_n == 0)
        return -1;
    if (counter_n == 2)
    {
        counter_n = 0;
        int result = get(a);
        counter_n = y;
        return result;
    }
    if (a[counter_n - 3] == ' ')
    {
        counter_n -= 2;
        int result = get(a);
        counter_n = y;
        return result;
    }
    else
    {
        counter_n -= 3;
        int result = get(a);
        counter_n = y;
        return result;
    }
}


void left_till_detect(){
  read_all();
  forward(high);
  Serial.println("");
  while(s1Val == 0){
    Serial.println("left_turn");
    read_all();
    adjustRight();
  }
  while(s2Val == 0){
    Serial.println("left_turn");
    read_all();
    adjustRight();
  }
  while(s3Val == 0){
    Serial.println("x_left_turn");
    read_all();
    adjustRight();
  }
}

void left_till_detect_1(){
  read_all();
  forward(high);
  Serial.println("");
  // while(s1Val == 0){
  //   Serial.println("left_turn");
  //   read_all();
  //   adjustRight();
  // }
  while(s2Val == 0){
    Serial.println("left_turn");
    read_all();
    adjustRight();
  }
  while(s3Val == 0){
    Serial.println("x_left_turn");
    read_all();
    adjustRight();
  }
}

void adjustLeftTill_u1(){
  read_all();
  while(s2Val == 0){
    Serial.println("left_turn");
    read_all();
    adjustRight();
  }
  while(s3Val == 0){
    Serial.println("x_left_turn");
    read_all();
    adjustRight();
  }
}

void adjustLeftTill_u2(){
  read_all();
  while(s1Val == 0){
    Serial.println("left_turn");
    read_all();
    adjustRight();
  }
}


void loop() {

  if (!client.connect(host, port)) {
    Serial.println("Connection to host failed");
    delay(200);
    return;
  }

  while(wait_for_node_8 == 0){
    if(client.available()) {
      msg = client.readStringUntil('z');
      node_path = client.readStringUntil('z');
      if(node_path) Serial.println(node_path);   
      Serial.println(msg); 
      client.flush();
      wait_for_node_8 = 1;
    }
    delay(5);
  }

  n = msg.length();
  String new_message;
  new_message = "f";

  //------------------------------------------init ends, run starts

  delay(5000);
  read_all();
  run_start();
  forward(high);
  Serial.println("Run start");
  
  forward(high);

  while(1){

    Serial.println("at node now: ");
    Serial.println(get(node_path));
    Serial.println(counter);

    if(get(node_path)==13 && get_next(node_path)==16){
      high = 87;
      low = 85;
      u_turn_reverse_speed = 133;
      u_turn_forward_speed = 123;
      sharp_turn_low = 89; //node 13,16,high speed turn part
    }

    if(get(node_path)==16 && get_next(node_path)==17){
      high = 95;
      low = 93;
      u_turn_reverse_speed = 133;
      u_turn_forward_speed = 123;
      sharp_turn_low = 97; //node 13,16,high speed turn part
    }

    // if(get(node_path)==13 && get_next(node_path)==8){
    //   high = 100;
    //   low = 97;
    //   u_turn_reverse_speed = 133;
    //   u_turn_forward_speed = 123;
    //   sharp_turn_low = 103; //node 13,16,high speed turn part
    // }

    if(get(node_path)==17 && get_next(node_path)==15){
      high = 87;
      low = 85;
      u_turn_reverse_speed = 133;
      u_turn_forward_speed = 123;
      sharp_turn_low = 89; //node 13,16,high speed turn part
    }

    if(get(node_path)==12 && get_next(node_path)==7){
      high = 95;
      low = 93;
      u_turn_reverse_speed = 133;
      u_turn_forward_speed = 123;
      sharp_turn_low = 97;
    }

    if(get(node_path)==9 && get_next(node_path)==8){
      high = 87;
      low = 85;
      u_turn_reverse_speed = 133;
      u_turn_forward_speed = 123;
      sharp_turn_low = 89;  //node 13,16,high speed turn part
    }

    if(get(node_path)==4 && get_next(node_path)==1){
      high = 95;
      low = 93;
      u_turn_reverse_speed = 133;
      u_turn_forward_speed = 123;
      sharp_turn_low = 97;
    }

    if(get(node_path) == 16 && get_next(node_path) == 13){
      illa = 0;
    }
    if(get(node_path) == 13 ){
      illa = 1;
    }
    // high = 250;
    // if(get(node_path) == 13 && get_next(node_path) == 16){
    //   high = 230;
    // }
    Serial.println("");
    
    if(client.available()) {
      event_msg = client.readStringUntil('z');
      if(event_msg) Serial.println("event");    
        client.flush();
      if(event_msg == "event")
        event_node_flag = 1;
      else if(event_msg == "node"){
        counter++;
        Serial.println("counter by node mssg from client");
        next(node_path);
      }
    }

    read_all();  

    if(!s2Val && !s3Val && !s4Val){
      Serial.println("");
      read_all();
      if(!illa){
        s1Val = 0;
      }
      if(s1Val){
        if((get(node_path)==13 && get_next(node_path)==16) || (get(node_path)==17 && get_next(node_path)==15)){
          left2(high-10,sharp_turn_low+80);
        }
        else
          adjustLeft();
      }
      else if(s5Val){
        if((get(node_path)==13 && get_next(node_path)==16) || (get(node_path)==16 && get_next(node_path)==13)){
          for(int g = 0;g<=50;g++){
            right2(sharp_turn_low+80,high-10);
          }
        }
        else
          adjustRight();
        // if(!illa){
        //   adjustRight();
        //   adjustRight();
        //   adjustRight();
        // }
      }
      else{ 
        forward(high);
      }     
      if(event_node_flag && msg[counter] == 'u'){

          // delay(93);
          Serial.print("U turn 1");
          stop();
          
          client.print("U TURN");

          digitalWrite(buzzerPin, LOW);
          digitalWrite(ledgreen, HIGH);

          delay(1000);
          
          digitalWrite(ledgreen, LOW);
          digitalWrite(buzzerPin, HIGH);
          int xc = 1;
          new_message = "0";
          while (new_message != "stop"){
            Serial.println("turning");
            if(client.available()){
              // Serial.println("yy");
              new_message = client.readStringUntil('z');
              // Serial.println(new_message);
               
            }
            if(xc){
              forwardu(0);
              xc = 0;
            }
            adjustRight2();
            delay(75);
            forwardu(0);
            delay(37);
            
          }
          digitalWrite(ledgreen, HIGH); 
          digitalWrite(ledred, HIGH); 
          stop();
          delay(50);
          digitalWrite(ledgreen, LOW); 
          digitalWrite(ledred, LOW); 

          int u_node = get_next(node_path);
          if(u_node == 2 || u_node == 9 ){
            adjustLeftTill_u1();
          }
          else{
            adjustLeftTill_u2();
          }          
          if(get_next(node_path) == 16 && get(node_path) == 13 )
          {
            Serial.println("Enter if loop");
            Serial.println("Enter if loop");

            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);

            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);

            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);

          }
          stop();
          delay(50);
          client.flush();
          Serial.println("at node: ");
          Serial.println(get(node_path));
          counter++;
          next(node_path);
          Serial.println("Counter by loop corner with u");
          event_node_flag = 0;
          read_all();
      }


      else if(event_node_flag){

        // delay(95);
        stop();        
        digitalWrite(ledgreen, HIGH);
        digitalWrite(buzzerPin, LOW);
        delay(1000);
        digitalWrite(ledgreen, LOW);
        digitalWrite(buzzerPin, HIGH);
        Serial.println("");
        forward(high);
        Serial.println("at node: ");
        Serial.println(get(node_path));
        counter++ ;
        next(node_path);
        Serial.println("counter by corner without u");
        event_node_flag = 0;
      }
      
    }

    else if(s2Val || s3Val || s4Val){
      read_all();
      Serial.println("");
      if(event_node_flag && msg[counter] == 'u'){

          // delay(95);
          Serial.print("U turn 2");
          stop();
          
          client.print("U TURN");
          digitalWrite(buzzerPin, LOW);
          digitalWrite(ledgreen, HIGH);
          
          delay(1000);

          digitalWrite(ledgreen, LOW);
          digitalWrite(buzzerPin, HIGH);

          int xc = 1;
          new_message = "0";
          while (new_message != "stop"){
            if(client.available()){
              // Serial.println("yy");
              new_message = client.readStringUntil('z');
              // Serial.println(new_message);
            }
            if(xc){
              forwardu(0);
              xc = 0;
            }
            adjustRight2();
            delay(75);
            forwardu(0);
            delay(37);
          }
          digitalWrite(ledgreen, HIGH); 
          digitalWrite(ledred, HIGH); 
          stop();
          delay(50);
          digitalWrite(ledgreen, LOW); 
          digitalWrite(ledred, LOW); 

          int u_node = get_next(node_path);
          if(u_node == 2 || u_node == 9){
            adjustLeftTill_u1();
          }
          else{
            adjustLeftTill_u2();
          }     
          if(get_next(node_path) == 16 && get(node_path) == 13 )
          {
            Serial.println("Enter if loop");
            Serial.println("Enter if loop");
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);

            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);

            adjustLeft2();
            delay(10);
            adjustLeft2();
            delay(10);

          }
          stop();
          delay(50); 

          client.flush();
          Serial.println("at node: ");
          Serial.println(get(node_path));
          counter++;
          next(node_path);
          Serial.println("counter by middle with u");
          event_node_flag = 0;
      }


      else if(event_node_flag){
        
        // delay(95);
        stop();       
        digitalWrite(ledgreen, HIGH); 
        digitalWrite(buzzerPin, LOW);
        delay(1000);
        digitalWrite(ledgreen, LOW);
        digitalWrite(buzzerPin, HIGH);
        Serial.println("");
        forward(high);
        Serial.println("at node: ");
        Serial.println(get(node_path));
        counter++ ;
        next(node_path);
        Serial.println("counter by middle without u");
        event_node_flag = 0;
      }


      bool decision = (s3Val && s2Val && s4Val);
      // int decision_node = get_next(node_path);
      // if(decision_node == 1 || decision_node == 13 ||decision_node == 4 ||decision_node == 8 || decision_node == 7 || decision_node == 12 ||decision_node == 15){
      //   decision = ((s3Val && s4Val) || (s3Val && s2Val));
      // }


      if(decision){
          digitalWrite(ledgreen, HIGH);
          //how much to delay
            if(counter==1){
              delay(250);
              stop();
              delay(50);
              forward(high);
            }
            else if(get(node_path)==11 && get_next(node_path)==10){
              delay(125);
              stop();
              delay(50);
              forward(high);
            }
            else if(get(node_path)==6 && get_next(node_path)==5){
              delay(180);
              stop();
              delay(50);
              forward(high);
            }
            else if(get(node_path)==9 && get_next(node_path)==10){
              delay(350);
              stop();
              delay(50);
              forward(high);
            }
            else if(get(node_path) == 9 && get_next(node_path) == 8){
              delay(50);
              stop();
              delay(1000);
              forward(high);
            }
            else if(get_next(node_path) == 7){
              delay(150);
              stop();
              delay(100);
              forward(high);
            }
            else{
              delay(180);
              stop();
              delay(50);
              forward(high);
            }
          //end of how much to delay
          digitalWrite(ledgreen, LOW);
          //what to do at the node?
            if(msg[counter] == 's'){
              Serial.println("S");
              forward(high);
            }

            else if(msg[counter] == 'r'){
                right_till_detect();
                Serial.println("right till detect");
            }

            else if(msg[counter] == 'l' && get_next(node_path) == 1){
              left_till_detect_1();
              Serial.println("left till detect");
            }

            else if(msg[counter] == 'l'){
              left_till_detect();
              Serial.println("left till detect");
            }
            else
              stop();
          //end of what to do at the node

          if(counter >=  (n-1)){
            forward(high);
            read_all();
            flag = 1;
          }
          
        Serial.println("at node: ");
        Serial.println(get(node_path));
          counter++;
          next(node_path);
          Serial.println("Counter by middle node");
          Serial.println("node");
      }

      else if(s2Val && !s3Val){
        adjustRight();
      }

      else if(s4Val && !s3Val){
        adjustLeft();
      }
      else{
        forward(high);
      }
    }


    else{
      Serial.println("");
      forward(high);
    }


    //////slow stop
    if(flag == 1){
      counter2 += 1;
    }
    if(counter2 == 520){
      stop();
      run_stop();
      while(1){
        Serial.println("");
      }
    }
    //end of slow stop code

  } 
}



//e: 
//b: aage too kich 
//aaage too much
//daage roo mucvh
//c