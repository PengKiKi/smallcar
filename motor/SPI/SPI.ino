

typedef struct rec
{
     float  gas=1.0;
     float  stopper=2.0;
     float  steering=3.0;

}rec;
rec *re;
int sync=0;
byte *comchar=(byte*)&re;
int t;


  void setup(){

    
     Serial.begin(115200);
     
      while(Serial.read()>= 0){}//clear serial port  
  }

  void loop(){
    comchar=(byte*)&re;
    while (Serial.available() > 0);
    
      if (Serial.read() == 255 )
        {
          while (Serial.available() > 0);
          if (Serial.peek() == 254 )
          {
            Serial.read();
            sync=1;
          }
        }
        
     while (sync)
     {
      for(int i=0; i < 12; i++){
      while (Serial.available() > 0);
      *comchar=Serial.read(); 
      comchar++;
     }
     sync=0;
     
     }
    
Serial.println(re->gas,HEX);
    
  }
