#include <SPI.h>
#include <Servo.h> 

byte buf [5];
byte buf2[4];
byte sync[3];
int flag=0;
volatile byte pos;
volatile boolean process_it;
int acc=0,dcc=0,diif=0;
int spdl=0,spdr=0;
Servo myservol;
Servo myservor;

union data
{
   uint16_t a;
   byte b[2];
};

data readrel;



void setup (void)
{
  Serial.begin (250000);

  myservol.attach(9);
  myservor.attach(8);

  // have to send on master in, *slave out*
  pinMode(MISO, OUTPUT);
  digitalWrite(SS, LOW);
  // 设置为接收状态
  SPCR |= _BV(SPE);

  // 准备接受中断
  pos = 0;   // 清空缓冲区
  process_it = false;

  // 开启中断
  SPI.attachInterrupt();
  Serial.println (SPCR);
}


// SPI 中断程序
ISR (SPI_STC_vect)
{
  byte c = SPDR;  // 从 SPI 数据寄存器获取数据
  //Serial.println (c);
 if (flag==0)
 {
  sync[0]=sync[1];
  sync[1]=sync[2];
  sync[2]=c;

// Serial.print (" ");
// Serial.print (sync[2]);

  
  if (sync[0]==10 && sync[1]==80 && sync[2]==10){
  flag=1;
//  Serial.println ("  flaged");
  }
  
  }

  
  if (flag==1){
//   Serial.println ("processing ");
    buf [pos++] = c;
    if (pos==5){
      process_it = true;
      flag=0;}
  }
}

void loop (void)
{float angle=0;
  if (process_it)
  {
    for (int i=0;i<4;i++)
    {buf[i]=buf[i+1];}  
    readrel.b[1]=buf[0];
    readrel.b[0]=buf[1];
    angle = readrel.a;
    angle = angle/65535*360-180;
    acc = map(buf[2], 0, 256, 0, 270);
    dcc = map(buf[3], 0, 256, 0, 180);
    acc=270-acc;
    dcc=180-dcc;
    
    
    spdl=angle+acc-dcc+90;
    if (spdl>180)
    spdl=180;
    if (spdl<0)
    spdl=0;
    spdr=-angle+acc-dcc+90;
    if (spdr>180)
    spdr=180;
    if (spdr<0)
    spdr=0;
   
    myservol.write(spdl);
    myservor.write(spdr);

    if (angle>0){
    Serial.print ("  ");Serial.print(abs(angle));Serial.print ("__\t");}
    else {
    Serial.print ("__");Serial.print(abs(angle));Serial.print ("  \t");}
    Serial.print ("  acc: ");Serial.print(acc/2.7,0);Serial.print ("%  dcc:");Serial.print(dcc/1.8,0);Serial.println ("%");
    
    pos = 0;
    process_it = false;
  } 
}
