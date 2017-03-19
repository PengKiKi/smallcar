#include <SPI.h>

byte buf [5];
byte buf2[4];
byte sync[3];
int flag=0;
volatile byte pos;
volatile boolean process_it;

union data
{
   uint16_t a;
   byte b[2];
};

data readrel;



void setup (void)
{
  Serial.begin (250000);

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
    if (angle>0){
    Serial.print ("  ");
    Serial.print(abs(angle));
    Serial.print ("__\t");
    }
    else
    {
    Serial.print ("__");
    Serial.print(abs(angle));
    Serial.print ("  \t");
    }
    Serial.print (" ---  acc: ");
    Serial.print(buf[2]);Serial.print ("  dcc:");
    Serial.println(buf[3]);
    
    pos = 0;
    process_it = false;
  } 
}
