# Assignment3 : Stock Prediction
Sequence Data with RNN Simple/RNN Encoder-Decoder/LSTM/GRU/CNN combined with LSTM


# Highlight
-  Pretrained Model ทางกลุ่มได้เเบ่งข้อมูลเป็น Train 80%, Validation 10%, Test 10% โดยการ Train ข้อมูลนั้น ทางกลุ่มได้นำราคาปิดของหุ้นที่เลือกในเเต่ละวัน มาเเปลงเป็น daily return(% Change) เพื่อทำให้ข้อมูลมีความเป็น Stationary
-  ทุก Model ทางกลุ่มได้ทำการ Train โมเดล เพื่อวัตถุประสงค์ให้ได้ผลลัพท์ในการทำนายราคาหุ้นสูงที่สุด โดยใช้ Mean Squared Error เเละ Mean Absolute Error ในการประเมินความเเม่นยำ(ใช้ Test Period อยู่ที่ 244 วัน)
-  หากต้องการนำไปพัฒนาต่อ ทางกลุ่มเเนะนำว่า เราสามารถใช้ ผลตอบเเทนในช่วงระยะเวลาอื่นๆ เช่น Weekly, Monthly เเละในส่วนของโมเดล เราพบว่าการใช้ CNN+LSTM Architecture หรือการประยุกต์ใช้ encoder-decoder นั้นให้ผลลัพท์ที่น่าพอใจ เเละสามารถนำไปต่อยอดได้

# Introduction
ข้อมูลส่วนมากมักจะมีรูปแบบเป็น sequence data เช่น ข้อความ, ข้อความเสียง, รหัสพันธุกรรม, วิดีโอ หรือ ข้อมูลอนุกรมเวลา (time-series data) จำพวกราคาหุ้น การจะนำข้อมูลเหล่านี้ไปใช้เพื่อวิเคราะห์ หรือใช้เพื่อทำนายผลอะไรบางอย่างจะไม่สามารถใช้ Traditional Neural Network (Traditional NN) model เข้ามาสร้างผลการวิเคราะห์หรือทำนายในส่วนนี้ได้ เนื่องจากการสร้าง model ที่มีรูปแบบ input และ output เป็น sequence ตัว model จะต้องสามารถเรียนรู้และส่งผ่านความรู้จาก input ไปจนถึง output เนื่องด้วยข้อมูลลักษณะ sequence data ข้อมูล input ก่อนหน้าจะมีความสำคัญกับการทำนาย output ที่เราต้องการในลำดับถัดไป เช่น หากเราต้องการทำนายคำลำดับถัดไปในข้อความบริบทหนึ่ง อย่างน้อยเราก็จำเป็นที่จะต้องทราบคำอย่างน้อย 2-3 คำก่อนหน้าในบริบทนั้นๆ เสียก่อนถึงจะสามารถทำนายคำถัดไปได้ ซึ่ง Traditional NN ไม่ได้ถูกออกแบบมาให้มี memory เพียงพอที่จะใช้สำหรับการทำงานในลักษณะนี้ เพราะฉะนั้นเราจึงต้องใช้ sequence model ที่ถูกออกแบบมาสำหรับงานในลักษณะนี้โดยเฉพาะ ซึ่งก็มีตั้งแต่ RNN, LSTM, GRU และ Transformer model ในงานชิ้นนี้เราจะทำการทดลองใช้ sequence model ทั้งหมด 4 แบบ และอีก 1 traditional ML model กับข้อมูลที่เป็น sequence data คือข้อมูลราคาหุ้นบริษัท บางกอก เชน ฮิสปิทอล จำกัด (มหาชน) (BCH) จุดประสงค์ก็คือเพื่อทดลองทำนายราคาหุ้น BCH ในวันถัดไปโดยใช้ราคาหุ้นในวันก่อนหน้าเป็น input data ให้ model เรียนรู้ เมื่อทดลองเสร็จจะทดสอบประสิทธิภาพในการทำนายของแต่ละ model เปรียบเทียบกันในลำดับถัดไป

# Data
ข้อมูลที่ใช้ในการศึกษาเป็นข้อมูลราคาหุ้นบริษัท บางกอก เชน ฮอสปิทอล จำกัด (มหาชน) หรือตัวย่อ BCH ซึ่งจดทะเบียนอยู่ในตลาดหลักทรัพย์แห่งประเทศไทยในหมวดธุรกิจการแพทย์ ข้อมูลที่ใช้ได้แก่ ราคาปิด ราคาเปิด ราคาสูงสุด ราคาต่ำสุด และปริมาณการซื้อขาย ซึ่งเป็นข้อมูลรายวัน ตั้งแต่วันที่ 9 กุมภาพันธ์ 2555 ถึงวันที่ 23 มีนาคม 2565 คิดเป็นจำนวนข้อมูลทั้งหมด 2,469 วัน แหล่งข้อมูลนำมาจาก Bloomberg ซึ่งเป็นผู้ให้บริการข้อมูลทางการเงินที่น่าเชื่อถือ โดยข้อมูลมีการปรับปรุงการเปลี่ยนแปลงของราคาพาร์ให้เป็นพาร์ปัจจุบัน ทำให้สามารถเปรียบเทียบราคาหุ้นในอดีตกับปัจจุบันได้ดียิ่งขึ้น
โดยเมื่อนำข้อมูลราคาปิดของหุ้น BCH ในแต่ละวันมาสร้างกราฟดูพบว่าข้อมูลมีลักษณะมีแนวโน้ม(trend) และฤดูกาล(seasonal) ทางทีมจึงแก้ปัญหาดังกล่าวด้วยการใช้อัตราผลตอบแทนในการพยากรณ์แทนการใช้ราคาปิด แล้วนำผลตอบแทนที่ได้จากการพยากรณ์มาคำนวณราคมอีกครั้ง หลังจากนั้นแบ่งข้อมูลออกเป็น 3 ส่วนคือ 
1) ข้อมูลสำหรับ Training โมเดลจำนวน 1,958 วัน
2) ข้อมูลสำหรับ Validate โมเดลจำนวน 246 วัน
3) ข้อมูลสำหรับ Test โมเดลจำนวน 244 วัน

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/87576892/160978428-79ef1040-ef96-4368-8f62-ddba0c5501c5.png">
</p>


# Training Strategy
ในส่วนของการทำ Training Strategy นั้น ทางทีมจะทำการแบ่งออกเป็นทั้งหมด 3 sections: Deep Learning Model, Traditional Model และ Traditional indicator โดยจะแบ่งรูปแบบของการเปรียบเทียบ result ดังนี้

Section1: จะทำการคำนวนค่า MSE และ MAE ของแต่ละ models ของ deep learning: RNN Simple, LSTM, GRU, CNN+LSTM และ RNN Encoder-Decoder โดยทางกลุ่มจะทำการแบ่ง train set, validation set และ test set เป็นดังนี้ 0.8,0.1,0.1 คามลำดับ จากนั้นจะนำมาทำการเปรียบเทียบกันและทำการเลือก model ที่ให้ค่า MSE และ MAE น้อยสุดมา

Section2: จะทำการคำนวนค่า MSE และ MAE ของ traditional model: LGBM

Section3: จะทำการคำนวนค่า MSE และ MAE ของ basic indicator: Moving Average


<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/71161635/160892555-e92f90e2-a453-4dcc-946c-0863f687de53.png">
</p>

ซึ่งหลังจากที่ทางทีมได้ the best model จาก section1 มานั้น จะนำมาทำการเปรียบเทียบกับ result ที่ได้จาก Section2 และ Section3 เพื่อที่จะได้ทำการ compare ว่า result ที่ได้จากการ run deep learning model มี performance ที่ดีกว่าตัว traditional model หรือ basic indicator ไหม


# Result 
จากการเปรียบเทียบผลการการพยากรณ์ราคาหุ้น BCH ในเวลา t+21 ของโมเดลทั้ง 7 ตัวพบว่าโมเดลที่ให้ผลพยาการณ์ดีที่สุดคือ **RNN แบบ Encoder-Decoder โดยมี MSE - 0.611 และ MAE - 0.614** และโมเดลที่ได้ผลดีรองลงมาคือโมเดล **CNN+LSTM ที่มี MSE - 0.995 และ MAE - 0.804** ในขณะที่โมเดลที่เหลือจะมีผลแย่กว่าการใช้ moving average ในการพยาการณ์ รายละเอียดเพิ่มเติมตามรูปภาพด้านล่างนี้
 <p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/87576892/160893403-7ff64979-a089-4736-ac12-a4f977f24618.png">
  <img width="800" src="https://user-images.githubusercontent.com/87576892/160887534-8fe6fa7a-e3c4-483d-810b-87324c1ecdc5.png">
</p>

# Network Architecture and Result
## Traditional ML - LightGBM 
 ทางทีมได้เลือก LightGBM เนื่องจากเป็นโมเดลที่มีความเร็ว และมีประสิทธิ์ภาพที่สูง จากการใช้ Gradient Boosting หรือ เทคนิคการเรียนรู้ที่จะสร้างโมเดลที่มีความแม่นยำสูง โดยเรียนรู้จากค่าความคลาดเครลื่อนสะสมที่เกิดจากการทำนายของโมเดลที่สร้างก่อนหน้า โดยทางทีมได้กำหนด parameter ต่างๆ ไว้ดังนี้ { 'boosting_type': 'gbdt',  'objective': 'regression', 'metric':'l2', 'num_leaves':10, 'max_depth':5, 'drop_rate ':0.3,  'reg_sqrt':True,  'boost_from_average':True,  'learning_rate': 0.0001, 'verbose': 0, } และตั้งค่าในการ train ไว้เป็น num_boost_round=1000,  early_stopping_rounds=100, verbose_eval=50 เมื่อนำโมเดลที่ Train เสร็จเรียบร้อยแล้วมาทดสอบกับข้อมูล test set (ข้อมูลแยกไว้เพื่อทดสอบโมเดลในตอนสุดท้าย) ได้ผล Mean Squared Error (MSE) : 3.31 และ Mean Absolute Error : 1.51
![image](https://user-images.githubusercontent.com/87576892/161177914-d0859ad7-0e6b-4998-94f0-9e82006dab0e.png)


## RNN Simple
โมเดลถัดมาที่ทางทีมเลือกมาทดลองคือ Simple Recurrent Neural Network (RNN Simple) ซึ่งเป็น Network ที่นำ Output จาก State ที่แล้วมาร่วมเป็น Input ด้วย ซึ่งการทำงานจะคล้ายกับการทำงานเป็น Loop เหมือนกับ Neural Network ธรรมดาที่มีหลายๆ ตัว โดยมี Output ต่อกันเป็น Input เข้า Network ใหม่ และเนื่องจาก RNN ใช้ข้อมูลจาก Network ก่อนๆ ทำให้สามารถทำงานได้ดีในข้อมูลแบบ Time Series ซึ่งเหมาะสมกับข้อมูลราคาหุ้น BCH ที่นำมาใช้


![image](https://user-images.githubusercontent.com/71161635/161076262-624a63c3-1cfa-438f-96d3-b25de3e2b6e8.png)


จากการรันโมเดล RNN Simple พบว่าค่า MSE = 1.7033 และค่า MAE = 1.0598 และมีเส้นพยากรณ์ตามรูปด้านล่าง ซึ่งจากกราฟจะพบว่าโมเดลสามารถทำนายราคาหุ้น BCH ได้ค่อนข้างแม่นยำในช่วงราคาที่มีแนวโน้มขาลง แต่ในขณะที่ราคามีแนวโน้มขึ้น โมเดลจะมีความแม่นยำต่ำลง 


![image](https://user-images.githubusercontent.com/71161635/161076417-bbe19ec0-64ae-4a6a-8957-26a4acb001ba.png)


![image](https://user-images.githubusercontent.com/71161635/161076431-46323353-d9d4-4b09-af66-f5c750e7d109.png)


![image](https://user-images.githubusercontent.com/71161635/161076456-3e3275fb-4f1a-4337-ae82-c5fa4764d6c5.png)



## LSTM
เราได้นำ LSTM มาทดลองใช้กับการพยากรณ์ราคาหุ้น BCH โดยใช้ ราคาปิดมาใช้ในการพยากรณ์ โดยเราได้นำราคาปิด รายวันของหุ้นดังกล่าว มาหาผลตอบแทนรายวันเพื่อให้ข้อมูลมีความ Stationary จากนั้นจึงนำข้อมูลมาทำการ Scaling โดยเราได้เลือกการใช้ StandardScaler ในการแปลงข้อมูล ก่อนที่จะนำข้อมูลนั้นเข้าไป Train ในตัวของโมเดล โดยมี Architecture ของตัวโมเดล ดังนี้
 ![lstm-1](https://user-images.githubusercontent.com/98243238/160967748-d7ab47eb-4c03-4199-872c-2ef79f7ab1fe.JPG)

โดยจากการทดลองจะเห็นได้ว่าตัวแบบสามารถทำนายได้ค่อนข้างไกล้เคียงกับราคาหุ้น ในช่วง period เริ่มต้น ก่อนที่จะมีการ deviate จากราคาที่แท้จริงค่อนข้างมากในช่วงเวลาต่อๆ มา โดยโมเดลนั้นมีค่า MAE, MSE เมื่อเทียบระหว่างราคาปิดของแต่ละวันอยู่ที่ 4.41 และ 22.04 ตามลำดับ
 ![lstm-2](https://user-images.githubusercontent.com/98243238/160967846-9d4a6619-f689-448b-a365-0fd6a58e6261.JPG)


## GRU
Gated Recurrent Unit (GRU) เป็น Recurrent Neural Network (RNN) แบบหนึ่ง ที่ออกแบบมาคล้ายๆ Long Short Term Memory (LSTM)  LSTM 
แต่จะช่วยแก้ปัญหาเรื่อVanishing Gradient / Exploding Gradient เพราะประสิทธิภาพของ RNN จะแย่ลงถ้าเจอกับ Sequence ยาว ๆ ทำให้มีการออกแบบ GRU ขึ้นมาแก้ปัญหาเหล่านี้
โดยโครงสร้างของ GRU จะมีระบบปิดเปิดการอัพเดทสถานะภายใน RNN ที่คล้ายกับ (LSTM) ที่จะมี Forget Gate แต่มี Parameter น้อยกว่า LSTM 
เนื่องจากไม่มี Output Gate GRU มีประสิทธิภาพใกล้เคียงกับ LSTM ในหลาย ๆ งาน แต่เนื่องจาก Parameter น้อยกว่าทำให้เทรนได้ง่ายกว่า 
เร็วกว่า และในบางงานที่ Data Set มีขนาดเล็ก พบว่า GRU ประสิทธิภาพดีกว่า

![image](https://user-images.githubusercontent.com/83268624/160988210-6239bb5b-5625-4d45-b5d7-077c164c6a52.png)


Picture Credit: https://en.m.wikipedia.org/wiki/File:Gated_Recurrent_Unit.svg


Ref: https://www.bualabs.com/archives/3103/what-is-rnn-recurrent-neural-network-what-is-gru-gated-recurrent-unit-teach-how-to-build-rnn-gru-with-python-nlp

      การที่ GRU มีความเร็วที่มากกว่าLSTM เพราะมีการตัด input และ output gate ออกและเปลี่ยนมาใช้ reset gate และ update gate 
      โดยที่ข้อมูล input เพื่อเข้ามาครั้งแรกจะทำการเก็บค่านั้นไว้แล้วจะตัดดสินใจว่าจะนำข้อมูลไปใช้ในใหม่ GRU หรือ แค่แสดงผล 
      หากนำไปใช้จะเข้าส่งข้อมูลไปที่ reset gate เพื่อที่จะตัดสินใจว่าต้องลบข้อมูลไหน และเก็บข้อมูลก่อน (ธนดล สิงขรอาสน์ ,2564) ***
      *** วิทยานิพนธ์: การเรียนรู้เชิงลึกสำหรับการตรวจจับและรู้จำคำบรรยายในวีดิทัศน์ (ธนดล สิงขรอาสน์ ,2564)

ในที่นี้ทางทีมได้ออกแบบเปรียบเทียบค่าการพยากรหุ้นของโรงพยาบาล  BCH บริษัท บางกอก เชน ฮอสปิทอล จำกัด (มหาชน)ด้วยหลักการ GRU 
ซึ่งเป็นหนึ่งในรูปแบบสถาปัตถรยกรรมของ RNN และโดยกำหนดให้มี parameter ที่เหมือนกันดังนี้ Learning rate ที่ 0.001  , epoch 30 epoch , 
batch size 64 ,ข้อมูลราคา/return ที่ใช้ในการพยากรณ์ล่วงหน้า 20 วัน และ drop out ที่ 0.3 และ Optimizer เป็น Nadam 
โดยแบ่งข้อมูลเป็น Trian 80 % Validation 10 % และข้อมูล Test 10 % 

![image](https://user-images.githubusercontent.com/83268624/160992117-c58bce3f-702d-48fe-a0c6-66d03cf55bd4.png)

![image](https://user-images.githubusercontent.com/83268624/160992163-687efc59-e0a5-4b5e-8b05-0e183da9bad1.png)

พบว่า GRU  มีค่า MSE  = 3.6008    ,  MAE  = 1.765  และหน้าตาการพยากรณ์ราคาที่ได้เป็นไปตามรูปด้านล่างซึ่งมีทิศทางที่ใกล้เคียงกับความเป็นจริงของราคา
ทว่าความแม่นยำในเรื่องราคานั้นค่อนที่ต่ำกว่าราคาจริง ( Gap price ) และมีประสิทธิภาพด้วยกว่า RNN ประเภทอื่นๆยกเว้น LSTM 

![image](https://user-images.githubusercontent.com/83268624/160992275-e3990aa0-977b-41e1-838f-0e48bc61fd4b.png)

## RNN Encoder-Decoder
  อีกหนึ่งโมเดลที่ทางกลุ่มเลือกนำมาทดลองคือ RNN Encoder-Decoder architecture เพื่อช่วยแก้ปัญหาจากการใช้แค่ Simple RNN network เนื่องจากการที่ input เป็น sequence และ output เป็น sequence เป็นงานที่ซับซ้อนเกินไปสำหรับแค่ Simple RNN network การใช้ Encoder-Decoder ตัวโมเดลจะถูกแบ่งออกเป็น 2 ส่วนช่วยกันทำงานคือส่วน Encoder ที่รับ input เป็น sequence และส่วน Decoder ที่จะให้ output ออกมาเป็น sequence เช่นกัน ทำให้สามารถทำงานได้ดีกว่า ซึ่งการตั้งค่า parameter ในการทดลองครั้งนี้ กำหนดจำนวน cells สำหรับ encoder และ decoder โมเดลไว้เท่ากับ 20 และเพิ่ม dropout ในส่วน decoder โมเดลเท่ากับ 0.3 ซึ่งเมื่อนำโมเดลที่ train เสร็จเรียบร้อยแล้วมาทดลองทำนายราคาของหุ้น BCH (ในส่วนของ test set) พบว่าได้ผล Mean Squared Error (MSE) : 0.6113 และ Mean Absolute Error : 0.6136
  ![image](https://user-images.githubusercontent.com/87868128/160894484-1acac958-c2d6-48af-9630-b5f438488eb2.png)
  ![image](https://user-images.githubusercontent.com/87868128/160894745-0eacd64a-ee50-4de0-87b8-9851ede145f1.png)

## CNN combined with LSTM
ทางทีมได้นำความรู้ที่ได้จากการอ่าน paper ของ Eapen, J., Bein, D., & Verma, A. (2019). Novel deep learning model with cnn and bi-directional lstm for improved stock market index prediction. In 2019 IEEE 9th Annual Computing and Communication Workshop and Conference มาทำการประยุกต์ใช้โดยนำ CNN มา combine รวมกันกับ bi-directional LSTM เพื่อที่จะพัฒนาผลลัพธ์ของการทำนาย return ของ BCH ให้มีความแม่นยำยิ่งขึ้นกว่าการใช้ LSTM โดยทางทีมได้ทำการใช้ 1D Convolutional layer ทั้งหมด 3 layers โดยแต่ละตัวมี nodes ดังนี้ 128, 256 และ 512 nodes ตามลำดับ โดยในแต่ละ 1D convolutional layer นั้นจะตามด้วย max pooling layer ซึ่งตัว output จะทำการ fatten ก่อนที่จะ feed เข้าสู่ bi-directional LSTM โดยในแต่ละ layer ของ  bi-directional LSTM  จะทำการเพิ่ม dropout เท่ากับ 0.3 ดังตัวอย่าง ดังรูปด้านล่าง



<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/71161635/160901701-8768968d-4e12-4079-9a98-28858a1e584f.png">
</p>

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/71161635/160901760-ca503c19-c131-4970-9be7-a16a2e18a9ce.png">
</p>

ซึ่งเมื่อนำโมเดลที่ train เสร็จเรียบร้อยแล้วมาทดลองทำนายราคาของหุ้น BCH (ในส่วนของ test set) พบว่าได้ผล Mean Squared Error (MSE) : 0.9945 และ Mean Absolute Error : 0.8041

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/71161635/160902211-252bd818-c752-426c-835f-3297d86994d5.png">
</p>

# Referance
[1] Eapen, J., Bein, D., & Verma, A. (2019). Novel deep learning model with cnn and bi-directional lstm for improved stock market index prediction. In 2019 IEEE 9th Annual Computing and Communication Workshop and Conference (CCWC) (pp. 0264{0270). doi:10.1109/CCWC.2019.8666592.

# Member
1) ณัฐภณ อัศวเหม 6310422052 (% contribution in this homework: 16.67%)
<br>LSTM , Highlight
2) ดวงธิดา แซ่แต้ 6310422056 (% contribution in this homework: 16.67%)
<br>RNN Simple, DATA
3) เมธี ประเสริฐกิจพันธุ์ 6310422053 (% contribution in this homework: 16.67%)
<br>CNN combined with LSTM , Training Strategy  
4) พีรพัทธ ตั้งไพบูลย 6310422024 (% contribution in this homework: 16.67%)
<br>RNN Encoder-Decoder, Introduction
5) วิชิต ชำนาญนาวา 6310422055 (% contribution in this homework: 16.67%)
<br>GRU 
6) ไตรทิพย์ ศุภศิริวัฒนา 6310422009 (% contribution in this homework: 16.67%)
<br>LightGBM , Summary Training result  
