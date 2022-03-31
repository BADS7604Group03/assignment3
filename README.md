# Assignment3 : Stock Prediction
Sequence Data with RNN Simple/RNN Encoder-Decoder/LSTM/GRU/CNN combined with LSTM


# Highlight
-  Pretrained Model ทางกลุ่มได้เเบ่งข้อมูลเป็น Train 80%, Validation 10%, Test 10% โดยการ Train ข้อมูลนั้น ทางกลุ่มได้นำราคาปิดของหุ้นที่เลือกในเเต่ละวัน มาเเปลงเป็น daily return(% Change) เพื่อทำให้ข้อมูลมีความเป็น Stationary
-  ทุก Model ทางกลุ่มได้ทำการ Train โมเดล เพื่อวัตถุประสงค์ให้ได้ผลลัพท์ในการทำนายราคาหุ้นสูงที่สุด โดยใช้ Mean Squared Error เเละ Mean Absolute Error ในการประเมินความเเม่นยำ(ใช้ Test Period อยู่ที่ 244 วัน)
-  หากต้องการนำไปพัฒนาต่อ ทางกลุ่มเเนะนำว่า เราสามารถใช้ ผลตอบเเทนในช่วงระยะเวลาอื่นๆ เช่น Weekly, Monthly เเละในส่วนของโมเดล เราพบว่าการใช้ CNN+LSTM Architecture หรือการประยุกต์ใช้ encoder-decoder นั้นให้ผลลัพท์ที่น่าพอใจ เเละสามารถนำไปต่อยอดได้

# Introduction


# Data
ข้อมูลที่ใช้ในการศึกษาเป็นข้อมูลราคาหุ้นบริษัท บางกอก เชน ฮอสปิทอล จำกัด (มหาชน) หรือตัวย่อ BCH ซึ่งจดทะเบียนอยู่ในตลาดหลักทรัพย์แห่งประเทศไทยในหมวดธุรกิจการแพทย์ ข้อมูลที่ใช้ได้แก่ ราคาปิด ราคาเปิด ราคาสูงสุด ราคาต่ำสุด และปริมาณการซื้อขาย ซึ่งเป็นข้อมูลรายวัน ตั้งแต่วันที่ 9 กุมภาพันธ์ 2555 ถึงวันที่ 23 มีนาคม 2565 คิดเป็นจำนวนข้อมูลทั้งหมด 2,469 วัน แหล่งข้อมูลนำมาจาก Bloomberg ซึ่งเป็นผู้ให้บริการข้อมูลทางการเงินที่น่าเชื่อถือ โดยข้อมูลมีการปรับปรุงการเปลี่ยนแปลงของราคาพาร์ให้เป็นพาร์ปัจจุบัน ทำให้สามารถเปรียบเทียบราคาหุ้นในอดีตกับปัจจุบันได้ดียิ่งขึ้น
โดยเมื่อนำข้อมูลราคาปิดของหุ้น BCH ในแต่ละวันมาสร้างกราฟดูพบว่าข้อมูลมีลักษณะมีแนวโน้ม(trend) และฤดูกาล(seasonal) ทางทีมจึงแก้ปัญหาดังกล่าวด้วยการใช้อัตราผลตอบแทนในการพยากรณ์แทนการใช้ราคาปิด แล้วนำผลตอบแทนที่ได้จากการพยากรณ์มาคำนวณราคมอีกครั้ง

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/87576892/160892971-b7cd7b35-9e4a-4d36-96cc-69606148db58.png">
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
  <img width="800" src="https://user-images.githubusercontent.com/87576892/160893403-7ff64979-a089-4736-ac12-a4f977f24618.png">
  <img width="800" src="https://user-images.githubusercontent.com/87576892/160887534-8fe6fa7a-e3c4-483d-810b-87324c1ecdc5.png">
</p>

# Network Architecture and Result
## Traditional ML - LightGBM 
 ทางทีมได้เลือก LightGBM เนื่องจากเป็นโมเดลที่มีความเร็ว และมีประสิทธิ์ภาพที่สูง จากการใช้ Gradient Boosting หรือ เทคนิคการเรียนรู้ที่จะสร้างโมเดลที่มีความแม่นยำสูง โดยเรียนรู้จากค่าความคลาดเครลื่อนสะสมที่เกิดจากการทำนายของโมเดลที่สร้างก่อนหน้า โดยทางทีมได้กำหนด parameter ต่างๆ ไว้ดังนี้ { 'boosting_type': 'gbdt',  'objective': 'regression', 'metric':'l2', 'num_leaves':10, 'max_depth':5, 'drop_rate ':0.3,  'reg_sqrt':True,  'boost_from_average':True,  'learning_rate': 0.0001, 'verbose': 0, } และตั้งค่าในการ train ไว้เป็น num_boost_round=1000,  early_stopping_rounds=100, verbose_eval=50 เมื่อนำโมเดลที่ Train เสร็จเรียบร้อยแล้วมาทดสอบกับข้อมูล test set (ข้อมูลแยกไว้เพื่อทดสอบโมเดลในตอนสุดท้าย) ได้ผล Mean Squared Error (MSE) : 0.207 และ Mean Absolute Error : 0.335 
![image](https://user-images.githubusercontent.com/87576892/160876781-285c2681-db7c-476d-87b1-f771f6b692e8.png)

## RNN Simple
RNN write here ---> 
## LSTM
เราได้นำ LSTM มาทดลองใช้กับการพยากรณ์ราคาหุ้น BCH โดยใช้ ราคาปิดมาใช้ในการพยากรณ์ โดยเราได้นำราคาปิด รายวันของหุ้นดังกล่าว มาหาผลตอบแทนรายวันเพื่อให้ข้อมูลมีความ Stationary จากนั้นจึงนำข้อมูลมาทำการ Scaling โดยเราได้เลือกการใช้ StandardScaler ในการแปลงข้อมูล ก่อนที่จะนำข้อมูลนั้นเข้าไป Train ในตัวของโมเดล โดยมี Architecture ของตัวโมเดล ดังนี้
 ![lstm-1](https://user-images.githubusercontent.com/98243238/160967748-d7ab47eb-4c03-4199-872c-2ef79f7ab1fe.JPG)

โดยจากการทดลองจะเห็นได้ว่าตัวแบบสามารถทำนายได้ค่อนข้างไกล้เคียงกับราคาหุ้น ในช่วง period เริ่มต้น ก่อนที่จะมีการ deviate จากราคาที่แท้จริงค่อนข้างมากในช่วงเวลาต่อๆ มา โดยโมเดลนั้นมีค่า MAE, MSE เมื่อเทียบระหว่างราคาปิดของแต่ละวันอยู่ที่ 4.41 และ 22.04 ตามลำดับ
 ![lstm-2](https://user-images.githubusercontent.com/98243238/160967846-9d4a6619-f689-448b-a365-0fd6a58e6261.JPG)


## GRU
GRU write here ---> 
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
<br>LSTM 
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
