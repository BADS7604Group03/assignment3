# assignment3
Sequence Data with RNN Simple/RNN Encoder-Decoder/LSTM/GRU/CNN combined with LSTM


# Highlight


# Introduction


# Data


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

 <p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/87576892/160887534-8fe6fa7a-e3c4-483d-810b-87324c1ecdc5.png">
</p>

 <p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/87576892/160887292-89e4dafe-4f7f-43c7-bc5f-da1b18a782c5.png">
</p>


# Network Architecture and Result
## Traditional ML - LightGBM 
 ทางทีมได้เลือก LightGBM เนื่องจากเป็นโมเดลที่มีความเร็ว และมีประสิทธิ์ภาพที่สูง จากการใช้ Gradient Boosting หรือ เทคนิคการเรียนรู้ที่จะสร้างโมเดลที่มีความแม่นยำสูง โดยเรียนรู้จากค่าความคลาดเครลื่อนสะสมที่เกิดจากการทำนายของโมเดลที่สร้างก่อนหน้า โดยทางทีมได้กำหนด parameter ต่างๆ ไว้ดังนี้ { 'boosting_type': 'gbdt',  'objective': 'regression', 'metric':'l2', 'num_leaves':10, 'max_depth':5, 'drop_rate ':0.3,  'reg_sqrt':True,  'boost_from_average':True,  'learning_rate': 0.0001, 'verbose': 0, } และตั้งค่าในการ train ไว้เป็น num_boost_round=1000,  early_stopping_rounds=100, verbose_eval=50 เมื่อนำโมเดลที่ Train เสร็จเรียบร้อยแล้วมาทดสอบกับข้อมูล test set (ข้อมูลแยกไว้เพื่อทดสอบโมเดลในตอนสุดท้าย) ได้ผล Mean Squared Error (MSE) : 0.207 และ Mean Absolute Error : 0.335 
![image](https://user-images.githubusercontent.com/87576892/160876781-285c2681-db7c-476d-87b1-f771f6b692e8.png)

## RNN Simple
RNN write here ---> 
## LSTM
LSTM write here ---> 
## GRU
GRU write here ---> 
## RNN Encoder-Decoder
RNN Encoder-Decoder write here ---> 
## CNN combined with LSTM
CNN combined with LSTM write here ---> 

# Referance

# Member
1) ณัฐภณ อัศวเหม 6310422052 (% contribution in this homework: 16.67%)
<br>Train 
2) ดวงธิดา แซ่แต้ 6310422056 (% contribution in this homework: 16.67%)
<br>Train 
3) เมธี ประเสริฐกิจพันธุ์ 6310422053 (% contribution in this homework: 16.67%)
<br>Train 
4) พีรพัทธ ตั้งไพบูลย 6310422024 (% contribution in this homework: 16.67%)
<br>Train 
5) วิชิต ชำนาญนาวา 6310422055 (% contribution in this homework: 16.67%)
<br>Train 
6) ไตรทิพย์ ศุภศิริวัฒนา 6310422009 (% contribution in this homework: 16.67%)
<br>Train  
