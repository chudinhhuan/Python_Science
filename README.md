# Python Science 
# I.Tìm hiểu về thư viện pandas 
![pandas](https://user-images.githubusercontent.com/90398366/164951745-80fd0300-5c22-4b67-bcb1-68a7dd9ad8fd.png)

## 1.Cách import thư viện 
``` 
import pandas as pd
import numpy as np
```
## 2.Các câu lệnh và thao tác với pandas
### Đọc file dataset 
``` df = pd.read_csv("chipotle.tsv.txt",sep="\t") ```
- đọc các hàng đầu 
``` df.head()
```
```
df.shape
#view row and column

#view type data
df.info()
```
- Xem tên các cột của dataset
``` df.columns```
-chuyển về dạng list
``` list(df.columns) ```
-xem index - theo hàng trong dataset -> kq:  RangeIndex(start=0, stop=4622, step=1)
``` df.index ```
- describe() : tính toán được một số thứ như mean,min, max ....
- describe(include="all") : tương tự như describe nhưng là tất cả
- loc['điều kiện '] : giúp lấy ra giá trị phù hợp với điều kiện ta cũng có thể kết hợp nhiều điều kiện để lấy ra thứ mong muốn ( | & ....)
- iloc[[index]] : select want row hoặc iloc[start : end]
- sum() 
- sort_values() : sort từ nhỏ -> lớn
- sort_values(ascending = False) -> lớn -> bé
- value_counts() : đếm tổng các giá trị
- nunique() : item duy nhất tức là trong dataset chỉ có một cái đó thôi

## end pandas - thiếu bổ sung sau nhé !!

# II.Tìm hiểu về thư viện numpy
![numpy](https://user-images.githubusercontent.com/90398366/164952064-326477c9-ba08-44a8-af59-c4b7be43deeb.png)

## Import thư viện
```
import numpy as np
```
- tạo mảng :
```
np.array([....,...])
```
- kiểm tra kiểu dữ liệu 
```
type()
```
- Convert type
```
array([int, int,],dtype='float32')
```
- Tạo mảng hai chiều - ma trận 
```
array([[...,....,....],[...,...,...]])
```
- xem dạng ,chiều , size
```
shape,
ndim,
size
```
- Creating Numpy Array from Scratch  ( #zeros , ones ,full ,arange , linspace)
- Random
>vidu
```
np.random.random((4,4))
```
###  random.seed(0) -> giúp random ra giá trị cố định không bị thay đổi
## random.normal
```
normal('mean','do lech chuan','size')
```
- random.randint()
> Vidu 
``` np.random.randint(0,10,[3,5]) ```
- random.rand()
> vidu 
``` np.random.rand(3,4)```
-reshape() 
-T : row <-> column
- concatenate() : chuyển matrix 2 chiều về 1 chiều
-  vstack -> noi theo chieu doc
-  hstack -> noi theo chieu ngang 
-  split(x,[start : end ])
## Statistics - xac suat thong ke
- std : độ lệch chuẩn 
``` np.std(dog_height) ```
- var : phương sai 
- sqrt() : độ lệch chuẩn lấy căn 2
- argsort() : return index cua cac element khi sort 
- A.dot(B) hoặc A @ B  : nhân
- note :  B(3x2) A(3x3) thì không được cần .T để thành B(2x3)@A(3x3)
## Cách tự tạo ra một dataFame 
![image](https://user-images.githubusercontent.com/90398366/164952528-ee5528df-ed35-4cb3-b46e-55b3424fb6f8.png)
```
#creat dataFrame
import pandas as pd
weekly_sales = pd.DataFrame(sales_amouts,index = ['Mon','Tues','Wed','Thurs','Fril'],
                            columns=['Almond Butter','Peanut Butter','Cashew Butter']) 
```
## Example 
``` 
np.random.seed(0)
sales_amouts = np.random.randint(20,size=(5,3))

#creat dataFrame
import pandas as pd
weekly_sales = pd.DataFrame(sales_amouts,index = ['Mon','Tues','Wed','Thurs','Fril'],
                            columns=['Almond Butter','Peanut Butter','Cashew Butter'])
weekly_sales
        
#create a price array
prices = np.array([10,8,12])

butter_prices = pd.DataFrame(prices.reshape(1,3),index=['Price'],
                             columns=['Almond Butter','Peanut Butter','Cashew Butter'])
                             
weekly_sales.shape,butter_prices.T.shape

total_prices = weekly_sales.dot(butter_prices.T)

weekly_sales['Total Price'] = total_prices 

weekly_sales 

```
## end numpy - thiếu bổ sung sau !




