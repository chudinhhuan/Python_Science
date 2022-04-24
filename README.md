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
- Lấy dataset
[dataset](https://github.com/chudinhhuan/Python_Science/blob/main/chipotle.tsv.txt)
- Cách đọc dataset : 
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

# Xem thêm Pandas tại đây : 
[Xem thêm tại đây ](https://github.com/chudinhhuan/Python_Science/blob/main/lesson1.ipynb)


## end pandas - thiếu bổ sung sau nhé !!
![image](https://user-images.githubusercontent.com/90398366/164959859-3e9b9845-b3e1-4e09-98d7-1ea8a7d56c7b.png)

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
# Xem thêm tại đây để hiểu rõ hơn 
[xem thêm tại đây](https://github.com/chudinhhuan/Python_Science/blob/main/lesson2-numpy.ipynb)
![image](https://user-images.githubusercontent.com/90398366/164959939-77f17ead-c6da-4e82-a860-e84340cb33b3.png)

# end numpy - thiếu bổ sung sau !

# III . Tìm hiểu về thư viện Matplotlip 
![image](https://user-images.githubusercontent.com/90398366/164960474-2cd6f99a-4454-4780-908f-f69b4e68d7e2.png)

### import library 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
## 1.introduction matplotlip 
### Hiển thị backgroud đơn giản 
``` 
plt.style.available
plt.style.use('seaborn-whitegrid')
# backgroud for graph
plt.plot(); #display
```
### ví dụ vẽ đồ thị đơn giản
```
x = [1,2,3,4]
y=[10,8,6,9]
plt.plot(x,y,color='red');
```
### Pylot API vs Object-Oriented(OO) API
- #pylot API - Quickly #OO API -> Advanced
## Most common types of Matplotlib plost
### line  scatter  bar  hist   subplots()
- line : đồ thị đạng đường
```
# OO API plot of line chart
fig,ax = plt.subplots()
ax.plot(x,x**3);
```
-  Scatter: đồ thị dạng nét đứt , chấm chấm 
```
#Scatter
plt.scatter(x,np.exp(x)) 
# y=e^x
```
> Vidu2 : Scatter 
```
#Prepare new data
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
sizes = 100*rng.rand(100)
color = rng.rand(100)
fig,ax = plt.subplots()
img1 = ax.scatter(x,y,s=sizes,c=color,cmap='viridis')#alpha=0.3;
fig.colorbar(img1)
```
- Bar  : dạng đồ thị cột vertical và horizontal 

# Example
> Lấy dữ liệu 
[dataset](https://github.com/chudinhhuan/Python_Science/blob/main/california_cities.csv)
>Click vào raw sau đó mở ra và bấm vào lưu dataset -> tải thành công
- Đọc dataset 
```
cities = pd.read_csv('california_cities.csv')
```
- Tương quan dữ liệu được thể hiện như sau 
```
# extract latd , longd
# lay thong tin toa do voi latd vi do, longd kinh do
lat,lon = cities['latd'],cities['longd']
# lay them thong tin nhu dan so va tong dien tich
polulation,area = cities['population_total'],cities['area_total_km2']

# plt.figure(figsize(8,10)) #size to graph

plt.style.use('seaborn') #set style

# plot using pylot API
# sactter use drow dot graph
plt.scatter(lon,lat,c=np.log10(polulation),
            cmap='viridis',s=area,
           linewidths=0,alpha=0.5);
# trang tri
plt.axis('equal');
plt.xlabel('longtitude')
plt.ylabel('latitude');
# thanh tab3
plt.colorbar(label='log$_{10}&(population)');
plt.clim(3,7);


#creat a lengend for cites size
area_range = [50,100,300,500]
for area in area_range:
    plt.scatter([],[],s=area,c='k',alpha=0.4,
                
                label=str(area) + 'km$^2$')
plt.legend(labelspacing = 1,title='City Area') #scatterpoints=1


plt.title('California Cites: Population and Area Distribution');
```
# Hiểu rõ hơn bấm vào đây 
[hiểu rõ hơn](https://github.com/chudinhhuan/Python_Science/blob/main/lesson3_matplotlib.ipynb)

![image](https://user-images.githubusercontent.com/90398366/164960483-97eb77d4-0468-40c0-a052-f7d4d74c4599.png)


# End matplotlib

# IV. Tìm hiều về thư viện Seaborn 
![image](https://user-images.githubusercontent.com/90398366/164960608-b472385b-7846-4633-b389-a30c525a908b.png)
> Seaborn có sẵn một số dataset để học tập
## Một số kỹ thuật trong seaborn
- Distribution: Histogram(biểu đồ phân bố tần suất),KDE - phân bổ ước lượng ,displot\
- Bar Plot : biểu đồ cột
- Count Plot : biểu đồ thể hiện số lượng
- Facel Plot : đồ thị dạng 2D
- Box plot : phương pháp mô tả bằng đồ thị từ nhóm dữ liệu thông qua các phần tử của mảng
- Join Plot : biểu đồ hình dung phân phối chuẩn
- Pair Plot
- Heat Map : kỹ thuật trực quan hóa dữ liệu cho thấy cường độ là màu sắc hai chiều

## Example 
```
# draw headmap
# vmin, vmax -> điều chỉnh thanh bar
# linecolor="color" -> đường ngăn cách
# diverging_palette(value_color,as_cmap=True) -> thang màu


cmap = sns.diverging_palette(0,230,90,40, as_cmap=True)
fig,ax = plt.subplots(figsize=(10,8))
# fig,ax = plt.subplots(điều chỉnh size)
sns.heatmap(data=adjusted_mask_corr,mask=adjusted_mask,
           annot=True,fmt=".2f",cmap=cmap,
            linecolor="white",linewidths=0.5
           );
# sns.heatmap(data = data_input ,mask = mask_input,annot=True(in giá trị từng ô),fmt=".2f"(làm tròn ),cmap="color")
# lam cho x va y label viet hoa
yticks = [i.upper() for i in adjusted_mask_corr.index]
xticks = [i.upper() for i in adjusted_mask_corr.columns]

# điều chỉnh góc hiển thị chữ 
ax.set_yticklabels(yticks,rotation=0)
ax.set_xticklabels(xticks,rotation=90);

title = 'Correlation matrix\nSampled cereals composition\n' #su tuong quan giua cac chat dinh duong trong ngu coc
ax.set_title(title,loc='left',fontsize=28);

```
## Xem nhiều hơn để hiểu rõ hơn
[Seaborn](https://github.com/chudinhhuan/Python_Science/blob/main/lesson4_Seaborn.ipynb)
