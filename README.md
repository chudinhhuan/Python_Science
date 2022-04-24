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
