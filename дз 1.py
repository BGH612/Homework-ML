import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Мной был выбран датасет Used Car Price Prediction Dataset, url : https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset?resource=download, в данном датасете присутвуют пропущенные значения
#import matplotlib as plt 
# 1. считывание данных

df=pd.read_csv("C:/Users/yaneg/Downloads/used_cars.csv")
print(df.isnull().sum())
# мы знаем, что для типа топлива и факта аварии присутвуют пропущенные значения


#plt.show()
# Задания 2,4,5 и 7 в следующей части я буду одновременно производить чистку данных, анализировать а также убирать неныжные признаки и создавать новые
#для полноценной работы, необходимо присвоить признакам числовые значения


#необходимо привести такие приззнаки, как пробег и цену  к числовым

df1=pd.read_csv("C:/Users/yaneg/Downloads/used_cars.csv")
df1
df['milage']=df1['milage']
df['price']=df["price"].str.replace('$', '', regex=False)
df['price'] = df['price'].str.replace(',', '').astype(float)
df['milage']=df['milage'].str.replace('mi.', '', regex=False)
df['milage']=df['milage'].str.replace(',', '').astype(float)


# Задание 3 в качестве целевого признака, для предсказания будет использоваться price, поскольку довольно важно понимать, за какую цену можно продать автомобиль, зная его характеристики

sns.boxplot(x = 'milage', data = df)
plt.title('пробег автомобиля')
plt.show()
# основная часть в диапазоне от 25 до 100 тысяч пробега

df['milage'].describe()
df['price'].describe()

sns.boxplot(x = 'price', data = df)
plt.title('цена автомобиля')
plt.show()

sns.countplot(x='price', data=df)
plt.title('цена авто')
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(8, 4))
plt.scatter(df.index, df['price'], color='blue', label='цена авто')
plt.title('цена автомобиля')
plt.show()
#для цены автомобиля можно увидеть довольно экстремальные значения, обусловленные экзотическими моделями в выборке

df.drop('clean_title', axis=1, inplace=True)
print(df['brand'].describe())
# 57 уникальных значений
print(df['model'].describe())
#1898 уникальных значений
#признаки бренд и модель лучше не использовать, поскольку брендов и моделей довольно разное количество и сложно адекватно присвоить каждому числовое значение, поэтому я собираюсь поделить автомобильные марки на три категории: стандарт, премиум и экзотические
# print(df['brand'].unique()
result19=df.groupby('brand')['price'].mean().reset_index()
sns.barplot(data=result19, x='brand', y='price')
plt.title('Средняя цена по бренду ')
plt.xlabel('Категория бренда')
plt.ylabel('Средняя цена')
plt.show()
# отсюда мы можем увидеть наиболее дорогие бренды, чтобы поделить их по цене

def categorize_brand(car):
    if car in ['Bentley', 'Aston', 'Lamborghini','Maserati','Alfa','Ferrari','Bugatti','Rolls-Royce',
 'McLaren', 'Lotus','Maybach']:
        return '3'
    elif car in ['Lexus','INFINITI','Audi','Acura','BMW','Tesla','Land','Jaguar','Mercedes-Benz','Genesis','Lucid','Porsche','Volvo', 'Cadillac','Rivian','Karma']:
        return '2'
    else:
        return '1'
df['brand_cat'] = df['brand'].apply(categorize_brand)
result18=df.groupby('brand_cat')['price'].mean().reset_index()



sns.barplot(data=result18, x='brand_cat', y='price')
plt.title('Средняя цена по категории бренда ')
plt.xlabel('Категория бренда')
plt.ylabel('Средняя цена')
plt.show()


# в третюю категорию попали самые дорогие авто

sns.countplot(x='brand_cat', data=df)
plt.title('распределение по брендам')
plt.show()
# можно увидеть, что автомобилей из самой дорогой категории брендов наименьшее количество


#вместо того, чтобы использовать год автомобиля, лучше использовать его возраст.
df['model_year'].astype(int)
df['age']=2024-df['model_year']
sns.countplot(x='model_year', data=df)
plt.title('возраст автомобиля')
plt.show()
# в выобрке большее количество автомобилей 2020-х годов
sns.boxplot(x = 'model_year', data = df)
plt.show()
#основная часть авто с 2012 по 2020 года




# перейдем к категориям топлива, мы знаем что среди них присутвуют пропущенные значения
sns.countplot(x='fuel_type', data=df)
plt.title('категории топлива')
plt.xticks(rotation=45)
plt.show()
#наиболее часто-встречающаяся категория- бензин
#найдем пропущенные значения
df['fuel_type'].unique()
missing_fuet = df['fuel_type'].isnull().sum()
missing_fueltype_rows = df[df['fuel_type'].isnull()]
# отсюда мы видим, что тип топлива пропущен у автомобилей с электрическим двигателем, можно это проверить, посмотрев датафрейм, в котором в двигателе содержится слово electric
filtered_df = missing_fueltype_rows[missing_fueltype_rows['engine'].str.contains('electric','-')]
#print(missing_fueltype_rows)
# количество строк все еще 170, это означает что все пропущенные fueltype, это електродвигатели
df['fuel_type']=df['fuel_type'].fillna('electro')
sns.countplot(x='fuel_type', data=df)
plt.title('категории топлива1')
plt.xticks(rotation=45)
plt.show()
#для типа топлива будет логичным поделить на две категории: обычное топливо и электроэнергия
def categorize_fuel(fuel):
    if fuel in ['Hybrid','electro','Plug-In Hybrid','not supported']:
        return '1'
    else:
        return '0'
df['fuel_cat']=df['fuel_type'].apply(categorize_fuel)
sns.countplot(x='fuel_cat', data=df)
plt.title('категории топлива2')
plt.xticks(rotation=45)
plt.show()
# наибольшее количество авто- обычные двигатели


print(df['transmission'].value_counts())
#здесь можно увидеть 62 разных типа трансмиссии, на самом деле стоит их объединить, для большей эффективности модели, их можно разделить на автомат, робот, механика
filtered_df = df[df['transmission'].str.contains('auto|a/t', case=False, na=False)]
#print(filtered_df['transmission'].value_counts())
df['transmission'] = np.where(df['transmission'].str.contains('Shift', case=False), 'robot', df['transmission'])
df['transmission'] = np.where(df['transmission'].str.contains('auto|a/t|at|cvt', case=False), 'auto', df['transmission'])
df['transmission'] = np.where(df['transmission'].str.contains('manual|m/t|mt', case=False), 'manual', df['transmission'])
#все еще остаются значения которые не могут быть однозначно определены к какому- то типу, стоит записать их как other, 4-я категория
df['transmission'] = df['transmission'].apply(lambda x: x if x in ['manual', 'auto', 'robot'] else 'other')
# определить их можно по году выпуска автомобиля, более новые автомобили скорее оснащены роботом и автоматом 
result = df.groupby('transmission')['model_year'].mean().reset_index()

print(result)

#можно увидеть, что средний год выпуска для автомата и для работа, примерно одинаковые, для категории other средний год 2015, их можно скорее отнести к автомату, поскольку робот все же более новая технология

result5 = df.groupby('transmission')['price'].mean().reset_index()
print(result5)
sns.barplot(data=result5, x='transmission', y='price')
plt.title('Средняя цена транмиссии')
plt.xlabel('Категория')
plt.ylabel('Средняя цена')
plt.show()
# если посотреть по средней цене, то скорее стоит отнести к роботу
# теперь можно привести к числовому признаку, самая дорогая трансмиссия - робот, после чего идет автомат и потом уже механика
def categorize_trans(trans):
    if trans in ['robot']:
        return '3'
    elif trans in ['auto']:
        return '2'
    elif trans in ['manual']:
        return '1'
    else:
        return '3'
df['transmission']=df['transmission'].apply(categorize_trans)

plt.figure(figsize=(10, 6))


sns.countplot(x='transmission', data=df)
plt.title('распределение по трансмиссиям')
plt.xticks(rotation=45)
plt.show()
# наиболее популярная трансмиссия- автомат, робот и механика примерно на равне 



#print(df['ext_col'].value_counts())
#print(df['ext_col'].describe())
#можно увидеть чтое есть, как базовые цвета, так и различные их вариации, стоит объединить все цвета по базовому, например преобразовать конфетный красный в красный
df['ext_col'] = np.where(df['ext_col'].str.contains('red', case=False), 'red', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('black', case=False), 'black', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('green', case=False), 'green', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('white', case=False), 'white', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('brown', case=False), 'brown', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('orange', case=False), 'orange', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('blue', case=False), 'blue', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('gray|grey', case=False), 'gray', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('silver', case=False), 'blue', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('yellow', case=False), 'yellow', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('purple', case=False), 'purple', df['ext_col'])
df['ext_col'] = np.where(df['ext_col'].str.contains('biege|beige', case=False), 'biege', df['ext_col'])
df['ext_col'] = df['ext_col'].apply(lambda x: x if x in ['red', 'black', 'green','white','brown','blue','gray','yellow','purple','biege','orange'] else 'specific')
# цветов все равно большое количество, стоит поделить их на три категории, basic, colour, specific; цвет отличный от базового может указывать на лучшую комплектацию, поэтому им будут присвоены соответствующие значения 1,2 и 3

#print(df['ext_col'].value_counts())
def categorize_color(color):
    if color in ['black', 'white', 'grey','silver','biege']:
        return '1'
    elif color in ['red','green','brown','blue','yellow','purple','orange']:
        return '2'
    else:
        return '3'
df['ext_col_cat'] = df['ext_col'].apply(categorize_color)
result6 = df.groupby('ext_col_cat')['price'].mean().reset_index()
print(result6)

sns.barplot(data=result6, x='ext_col_cat', y='price')
plt.title('Средняя цена цвета')
plt.xlabel('Категория цвета')
plt.ylabel('Средняя цена')
plt.show()

# данное предположение действительно подтвердилось, средняя цена для третьей категории наибольшая
sns.countplot(x='ext_col_cat', data=df)
plt.title('распределение по цвету авто')
plt.xticks(rotation=45)
plt.show()


# тоже самое необходимо проделать для интерьера
df['int_col'] = np.where(df['int_col'].str.contains('red', case=False), 'red', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('black', case=False), 'black', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('green', case=False), 'green', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('white', case=False), 'white', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('brown', case=False), 'brown', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('orange', case=False), 'orange', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('blue', case=False), 'blue', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('gray|grey', case=False), 'gray', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('silver', case=False), 'blue', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('yellow', case=False), 'yellow', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('purple', case=False), 'purple', df['int_col'])
df['int_col'] = np.where(df['int_col'].str.contains('biege|beige', case=False), 'biege', df['int_col'])
df['int_col'] = df['int_col'].apply(lambda x: x if x in ['red', 'black', 'green','white','brown','blue','gray','yellow','purple','biege','orange'] else 'specific')
#print(df['int_col'].unique())
df['int_col_cat'] = df['int_col'].apply(categorize_color)


sns.countplot(x='int_col_cat', data=df)
plt.title('распределение по интерьеру авто')
plt.xticks(rotation=45)
plt.show()

# теперь необходимо присвоить значения для accident, когда аварий не было значения получат 0, если были -1 



df['accident'] = np.where(df['accident'].str.contains('None reported', case=False), '0', df['accident'])
df['accident'] = np.where(df['accident'].str.contains('At least 1 accident or damage reported', case=False), '1', df['accident'])
#print(df['accident'].value_counts())
sns.countplot(x='accident', data=df)
plt.title('распределение по количеству аварий')
plt.xticks(rotation=45)
plt.show()


# у некоторых автомобилей написаны литры, однако отсутсвует запись о мощности, важно привести все к общему- к мощности, я планирую вычислить среднюю производительность лошадиных сил для одного литра из выборки и уже исходя из этого искать мощность по литрам


def extract_HP(s):
    
    match = re.search(r'\b\d*\b[.][0][H][P]', s)
    if match:
        return match.group(0) 
    return 0
def extract_volume(s):
    
    match = re.search(r'\d*[.][\d][L]|\d*[.][\d]\s[L]', s)
    if match:
        return match.group(0)  
    return 0

df['hp'] = df['engine'].apply(extract_HP)
df['litres']= df['engine'].apply(extract_volume)


df['hp'] = df['hp'].str.replace('HP', '', regex=False)
df['hp'] = df['hp'].str.replace('.0', '', regex=False)


df['litres']= df['litres'].str.replace('L', '', regex=False)


#filtered_df = df2[df2['hp'].str.contains('turbo|tsi', case=False, regex=True)]
df['hp'].fillna(0, inplace=True)
df['litres'].fillna(0, inplace=True)
df['hp'].astype(int)
df['hp'] = pd.to_numeric(df['hp'], errors='coerce')
df['litres'] = pd.to_numeric(df['litres'], errors='coerce')
df['hp']=df['hp'].astype(float)


# Вычисление среднего значения для столбца 'HP'
#print(filtered_df['hp'].mean())
#print(mean_hp)

df2 = df[df['hp'] != 0]
df3 = df2[df['litres'] != 0]
df3['pow']=df3['hp']/df3['litres']
#print(df3)
#print(df3['pow'].mean())
#теперь мы знаем среднюю мощность с каждого литра , она составляет 92.4, посчитаем мощность для всех авто исходя из этого значения
df.loc[df['hp'] == 0, 'hp'] = df['litres'] * 92.4
#print(df['hp'].dtype)
#print(filtered_df1)
# автомобили у которых не найдены лошадиные силы, получат среднюю мощность в зависимости от категории бренда
zero_hp_rows = df.loc[df['hp'] == 0]

#print(zero_hp_rows)
#df2['hp'] = pd.to_numeric(df2['hp'], errors='coerce')
#print(df2['hp'].mean())
#print(df2)

df4=df[df['hp'] != 0]
#print(df4)
#найдем среднее значение мощности для каждой категории авто
cat1_hp = df4[df4['brand_cat'] == '1']
average_hp1 = cat1_hp['hp'].mean()
cat2_hp = df4[df4['brand_cat'] == '2']
average_hp2 = cat2_hp['hp'].mean()
cat3_hp = df4[df4['brand_cat'] == '3']
average_hp3 = cat3_hp['hp'].mean()
print(average_hp1)
print(average_hp2)
print(average_hp3)

#мы нашли среднюю мощность для каждой категории

result7 = df4.groupby('brand_cat')['hp'].mean().reset_index()

sns.barplot(data=result7, x='brand_cat', y='hp')
plt.title('Средняя мощность по категориям')
plt.xlabel('Категория')
plt.ylabel('Средняя мощность')
plt.show()
# мы видим, что авто из третьей категории наиболее мощные

df.loc[(df['hp'] == 0) & (df['brand_cat'] == '1'), 'hp'] = average_hp1
df.loc[(df['hp'] == 0) & (df['brand_cat'] == '2'), 'hp'] = average_hp2
df.loc[(df['hp'] == 0) & (df['brand_cat'] == '3'), 'hp'] = average_hp3


#print(df)

print(df['hp'].describe())
df['hp'].hist()
plt.title('мощность автомобиля')
plt.show()


# можно увидеть, что большинство значений располагаются в интервале от 250 до 400, одако сущетсвуют выбросы, которые присуще суперкарам, они начинаются после 650

# 3- целевой признак, который я буду в дальнейшем определять - price, соответсвенно задача предсказания-находить величину price
# создадим новый датафрейм, содержащий только те признаки, которые необходимы для анализа
df_w=df[['milage','transmission','brand_cat','age','fuel_cat','ext_col_cat','int_col_cat','hp','accident','price']]
print(df_w)



# 6- оценка зависимости между признаками, построим корреляционную матрицу
corr_w=df_w.corr()
print(corr_w)
# мы видим, что наиболее влиятельные признаки здесь- это пробег, категория бренда, возраст автомобиля и мощность двигателя. Сильная корреляция (>0.7) между признаками вовсе отсутсвует, это в первую очередь может быть связано с большим количеством выбросом в таких признаках, как стоимость авто, пробег и мощность двигателя

df_w.to_csv('car_price_w.csv', index= False, encoding='utf-8')
