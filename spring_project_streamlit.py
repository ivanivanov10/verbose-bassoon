import streamlit as stream
with stream.echo(code_location='below'):
    stream.title('New York City Taxi')
    '''## Мотивация '''

    'Идею для проекта я взяла с одного из соревнований с Kaggle: [New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data). Суть соревнования - обучить модель для предсказания длительности поездки. Выбор обусловлен тем, что в данных пристуствуют координаты начала и конца поездки, а значит датасет хорошо подходит под геоаналитику и работу с графами.'
    '''К заданию приложены очищенные и готовые к исследованию данные, при этом исходные данные взяты из [этого источника](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), но мы, в рамках данного курсового проекта и для чистоты эксперимента, возьмем данные с оригинального источника, и потом уже на их основе будем проводить собственный анализ.'''

    import pandas as pd
    import numpy as np
    from scipy import stats as st
    import datetime as dt
    from pandarallel import pandarallel
    from tqdm import tqdm
    from bs4 import BeautifulSoup
    from io import BytesIO
    import requests
    from itertools import repeat
    import copy
    import random
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns
    import geopandas as gpd
    import networkx as nx
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    def main():
        warnings.filterwarnings("ignore")  # отключим варнинги
        tqdm.pandas()  # подключим прогресс бар к пандас
        pandarallel.initialize(progress_bar=True)  # добавим возможность паралельных вычислений для пандас
        pd.set_option('display.max_colwidth', None)  # уберем ограничение на макисмально количество столбцов
        """ ## Анализ"""

        '''**Построение модели прогноза длительности поездки**'''

        'Попробуем выполнить задачу, поставленную в условиях Каггла: построить модель прогноза длительности поездки. Для этого обучим три регрессионые модели и сравним их качество.'

        ml_df = pd.read_csv('test_df.csv')
        ml_df['store_and_fwd_flag'] = ml_df['store_and_fwd_flag'].replace({0: False, 1: True})
        ml_df['pickup_datetime'] = pd.to_datetime(ml_df['pickup_datetime'])
        ml_df['dropoff_datetime'] = pd.to_datetime(ml_df['dropoff_datetime'])
        ml_df = ml_df.dropna(subset=['pickup_neighbourhood', 'dropoff_neighbourhood'])

        'Приведем **датафрейм**, с которым будем работать'

        stream.dataframe(ml_df)

        'Построим **матрицу корреляций**, чтобы узнать взаимозависимость признаков (к сожалению, придется подождать недолго, пока он все посчитает).'
        figura=plt.figure(figsize=(10, 5))
        sns.heatmap(ml_df[['vendor_id', 'pickup_datetime', 'dropoff_datetime',
                           'passenger_count', 'pickup_longitude', 'pickup_latitude',
                           'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
                           'trip_duration']].corr(), annot=True, cmap="YlGnBu", vmin=-1, vmax=1, center=0);
        stream.pyplot(figura)

        'Для избежания ошибки "заглядывания в будущее" отсортируем данные по времени и приведем их к формату, пригодному для машинного обучения (если интересен непосредственно код, Вы можете посмотреть его ниже, там видна конкретная технология).'
        ml_df = ml_df.sort_values('pickup_datetime')
        ml_df['pickup_datetime'] = ml_df['pickup_datetime'].map(dt.datetime.toordinal)

        'Далее я оставлю лишь словесное описание шагов, а сам код Вы можете посмотреть ниже. Что мы делаем:'

        '1) Удалим пропуски и оставим только нужные столбцы'
        ml_df = ml_df.dropna(subset=['pickup_neighbourhood', 'dropoff_neighbourhood'])
        ml_df = ml_df[
            ['vendor_id', 'pickup_datetime', 'passenger_count', 'pickup_neighbourhood', 'dropoff_neighbourhood',
             'trip_duration']]

        '2) Закодируем категориальные переменные'
        encoder = LabelEncoder()
        ml_df['pickup_neighbourhood'] = encoder.fit_transform(ml_df['pickup_neighbourhood'])
        ml_df['dropoff_neighbourhood'] = encoder.fit_transform(ml_df['dropoff_neighbourhood'])

        '3) Выделим признаки и целевую переменную, разобъем данные на обучающую и валидационную выборку'
        X = ml_df.drop('trip_duration', axis=1)
        y = ml_df['trip_duration']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

        '4) Отмасштабируем данные, для избежания перекоса в данных'
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_st = scaler.transform(X_train)
        X_test_st = scaler.transform(X_test)

        '5) Создадим вспомогательную функцию для обучения одной модели'
        def make_prediction(model, X_train, y_train, X_test, y_test):
            metrics = []
            metric_list = ['MAE', 'MSE', 'R2']
            m = model
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

            #Посчитаем метрики
            metrics = [mean_absolute_error(y_test, y_pred),
                       mean_squared_error(y_test, y_pred),
                       r2_score(y_test, y_pred)]
            return pd.Series(metrics, index=metric_list)

        '6) Создадим вспомогательную функцию для обучения всех моделей'
        def make_all_predictions(X_train, y_train, X_test, y_test):
            reg_models_list = [LinearRegression(), Lasso(), Ridge()]
            models_name = ['LinearRegression', 'Lasso', 'Ridge']
            metric_list = ['MAE', 'MSE', 'R2']

            #6.1 Итоговый датафрейм
            regression = pd.DataFrame(columns=metric_list)

            #6.2 Обучаем модели по циклу
            for model in reg_models_list:
                regression = regression.append(make_prediction(model, X_train, y_train, X_test, y_test),
                                               ignore_index=True)
            regression.index = models_name
            return regression.round(2)


        pred=make_all_predictions(X_train_st, y_train, X_test_st, y_test)

        'Вуаля, готово:'

        stream.dataframe(pred)

        'Как мы видим, регрессия дала плохие результаты, что скорее всего говорит о плохом выборе признаков. Плохой результат - тоже результат, возможно, какие-то из признаков не совсем осмысленные! Но технически все сделано верно.'

        '''## Подготовка данных'''

        'Вернемся к полноценному анализу данных, для этого перейдем обратно на полный датасет.'

        train_df = pd.read_csv('train.csv')
        
        stream.dataframe(train_df)
        
        train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].replace({0: False, 1: True})
        train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
        train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

        taxi_df = train_df

        #Достанем всю информацию из столбцов с датами
        taxi_df['pickup_datetime_hour'] = taxi_df['pickup_datetime'].dt.hour
        taxi_df['pickup_datetime_day'] = taxi_df['pickup_datetime'].dt.day
        taxi_df['pickup_datetime_weekday'] = taxi_df['pickup_datetime'].dt.weekday
        taxi_df['pickup_datetime_week'] = taxi_df['pickup_datetime'].dt.week
        taxi_df['pickup_datetime_month'] = taxi_df['pickup_datetime'].dt.month
        taxi_df['pickup_datetime_date'] = taxi_df['pickup_datetime'].dt.date

        taxi_df['dropoff_datetime_hour'] = taxi_df['dropoff_datetime'].dt.hour
        taxi_df['dropoff_datetime_day'] = taxi_df['dropoff_datetime'].dt.day
        taxi_df['dropoff_datetime_weekday'] = taxi_df['dropoff_datetime'].dt.weekday
        taxi_df['dropoff_datetime_week'] = taxi_df['dropoff_datetime'].dt.week
        taxi_df['dropoff_datetime_month'] = taxi_df['dropoff_datetime'].dt.month
        taxi_df['dropoff_datetime_date'] = taxi_df['dropoff_datetime'].dt.date

        'Найдем дистанцию, для этого посчитаем евклидово расстояние между координатами. Результат умножим на стандартный множитель для радиан 111.3 (считать будет долго, наберитесь терпения).'

        taxi_df['distance_calc'] = taxi_df.progress_apply(lambda row: 111.3 * np.linalg.norm(
            np.array([row['pickup_longitude'], row['pickup_latitude']]) -
            np.array([row['dropoff_longitude'], row['dropoff_latitude']])), axis=1)

        stream.dataframe(taxi_df['distance_calc'])

        'На всякий случай посчитаем длительность поездки вручную (как разность конца и начала) - в секундах и в часах.'
        taxi_df['trip_duration_calc'] = (taxi_df['dropoff_datetime'] - taxi_df['pickup_datetime']) / np.timedelta64(1,
                                                                                                                    's')
        taxi_df['trip_duration_calc_hour'] = (taxi_df['dropoff_datetime'] - taxi_df[
            'pickup_datetime']) / np.timedelta64(1,
                                                 's') / 60 / 60

        'Проверила, везде ли совпадает данная и посчитанная длительность.'
        stream.dataframe(taxi_df[taxi_df['trip_duration'] != taxi_df['trip_duration_calc']])

        'Как видим, у нас получится пустой датафрейм, значит все совпало. Совпало везде - нам повезло.'
        
        'Посчитаем среднюю скорость для каждой поездки.'
        taxi_df['speed'] = taxi_df['distance_calc'] / taxi_df['trip_duration_calc_hour']

        stream.dataframe( taxi_df['speed'])

        'Посмотрим, сколько у нас всего вендоров и сколько у них было поездок за три месяца.'
        taxi_df['vendor_id'].value_counts()

        'Их всего два. Имеет смысл сравнить их с точки зрения статистических метрик. Займемся этим чуть ниже.'

        'Сколько обычно пассажиров ездит в такси?'
        taxi_df['passenger_count'].value_counts()

        stream.dataframe(taxi_df['passenger_count'].value_counts())

        '0 пассажиров - это скорее всего баг, а 7 пассажиров - это выброс. И то, и другое отбросим.'

        taxi_df = taxi_df.query('passenger_count != 0 and passenger_count != 7')
        
        'Приведу очищенные данные:'
        
        stream.dataframe(taxi_df['passenger_count'].value_counts())

        '''## Сравнение двух вендоров'''

        'Сравним метрики для обоих вендоров.'

        stream.dataframe(taxi_df.groupby('vendor_id').agg(trip_count=('passenger_count', 'count'),
                                         mean_client_amount=('passenger_count', 'mean'),
                                         mean_duration=('trip_duration_calc_hour', 'mean'),
                                         mean_distance=('distance_calc', 'mean'),
                                         # unique_pickup_district = ('pickup_neighbourhood', 'nunique'), unique_dropoff_district = ('dropoff_neighbourhood', 'nunique'),
                                         mean_speed=('speed', 'mean')))

        'Второму вендору явно везет больше: больше поездок, в среднем больше пассажиров, дольше и дальше поездки, даже скорости чуть выше.'

        'Проверим следующую гипотезу: "Среднее количество клиентов вендоров 1 и 2 разное". Пусть **гипотеза H0** звучит так: ***"Средние значения выборок равны"***, а **гипотеза H1**: ***"Средние значения выборок не равны"***.'

        'Проверим гипотезу о равенстве среднего количества клиентов в каждом датасете.'

        'В качестве выборок возьмем количество пользователей за каждую дату.'
        vendor_agg = taxi_df.groupby(['vendor_id', 'pickup_datetime_date']).agg(
            trip_count=('passenger_count', 'count')).reset_index()
        vendor_1 = vendor_agg[vendor_agg['vendor_id'] == 1].sort_values('pickup_datetime_date')['trip_count'].tolist()
        vendor_2 = vendor_agg[vendor_agg['vendor_id'] == 2].sort_values('pickup_datetime_date')['trip_count'].tolist()

        'Для сравнения выборок проведем Т-тестирование. Напомню, что 0.05 - критический уровень статистической значимости '
        alpha = 0.05
        results_1_2 = st.ttest_ind(vendor_1, vendor_2)
        stream.write('p-значение для сравнения групп: ', results_1_2.pvalue)
        if results_1_2.pvalue < alpha:
            stream.write('Отвергаем нулевую гипотезу для сравнения групп')
        else:
            stream.write('Не получилось отвергнуть нулевую гипотезу для сравнения групп')

        '**Промежуточный итог**: Среднее количество клиентов вендоров 1 и 2 действительно разное.'

        '## Графы'

        'Рассмотрим данные с точки зрения графов. В данном случае данные будут представлять собой набор ребер из двух точек. Если отображать сразу все, то на рисунке можно просмотреть некоторые тренды, но не конкретные наблюдения, если же отображать малое количество точек, то по рисункам можно будет делать какие-то выводы. Для поставленной задачи я создам универсальную функцию для построения графов на координатной плоскости.'


        def make_networks_graph(df, shape):
            point_1 = list(df.head(shape)[['pickup_longitude', 'pickup_latitude']].itertuples(index=False, name=None))
            point_2 = list(df.head(shape)[['dropoff_longitude', 'dropoff_latitude']].itertuples(index=False, name=None))
            point_1.extend(point_2)
            points = point_1
            edges = [(i, i + shape) for i in range(shape - 1)]
            G = nx.Graph()
            for i in range(len(edges)):
                G.add_edge(points[edges[i][0]], points[edges[i][1]])
            pos = {point: point for point in points}
            fig, ax = plt.subplots(figsize=(15, 10))
            nx.draw(G, pos=pos, node_color='b', ax=ax)
            nx.draw(G, pos=pos, node_size=1, ax=ax)
            plt.axis("on")
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            stream.pyplot(fig)

        'Попробуем использовать эту функцию с конкретными количествами. Ниже граф для 1000 поездок: что интересно, явно видны скопления конца поездки, что может быть связано с каким-то популярным местом, которое многие люди посещают. По данному графу, где много точек, можно попробовать сделать гипотезы относительно популярных мест отправки и высадки пассажиров.'

        make_networks_graph(train_df, 1000)

        'Для примера нарисуем такой граф для 50 поездок. По нему тоже можно сделать некоторые наблюдения.'

        make_networks_graph(train_df, 50)

        '**Промежуточный итог**: Помимо выводов, которые описаны выше, данную функцию можно использовать для дальнейшего анализа.'

        '## Исследовательский анализ данных'

        'Исследуем данные подробнее.'

        'Построим взаимные распределения параметров.'

        my_data=taxi_df.head(1000)
        fig = sns.pairplot(my_data,
                     x_vars=['pickup_datetime', 'dropoff_datetime', 'trip_duration', 'distance_calc', 'speed'],
                     y_vars=['pickup_datetime', 'dropoff_datetime', 'trip_duration', 'distance_calc', 'speed'],
                     hue='vendor_id'
                     );

        stream.pyplot(fig)

        'Получаем следующие результаты:'
        
        '**1)** Зависимость времени начала и конца поездки строго линейна. Скорее всего это вызывано тем, что график обрезает данные вплоть до даты'
        '**2)** Зависимость дистанции от времени почти линейна'
        '**3)** Гистограммы подтверждают выводы, сделаннные выше по сгруппированным данным.'

        'Изучим зависимость количества пользователей от различных кусков времени.'
        #Для этого напишем вспомогательную функцию.

        time_dict = {'pickup_datetime_hour': "Час",
                     'pickup_datetime_weekday': "День недели",
                     'pickup_datetime_date': "Дата", }

        def print_time_graph(data, delta):
            vendor_agg = data.groupby(['vendor_id', delta]).agg(trip_count=('passenger_count', 'count')).reset_index()
            plt.figure(figsize=(15, 5))
            sns.lineplot(data=vendor_agg, x=delta, y='trip_count', hue="vendor_id")
            plt.xticks(rotation=80, fontsize=11)
            plt.title('Зависимость количества клиентов от даты и времени', fontsize=13)
            plt.ylabel('Количество клиентов', fontsize=13)
            plt.xlabel(time_dict[delta], fontsize=13)
            plt.legend(title="Вендор");

        'Построим график зависмости количества пользователей от часа.'
        print_time_graph(taxi_df, "pickup_datetime_hour")

        'Видим, что конец и начало графика почти совпадают. Ночью никто не ездит, утром все едут на работу, а вечером после 17 - едут домой.'

        'Построим график зависмости количества пользователей от дня недели.'
        print_time_graph(taxi_df, "pickup_datetime_weekday")

        'Ожидаемо на выходных ездят меньше.'

        'Построим график зависимости количества пользователей от даты.'

        print_time_graph(taxi_df, "pickup_datetime_date")

        'Видим ярко выраженную цикличность с просадками на выходных.'
       

        '''## Geopandas'''

        'Изучим координаты начала и конца поездки.'


        fig, ax = plt.subplots(figsize=(15, 5))
        sns.scatterplot(data=taxi_df, ax=ax, x="pickup_longitude", y='pickup_latitude', )
        ax.set_xlim(-74.1, -73.7)
        ax.set_ylim(40.5, 41)

        stream.pyplot(fig)


        fig, ax = plt.subplots(figsize=(15, 5))
        sns.scatterplot(data=taxi_df, ax=ax, x="dropoff_longitude", y='dropoff_latitude', )
        ax.set_xlim(-74.1, -73.7)
        ax.set_ylim(40.5, 41)

        stream.pyplot(fig)

        'Видим, что координаты явным образмо складываются в очертания районов и островов Нью-Йорка. Проверим эту гипотезу, наложив точки на карты - сначала Америки, потом Нью-Йорка. Тут нам и пригодится geopandas.'


        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
        nyc = nyc.to_crs(world.crs)

        slice_df = taxi_df.query('-74.4<=pickup_longitude<=-73.7 and 40.5<=pickup_latitude<=41.0')
        pickup_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(slice_df['pickup_longitude'], slice_df['pickup_latitude']))

        'Ниже мы видим карту для Северной Америки, она скорее служит вспомогательным элементом.'


        fig, ax = plt.subplots(figsize=(15, 10))
        ax = world[world.continent == 'North America'].plot(color='white', edgecolor='black', ax=ax)
        pickup_gdf.plot(ax=ax, color='red')
        stream.pyplot(fig)


        fig, ax = plt.subplots(figsize=(15, 10))
        ax = nyc.plot(color='white', edgecolor='black', ax=ax)
        stream.pyplot(fig)


        'И вот, наконец, карта для Нью-Йорка. Мы можем наблюдать, что наибольшее скопление точек в районe **Манхеттена**, что и следовало ожидать.'


        fig, ax = plt.subplots(figsize=(15, 10))
        ax = nyc.plot(color='white', edgecolor='black', ax=ax)
        pickup_gdf.plot(ax=ax, color='red')
        stream.pyplot(fig)
  
        '''В первой части (ipynb-файл) на этапе сбора информации были покрыты критерии:'''
        '**1.**Pandas (на продвинутом уровне)'        
        '**2.**Web-scrapping'        
        '**3.**SQL (создаю свою базу данных и с ней работаю)'        
        '**4.**Доп технологии (треды - не обсуждались в рамках курса)'        
        '**5.**Регулярные выражения'        
        '**6.** Работа с API (кажется, что код сложнее, чем в домашних заданиях)'

        '''В данной части проекта были покрыты критерии:'''
        '**1.** РPandas (продвинутые функции по типу groupby)'
        '**2.** Визуализация данных (в том числе представлена визуализация с наложением точек на карту)'
        '**3.** Математические способности Python (stats используется)'
        '**4.** Streamlit'
        '**5.** Машинное обучение (предсказательная модель)'
        '**6.** Графы'
        '**7.** Объем'
        
        'Таким образом по технологиям и объему покрыты все критерии.'

        'Целостность и общее впечатление - вещи субъективные:)'
    if __name__ == "__main__":
        main()
