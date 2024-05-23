# Предиктивная аналитика больших данных

Учебный проект для демонстрации основных этапов жизненного цикла проекта предиктивной аналитики.  

## Installation 

Клонируйте репозиторий, создайте виртуальное окружение, активируйте и установите зависимости:  

```sh
git clone https://github.com/Shumal07/pabd24
cd pabd24
python -m venv venv

source venv/bin/activate  # mac or linux
.\venv\Scripts\activate   # windows

pip install -r requirements.txt
```

## Usage

### 1. Сбор данных о ценах на недвижимость (создаются 3 файла в data/raw по 1,2,3-комнатных квартирах) 
```sh 
python parse_cian.py
```

### 2. Выгрузка данных в хранилище S3 
Для доступа к хранилищу скопируйте файл `.env` в корень проекта и пропишите KEY и SECRET 

```sh 
python upload_to_s3.py
```

### 3. Загрузка данных из S3 на локальную машину  

```sh 
python download_from_s3.py
```

### 4. Предварительная обработка данных 
Необходимо сначала создать папку 'log' в корне проекта, а также добавить папку 'proc' в 'data'

```sh 
python preprocess_data.py
```

### 5. Обучение модели 
Обучает модель, далее предсказывает на тестовой выборке (запускает test_model.py). Модель обучается на м^2 и цене. 
```sh 
python train_model.py
```

### 6. Запуск приложения flask 

todo

### 7. Запуск приложения gunicorn

todo

### 8. Использование сервиса через веб интерфейс

Для доступа к сервису используйте следующий адрес: http://192.144.14.183:8000/

Для использования сервиса используйте файл web/index.html.
