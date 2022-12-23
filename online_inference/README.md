# Сборка и запуск локально

docker build -t online-inference .

docker run -d --rm --name oi -p 80:80 online-inference

# Скачивание из docker hub и запуск локально

docker pull ikekz/made:online-inference

docker run -d --rm --name oi -p 80:80 ikekz/made:online-inference

# Тестирование

pytest ./online-inference/test_app.py

# Проверка запроса к сервису

python3 ./online_inference/make_predict_request.py ./online_inference/data/heart_cleveland_upload.csv