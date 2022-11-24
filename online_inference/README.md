# Сборка и запуск локально

docker build -t online-inference .
docker run -d --rm --name oi -p 80:80 online-inference

# Скачивание из docker hub и запуск локально

docker pull ikekz/made:online-inference
docker run -d --rm --name oi -p 80:80 ikekz/made:online-inference