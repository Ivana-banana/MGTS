# Приём стандартных ROS 1 bag

На синтетическом офисе (seed 41) проверены штатные команды `inspect`, `sync --require-complete` и `solve` для двух ROS 1 файлов: обхода и отдельной инспекции. Для них взяты те же сообщения, что и в контрольных ROS 2 SQLite bags. Тест использует `rosbags.convert` с типами ROS 1 Noetic и проверяет сигнатуру ROSBAG V2.0. Для запуска достаточно зависимостей проекта и pytest.

| Маршрут | PointCloud2 | Imu | Odometry | Метки сканов, покрытые одометрией | Межскановые интервалы, покрытые IMU |
|---|---:|---:|---:|---:|---:|
| Обход | 200 | 1001 | 401 | 200/200 | 199/199 |
| Инспекция | 100 | 501 | 201 | 100/100 | 99/99 |

Отчёты инвентаризации с декодированием и проверки временного покрытия совпали для обоих форматов. После двух запусков `solve` с автоматическим стартом побайтово совпали семь файлов: PGM и YAML карты, траектории обхода, инспекции и кинематической одометрии, непрерывная траектория и итоговый JSON инспекции. Проверка охватывает стандартные типы сообщений на демонстрационных данных. Для пользовательских типов и фактических калибровок нужна отдельная проверка на исходной записи.

## Повторить в Windows PowerShell

Из корня MGTS выполните команду. Она генерирует данные во временном каталоге, запускает оба конвейера и выводит SHA-256 сравниваемых файлов.

```powershell
& ..\Transport_MSK\.venv\Scripts\python.exe -m pytest -q -s tests/test_ros1_pipeline.py
```

При работе с собственными файлами ROS 1 передайте пути к `.bag` напрямую. Подставьте профиль установки и новое имя выходного каталога.

```powershell
$python = '..\Transport_MSK\.venv\Scripts\python.exe'
& $python -m mgts_nav inspect data/mapping.bag
& $python -m mgts_nav.sync data/inspection.bag --config data/inspection.json --require-complete
& $python -m mgts_nav.solve data/mapping.bag data/inspection.bag --config data/mapping.json --inspection-config data/inspection.json --out out/ros1-run
```
