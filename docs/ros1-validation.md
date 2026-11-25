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
& .\.venv-mgts\Scripts\python.exe -m pytest -q -s tests/test_ros1_pipeline.py
```

При работе с собственными файлами ROS 1 передайте пути к `.bag` напрямую. Подставьте профиль установки и новое имя выходного каталога.

```powershell
$python = '.\.venv-mgts\Scripts\python.exe'
& $python -m mgts_nav inspect data/mapping.bag
& $python -m mgts_nav.sync data/inspection.bag --config data/inspection.json --require-complete
& $python -m mgts_nav.solve data/mapping.bag data/inspection.bag --config data/mapping.json --inspection-config data/inspection.json --out out/ros1-run
```

## Плоский LaserScan в ROS 1

Для стандартного ROS 1 LaserScan сначала преобразуйте каждый исходный bag в отдельную запись ROS 2. Затем замените выбранный топик на PointCloud2 с помощью адаптера. Во всех трёх шагах используйте новые выходные каталоги. Пример предполагает, что топик в обеих записях называется `/scan` и указан в профилях как `lidar`.

```powershell
$python = '.\.venv-mgts\Scripts\python.exe'
foreach ($role in 'mapping','inspection') {
    & $python -X utf8 -m rosbags.convert --src "data/$role.bag" --dst "out/$role-scan" --dst-typestore ros2_humble
    if ($LASTEXITCODE -ne 0) { throw "Ошибка конвертации ROS 1: $role" }
    & $python -X utf8 -m tools.laser_scan "out/$role-scan" /scan "out/$role-cloud"
    if ($LASTEXITCODE -ne 0) { throw "Ошибка преобразования LaserScan: $role" }
}
& $python -X utf8 -m mgts_nav.solve out/mapping-cloud out/inspection-cloud --config data/mapping.json --inspection-config data/inspection.json --out out/ros1-laser-solve
```

Контроль использует синтетический офис с seed 41. Дальности записаны в 240 угловых ячеек; пропущенный луч имеет значение `inf`. Между соседними лучами задан интервал 0,2 мс. Разница времени записи и заголовка составляет 2 мс. До адаптации у контрольной ROS 2 и восстановленной ROS 2 совпали байты каждого сообщения LaserScan, IMU и одометрии. У всех трёх вариантов совпали времена записи. Для каждого ROS 1 LaserScan отдельно сверены время заголовка, кадр, дальности и шаг времени лучей. Исходные ROS 1 файлы сохранились побайтово.

Оба полных запуска с автоматическим стартом дали 199 привязок после первого скана карты и 100/100 привязок инспекции. Побайтово совпали семь продуктов: PGM, YAML и траектория карты, траектория и сводка инспекции, непрерывная траектория и кинематический baseline. SHA-256 траектории инспекции: `70721e578a56f5234ea0ad1ddca83b35e99658efd3822c5099d31cd88c5ba455`. Проверка охватывает стандартные типы и синтетические измерения. Для повторения обоих запусков и вывода всех семи хешей:

```powershell
& .\.venv-mgts\Scripts\python.exe -X utf8 -m pytest -q -s tests/test_laser_scan.py
```
