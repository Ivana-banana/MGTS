# Время точек и компенсация движения Velodyne на HEAP-1

В исходном Zarr время каждого принятого луча доступно. Архив VLP-16 содержит массивы `points` (XYZ float32), `time` (float64, секунды относительно заголовка), `intensity`, `ring`, `sequence_id`, `timestamp` и `valid`. В `time[3060,35000]` учитываются только первые `valid[i]` значений строки. Адаптер №80 записал в ROS 2 только XYZ и метку скана. Для всех 300 кадров (5 412 406 точек) проверены исходные байты XYZ, число точек и обе метки времени против ROS 2 bag. SHA-256 исходного TAR проверяется при запуске. [Закреплённая публикация HEAP-1](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/tree/af6b6dd6f91696f8ec8c98134e195514834911b2/2024-11-14-13-45-37/data)

**Привязка времени.** Для VLP-16 декодер складывает разность между меткой пакета и опорной меткой с табличной задержкой срабатывания лазера. Для исходного облака опорой служит заголовок скана. Для опубликованного `undistorted` опорой служит первый пакет, а геометрия пересчитывается по TF на каждый пакет. Атрибуты архива указывают на одометрию по ногам и IMU. Для этой миссии каталог издателя также перечисляет исходный файл `2024-11-14-13-45-37_anymal_velodyne_packets.bag` размером 560 557 769 байт. Эксперимент использовал декодированные времена Zarr. [Декодер VLP-16](https://github.com/leggedrobotics/anybotics_velodyne/blob/50448eebc39d5a50b9befc61838746d9138b69e5/velodyne_pointcloud/src/lib/rawdata.cc) · [Привязка обоих облаков](https://github.com/leggedrobotics/anybotics_velodyne/blob/50448eebc39d5a50b9befc61838746d9138b69e5/velodyne_pointcloud/src/conversions/transform.cc#L237-L285) · [Каталог миссий](https://grandtour.leggedrobotics.com/data/missions.json)

У всех 300 исходных оборотов первая валидная точка имеет смещение от −99,547 до −98,146 мс, последняя по времени — от +1,304 до +1,306 мс. Заголовок лежит близко к концу оборота. После сортировки по опубликованному времени развёрнутый полярный угол atan2(y,x) уменьшается на 356,24–364,27° за оборот (медиана 359,33°), то есть вращение по часовой стрелке при взгляде со стороны +Z лидара. В двух TAR соседние номера строк относятся к разным началам оборота: всем raw[150:450] соответствуют undistorted[149:449] по sequence_id. Заголовки raw позже на 98,146–99,547 мс, а вычисленные абсолютные времена первых точек совпадают в пределах 0,194 мкс. SHA-256 проверенного undistorted TAR — `fdc8336d2cf4c41345a15fd296fff3a7624b49c48682c0b98a2bd2955e7cd665`.

**Эксперимент.** Для каждого возврата вычисляется поза на `timestamp + time`. Угловая скорость берётся из синхронного IMU, переводится в оси корпуса штатной калибровкой, затем интегрируется её компонента Z. Исправляется только поворот в плоскости с учётом положения лидара относительно корпуса. Оценка переноса, крена и тангажа в эксперимент не входит. На кадрах 150 и 449 обрезанный IMU не покрывает весь оборот, поэтому они переданы без изменений. Все метки и сырые сообщения IMU сохранены. Два запуска конвертера дали 12 262 попарно идентичных сообщения.

На кадрах 220–230 проверены 192 752 точки. Медиана длины внесённой поправки — 1,19 см, 95-й перцентиль — 16,89 см, максимум — 1,42 м. Это изменение координат отдельных отражений. Ошибка относительного перемещения робота на том же интервале оценена отдельно по CPT7 после построения карт.

## Воспроизведение

Из корня MGTS в Windows PowerShell, после подготовки `out/heap1-long-80` по [описанию 300-кадрового обхода](heap1-long-mapping.md).

```powershell
$python = '.\.venv-mgts\Scripts\python.exe'
$run = 'out/heap1-timing-90'
New-Item -ItemType Directory -Force $run | Out-Null
$url = 'https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/resolve/af6b6dd6f91696f8ec8c98134e195514834911b2/2024-11-14-13-45-37/data/velodyne_points.tar'
curl.exe -fLsS $url -o "$run/velodyne_points.tar"
& $python -m tools.heap1_point_times "$run/velodyne_points.tar" out/heap1-long-80 "$run/gyro"
& $python -m mgts_nav map "$run/gyro" --config out/heap1-long-80/profile.json --resolution .08 --out "$run/map"
& $python -m mgts_nav map out/heap1-long-80/mapping --config out/heap1-long-80/profile.json --resolution .08 --out "$run/control"
```

## Ошибки траектории

| Обход, 300 кадров | Привязки после старта | RMSE XY | RMSE курса | Ошибка XY в конце | Ошибка относительного переноса 220–230 |
|---|---:|---:|---:|---:|---:|
| Исходные XYZ | 299/299 | 39,23 см | 0,620° | 62,70 см | 10,53 см |
| Компенсация поворота по IMU | 299/299 | 18,47 см | 0,112° | 24,92 см | 7,66 см |

Траектория контрольного прогона побайтово совпала с результатом №80. Поправки часов не вводились. Эффект измерен для одного 29,90-секундного окна и одной калибровки. Файл пакетов указан в [карточке Kleinkram](https://datasets.leggedrobotics.com/project/921edfe8-6efa-4328-b55b-91aed48f7b9f/mission/512edee1-de86-463f-bdda-f55362e2fda1/file/baa83377-c8ea-439c-918c-37a442c9c3f8). Анонимный запрос API к карточке файла вернул HTTP 401, его содержимое и значение параметра драйвера timestamp_first_packet на роботе не проверены.

Расчёт метрик после завершения обоих запусков:

```powershell
@'
from pathlib import Path
import numpy as np
from mgts_nav.metrics import read_csv
from tools.heap1_drift import rotate
from tools.heap1_long_mapping import load_result, score
truth = np.asarray(read_csv("out/heap1-long-80/mapping-reference.csv"))
for name in ("control", "map"):
    rows, statuses = load_result(Path("out/heap1-timing-90") / name)
    rows = np.asarray(rows)
    def local_step(p):
        return rotate(p[80:81, 1:3] - p[70:71, 1:3], -p[70:71, 3])
    print(name, score(rows, truth, statuses))
    print("220-230 error (m)", np.linalg.norm(local_step(rows) - local_step(truth)))
'@ | & $python -
```
