# Репетиция приёмки четырёх записей

Наборы A–D сгенерированы для [PR №35](https://github.com/Ivana-banana/MGTS/pull/35), seed 23. Это четыре варианта поставки синтетического маршрута. Исходная ревизия рабочего дерева перед репетицией — `99275cd`; прогоны выполнены в общей рабочей папке Windows при параллельной работе команды. Генератор и утилита разделения взяты из `997f7becc0f7806b0035e19789bc855c3e248541`. Для выбора профилей и интервалов использовались описания в `data/ORGANIZERS.md`, сообщения и калибровка. Каталог `truth` использован только после запуска.

| Запись | Первая пригодная карта, привязки / сканы | Инспекция, привязки / сканы | Покрытие одометрией, обход и инспекция | RMSE положения XY, обход и инспекция |
|---|---:|---:|---|---|
| A, MCAP и цепочка TF | 199/200 | 100/100 | 200/200 и 100/100 | 0,59 и 3,92 см |
| B, общий журнал | 199/200 | 100/100 | 200/200 и 100/100 | 0,57 и 4,06 см |
| C, наклонный лидар | 198/199 | 99/99 | 199/199 и 99/99 | 0,67 и 4,00 см |
| D, только лидар и одометрия | 199/200 | 100/100 | 177/200 и 90/100 | 0,55 и 3,74 см |

Первый скан каждой карты задаёт начало координат. Все остальные строки карты и все строки инспекции получили статус `matched`. RMSE вычислена по всем позам в исходной системе координат, без подгонки. Число поз совпало с числом эталонных строк во всех восьми траекториях, наибольшая разность сопоставленных времён составила 0,24 мкс.

## Подготовка в Windows PowerShell

Выполняйте из корня MGTS в новый каталог. Для воспроизведения требуется локальный Git-объект указанной ревизии PR. В штатной поставке внешняя утилита разделения может отсутствовать.

```powershell
$r = 'out/intake122-repeat'
$py = '.\.venv-mgts\Scripts\python.exe'
$env:PYTHONPATH = (Get-Location).Path
$rev = '997f7becc0f7806b0035e19789bc855c3e248541'
New-Item -ItemType Directory "$r/tools","$r/configs" | Out-Null
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
foreach ($tool in 'foreign_bags','split_recording') {
    git show "${rev}:tools/${tool}.py" | Set-Content "$r/tools/$tool.py" -Encoding UTF8
}
Copy-Item configs/example.json "$r/configs/"
& $py -X utf8 "$r/tools/foreign_bags.py" "$r/input" --seed 23
& $py -X utf8 "$r/tools/split_recording.py" "$r/input/b/data/session" --stops /demo/odom --min-stop-s 0.8
& $py -X utf8 "$r/tools/split_recording.py" "$r/input/b/data/session" --from 500 --to 520 --out "$r/b-map"
& $py -X utf8 "$r/tools/split_recording.py" "$r/input/b/data/session" --from 530 --to 540 --out "$r/b-ins"
```

В B неподвижные интервалы по заголовкам одометрии — 500–501, 519–524,05, 527,95–531 и 539–540 с. Описание указывает порядок «обход, остановка, переезд, инспекция». Срезы по 520 и с 530 с попали внутрь двух стоянок и исключили переезд. Если порядок участков неизвестен, запросите границы у организаторов. Разрез по времени записи может дать другие сообщения.

## Профили и запуск

Для каждого входа выполните `inspect --inventory`, полный `inspect` и `profile --verify-frames` по [инструкции приёмки](intake.md). Рабочий JSON содержит только объект `profile`. Для всех четырёх записей использованы `height_band_m = [0.15, 1.5]`, `max_sync_gap_s = 0.35` и `stationary_s = 0.5` при наличии IMU, иначе 0. Пути ниже даны относительно `$r`.

| Набор | Обход и инспекция | Что проверить и внести в оба профиля |
|---|---|---|
| A | `input/a/data/mapping`, `input/a/data/inspection_run2.mcap` | Черновик извлёк составную статическую калибровку в `base_footprint`. Отдельный MCAP инспекции передан напрямую. |
| B | `b-map`, `b-ins` | По описанию установки задать `base_from_lidar_xyz_m = [0.2,0,0.3]`, оба поворота — `[0,0,0,1]`. |
| C | `input/c/data/mapping`, `input/c/data/inspection` | В YAML проверить родителя `base_link`. Из `xyz_m` и `rpy_deg` для лидара получить перенос `[0.3,0,0.5]` и поворот xyzw `[0,0.08715574274765817,0,0.9961946980917455]`, для IMU — единичный поворот. Использован `Rotation.from_euler('xyz', rpy_deg, degrees=True).as_quat()`. По описанию лучей задать `"point_time":{"field":"t","unit":"ns"}`. |
| D | `input/d/data/mapping`, `input/d/data/inspection` | Оставить IMU выключенной. Черновик из `/tf_static` дал перенос лидара `[0.1,0,0.35]` и разворот на 180° вокруг X. Смещения часов оставлены нулевыми. Частичное покрытие зафиксировано перед запуском. |

Пример для D воспроизводит оба профиля, отчёты покрытия и полный запуск в новом выходном каталоге.

```powershell
foreach ($role in 'mapping','inspection') {
    $bag = "$r/input/d/data/$role"
    & $py -X utf8 -m mgts_nav inspect $bag --inventory
    & $py -X utf8 -m mgts_nav inspect $bag
    $p = ((& $py -X utf8 -m mgts_nav.profile $bag --verify-frames) | ConvertFrom-Json).profile
    $p.height_band_m = @(0.15,1.5)
    $p.max_sync_gap_s = 0.35
    $p.stationary_s = 0
    [IO.File]::WriteAllText("$((Get-Location).Path)/$r/d-$role.json", ($p | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
    & $py -X utf8 -m mgts_nav.sync $bag --config "$r/d-$role.json"
}
& $py -X utf8 -m mgts_nav.solve "$r/input/d/data/mapping" "$r/input/d/data/inspection" --config "$r/d-mapping.json" --inspection-config "$r/d-inspection.json" --out "$r/d-result"
```

Для A–C используйте те же команды с путями и калибровкой своей строки. Результаты для каждой пары — `map/map.yaml`, `map/map.pgm` и `inspection/trajectory.csv`. В D совпадение поз с эталоном при частичном покрытии не измеряет смещение часов. По устройству генератора заголовки одометрии опережают лидар на 0,25 с. Этот параметр в опыт не подставляли.
