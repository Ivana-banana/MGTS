# Ядро 3D-трекера на C++ (#190)

`tools/native_surface.cpp` повторяет посканную часть `SurfaceMap` из `mgts_nav/localize3d.py`: ICP point-to-plane (`align`) и остаток до карты (`score`). Нормали и признак плоскости по-прежнему считаются в Python, один раз при загрузке карты. Подключение — `mgts_nav/native_surface.py`. Выпущенный узел и комплект сдачи не менялись; DLL необязательна.

Поиск соседей заменён: вместо cKDTree точки разложены по кубам со стороной `MAX_MATCH_M` = 0,35 м, запрос смотрит 27 соседних кубов и пропускает те, где не может быть точки ближе. В радиусе ответ совпадает с cKDTree. За радиусом возвращается «промах»: трекер и так отбрасывает такие точки (`< MAX_MATCH_M`) и ограничивает их вклад в `score` этим радиусом.

## Сравнение на официальной инспекции

Карта `out/issue185/3d/map.npz` (38 251 воксель), один прогон каждого варианта подряд на одной машине (Intel Family 6 Model 186, Windows 11, Python 3.10, g++ 15.2 `-O3`).

| | Python + SciPy | C++ |
|---|---:|---:|
| Сканов / привязано | 1524 / 1524 | 1524 / 1524 |
| Регистрация, среднее / p95, мс | 3,42 / 4,34 | 1,57 / 1,94 |
| Весь скан, среднее / p95, мс | 4,35 / 5,35 | 2,50 / 3,01 |

Статусы совпали на всех 1524 сканах. Наибольшее расхождение позы — 1,8·10⁻¹⁵ м и 5,6·10⁻¹⁶ рад, то есть уровень округления.

Регистрация ускорилась в 2,2 раза, весь скан — в 1,7. Оставшееся время — декодирование, deskew и вызовы из Python. Для 1× запас был и раньше (медианный период облаков 72 мс), поэтому это запас для более слабой машины, а не условие сдачи.

## Воспроизведение

```powershell
g++ -O3 -std=c++17 -Wall -shared -static -o out/issue190-native/native_surface.dll tools/native_surface.cpp
& .\.venv-mgts\Scripts\python.exe -X utf8 -m tools.profile_go2_tracker data/official/inspection out/issue185/3d/map.npz out/issue190-native/python.csv
& .\.venv-mgts\Scripts\python.exe -X utf8 -m tools.profile_go2_tracker data/official/inspection out/issue185/3d/map.npz out/issue190-native/cpp.csv --native out/issue190-native/native_surface.dll
```

С `--native` профилировщик не отделяет поиск соседей: `neighbors_ms` равно 0, вся регистрация входит в `rest_of_align_ms`.

## Проверка с MSVC на Ryzen 5 5500U

Ядро `d5f8feb` проверено на Windows 10 x64 с MSVC 19.44.35216, Python 3.10.10, NumPy 2.2.6 и SciPy 1.15.3. Все четыре запуска использовали официальный MCAP инспекции и одну карту из прогона `a5ad606`. На общей машине последовательно запустили Python, C++, C++, Python. В таблице приведён диапазон между двумя прогонами каждого варианта. [Карта и контрольные суммы](final-run/index.json)

| Время на скан, мс | Python + SciPy | C++ (MSVC) |
|---|---:|---:|
| Регистрация, среднее | 7,51–8,09 | 4,66–4,70 |
| Регистрация, p95 | 10,76–13,39 | 6,32–6,41 |
| Вся обработка, среднее | 9,50–10,26 | 6,63–6,76 |
| Вся обработка, p95 | 13,27–16,36 | 8,93–9,01 |

Во всех четырёх прогонах метки 1 524 поз совпали с `/mgts/pose3d` в выходном rosbag `a5ad606`, а статусы — с `timings.csv` того же прогона. Первый из 1 525 сканов лежит вне покрытия одометрии. На каждой общей метке сравнили XYZ и угол относительного вращения Python и C++. Максимальное расхождение составило 1,83·10⁻¹⁵ м и 5,55·10⁻¹⁶ рад.

Профилировщик заранее загружает одометрию и измеряет вычисления от декодирования облака до обновления позы. Для p95 регистрации сначала сложили `neighbors_ms` и `rest_of_align_ms` для каждого скана. Доставку подписчику при воспроизведении 1× измеряли отдельно. [Проверка ROS 2](https://github.com/Ivana-banana/MGTS/issues/208)

Команды выполняются из корня MGTS в x64 Developer PowerShell для Visual Studio 2022. Для повтора укажите новый каталог вывода.

```powershell
$dir = 'out/issue190-msvc-d5f8feb'
$map = 'out/issue197/full3d/map.npz'
New-Item -ItemType Directory -Force $dir | Out-Null
cl.exe /nologo /O2 /MD /EHsc /std:c++17 /LD tools/native_surface.cpp "/Fo$dir/native_surface.obj" "/Fe$dir/native_surface.dll" /link "/IMPLIB:$dir/native_surface.lib"
foreach ($runName in 'python-1', 'msvc-1', 'msvc-2', 'python-2') {
    $profileArgs = @('data/official/inspection', $map, "$dir/$runName.csv")
    if ($runName.StartsWith('msvc')) { $profileArgs += @('--native', "$dir/native_surface.dll") }
    & .\.venv-mgts\Scripts\python.exe -X utf8 -m tools.profile_go2_tracker @profileArgs
}
```

## В узле ROS 2

`mgts_nav.ros_localize3d` принимает `--native-dll ПУТЬ`. Без флага узел работает как прежде, через SciPy. С флагом он пишет в журнал `C++ alignment core: ПУТЬ` и использует DLL для `align`. В Linux библиотеку собирают с `-fPIC`:

```bash
g++ -O3 -std=c++17 -Wall -shared -fPIC -o native_surface.so tools/native_surface.cpp
python3 -m mgts_nav.ros_localize3d out/issue185/3d/map.npz --timings times.csv --native-dll ./native_surface.so
```

`tools.go2_live_replay` принимает тот же флаг. Полный live replay инспекции на 1× в контейнере ros:humble (Docker Desktop, тот же ноутбук):

```bash
python3 -m tools.go2_live_replay data/official/inspection out/issue185/3d/map.npz /tmp/live-cpp --native-dll /tmp/native_surface.so
```

- статусы: 1524 `matched`, 1 `missing_odometry` (первый скан без одометрии, как в опубликованном прогоне);
- опубликовано 1524 позы, все совпали с офлайн-прогоном Python до 1,8·10⁻¹⁵ м;
- время обработчика p50 / p95 / p99: 2,79 / 5,95 / 8,27 мс.

Опубликованные 12,03 / 25,25 / 31,77 мс (#197) сняты на другой машине и без DLL, поэтому их нельзя прямо сравнивать с этими.

### Сравнение при воспроизведении 1× на Windows

На Ryzen 5 5500U последовательно воспроизвели всю официальную инспекцию с DLL MSVC и со штатным SciPy. Для обоих запусков использовали одну карту, единичное `map ← odom` и нативный ROS 2 Humble с Python 3.14.10 и SciPy 1.18.1. Измеряли время от входа облака в обработчик до записи статуса после публикации позы и TF, включая ожидание одометрии. Указаны перцентили по 1 524 строкам `matched` одного прогона каждого варианта.

| Обработчик при 1×, мс | SciPy | C++ (MSVC) |
|---|---:|---:|
| p50 | 12,31 | 8,06 |
| p95 | 25,32 | 21,58 |
| p99 | 31,78 | 24,36 |

В обоих выходных MCAP совпали все 1 524 метки поз. Единственный пропуск — первый скан вне покрытия одометрии. В каждой записи пришли два сообщения карты. Координаты XYZ в обоих поэлементно совпали с массивом `map.npz` в формате `float32` (38 251 точка). Все 1 525 статусов совпали с журналом `a5ad606`. Позы SciPy совпали с опубликованными, максимальная разница C++ относительно SciPy составила 1,83·10⁻¹⁵ м и 6,71·10⁻¹⁶ рад.

Для воспроизведения активируйте окружение ROS 2 и используйте DLL из раздела про MSVC. [Настройка PowerShell](go2-windows-ros.md)

```powershell
$dir = 'out/issue190-msvc-d5f8feb'
$inspection = 'data/official/inspection'
$map = 'out/issue197/full3d/map.npz'
python -m tools.go2_live_replay $inspection $map "$dir/cpp-live" --native-dll "$dir/native_surface.dll"
python -m tools.go2_live_replay $inspection $map "$dir/python-live"
foreach ($runName in 'cpp-live', 'python-live') {
    & .\.venv-mgts\Scripts\python.exe -m mgts_nav.result_bag_audit $inspection "$dir/$runName/result" --pose-topic /mgts/pose3d --map-topic /mgts/map3d
}
& .\.venv-mgts\Scripts\python.exe -m tools.compare_go2_replays "$dir/cpp-live" "$dir/python-live" 0
```

## Чего нет

- Проверка C++-ядра охватывает полную официальную инспекцию. Отдельного модульного теста у прототипа пока нет.
