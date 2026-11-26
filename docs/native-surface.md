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

## Чего нет

- Отдельного модульного теста нет; проверка — совпадение поз до 1e-15 м и статусов на полной официальной инспекции.
- Сборка с MSVC не проверялась, только MinGW g++.
