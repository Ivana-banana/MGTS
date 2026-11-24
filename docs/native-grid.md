# Обновление карты занятости на C++

На 40 облаках Velodyne из **открытой миссии GrandTour HEAP‑1** обновление карты заняло **13,10 с на Python и 1,86 с с ядром C++** (медианы трёх прогонов). Значения накопленных голосов совпали во всех **168 391 клетках** в каждом повторе. Кадры 150–189 записаны ANYmal 14 ноября 2024 года. Это независимый тест до получения записей организаторов. [Исходные файлы ETH Zürich](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/tree/af6b6dd6f91696f8ec8c98134e195514834911b2/2024-11-14-13-45-37)

| Реализация этапа | Замер 1, с | Замер 2, с | Замер 3, с |
|---|---:|---:|---:|
| Штатная `Grid.observe` | 13,10 | 12,95 | 13,19 |
| NumPy + C++17 через ctypes | 1,86 | 1,79 | 1,86 |

Предварительное профилирование всей команды `map` через cProfile дало 20,75 с, включая 16,34 с в `Grid.observe` и 1,55 с в `slam.match`. Отдельный замер в таблице выполнялся без профилировщика. Общий трекер один раз регистрирует сканы и сохраняет точки в системе карты. Обоим вариантам передаются одинаковые позиции датчика и концы лучей. В таймер входят отбор в радиусе 35 м, разбиение на клетки 8 см, удаление дублей, растеризация лучей, обновление голосов и, для C++, экспорт результата в словарь Python. Порядок реализаций чередуется. Чтение bag, ICP, сборка DLL и запись PGM находятся за пределами этого таймера. Коэффициент 7,04 относится только к этапу обновления.

Среда замера — Ryzen 5 5500U, локальная Windows x64, Python 3.10, MSVC 14.44 с оптимизацией `/O2`. На этой рабочей станции параллельно выполнялись тесты. Перед замером скрипт также проверяет граничные случаи — отрицательные координаты, округление на половине клетки, конфликт свободной и занятой клетки, пустое облако и насыщение голосов.

## Полное построение карты

Штатная команда `map` принимает `--grid-dll PATH`; `solve` передаёт этот же флаг при построении карты. По умолчанию обе команды работают на Python. Начальное и повторное построение после замыкания петли используют выбранную реализацию.

| Команда | Python, с | С C++, с | Проверка |
|---|---:|---:|---|
| Полная `map`, HEAP‑1, 40 сканов | 15,80 / 18,39 / 18,48 | 4,97 / 4,92 / 4,54 | PGM, YAML, CSV, summary.json и PNG совпали побайтово в трёх повторах |
| Полная `solve --close-loop`, синтетический MCAP, 200 + 100 сканов | 15,11 | 7,15 | Петля подтверждена; карта, результат замыкания, траектории обхода, инспекции и одометрии совпали побайтово |

Медианное ускорение полной `map` на этом фрагменте составило **3,74 раза**. Измерялось время нового процесса от запуска до завершения записи файлов. Для оценки времени на финальном маршруте тот же замер потребуется повторить на записи организаторов.

## Повторить в PowerShell

Подготовьте `out/heap1-revisit-v2` по [инструкции к публичным данным](public-data.md). Для сборки нужен Visual Studio с компонентом MSVC x64. Команды выполняются из корня MGTS.

```powershell
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
New-Item -ItemType Directory -Force out/native-grid | Out-Null
cl.exe /nologo /O2 /EHsc /std:c++17 /LD tools/native_grid.cpp /Fo:out/native-grid/native_grid.obj /Fe:out/native-grid/native_grid.dll
if ($LASTEXITCODE -ne 0) { throw 'MSVC build failed' }
$data = 'out/heap1-revisit-v2'
& .\.venv-mgts\Scripts\python.exe -m tools.benchmark_native_grid "$data/mapping" --config "$data/profile.json" --dll out/native-grid/native_grid.dll
& .\.venv-mgts\Scripts\python.exe -m mgts_nav map "$data/mapping" --config "$data/profile.json" --grid-dll out/native-grid/native_grid.dll --out out/heap1-map-cpp
```
