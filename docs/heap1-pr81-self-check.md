# Самопроверка профиля PR №81 на записи HEAP-1

Проверена ревизия `22f62d3871111cb280d0a3659d8cc817ba4f105e` на записи GrandTour HEAP-1 из [эксперимента №80](https://github.com/Ivana-banana/MGTS/issues/80). В mapping за 29,90 с записаны 300 сообщений Velodyne и 11 962 сообщения IMU. Одометрии нет. Во втором запуске добавлен только тангаж +12° в системе base, ожидаемая обратная поправка −12°. Пара профилей, выборка и бюджет [зафиксированы до измерений](https://github.com/Ivana-banana/MGTS/issues/83#issuecomment-7148262958).

## Воспроизведение

Из корня MGTS в Windows PowerShell, после подготовки `out/heap1-long-80` по №80. Использованы CPython 3.10.10 x64, NumPy 2.2.6, SciPy 1.15.3 и rosbags 0.11.5. SHA-256 проверенного mapping.db3 — `3405dd5688a870ea2519bfe1a930bf1877ccc152937c24860434a0ccfdf104b3`, штатного profile.json — `1637bebe6db4605e6ff5b62ec080100805f11357112eb30d8460fe48abb5da7a`.

```powershell
$python = '.\.venv-mgts\Scripts\python.exe'
git fetch origin refs/pull/81/head
New-Item -ItemType Directory -Force out/pr81-83 | Out-Null
git archive --format=zip -o out/pr81-83/code.zip 22f62d3871111cb280d0a3659d8cc817ba4f105e mgts_nav
@'
import json
from pathlib import Path
from scipy.spatial.transform import Rotation
cfg = json.loads(Path("out/heap1-long-80/profile.json").read_text())
rotation = Rotation.from_quat(cfg["base_from_lidar_xyzw"])
cfg["base_from_lidar_xyzw"] = (Rotation.from_euler("y", 12, degrees=True) * rotation).as_quat().tolist()
Path("out/pr81-83/pitch.json").write_text(json.dumps(cfg))
'@ | & $python -
$runner = "import sys,runpy; sys.path.insert(0,'out/pr81-83/code.zip'); import mgts_nav; print(mgts_nav.__file__); assert 'code.zip' in mgts_nav.__file__; runpy.run_module('mgts_nav.self_check',run_name='__main__')"
foreach ($case in 'nominal','pitch') {
    $profile = if ($case -eq 'pitch') {'out/pr81-83/pitch.json'} else {'out/heap1-long-80/profile.json'}
    & $python -X utf8 -c $runner out/heap1-long-80/mapping --config $profile --out "out/pr81-83/$case-report.json" --max-scans 60 --budget 180 -q
    Write-Output "$case exit=$LASTEXITCODE"
}
```

## Результаты

В проектной .venv-mgts при шаге выборки 5 оба профиля сопоставили 60 из 60 сканов. Оба запуска завершились кодом 1.

| Профиль | Предупреждения | Согласованность исходная → лучшая | Поправка крена / тангажа | Гипотезы / перебор | Время |
|---|---|---|---|---|---|
| Штатный | clock_imu | 0,1542 → 0,1672 | 0° / 0° | 17, завершён | 144,8 с |
| Тангаж +12° в base | tilt, clock_imu | 0,0153 → 0,1757 | 0° / −15° | 26, исчерпан бюджет | 186,8 с |

Прирост согласованности составил 0,0130 и 0,1604 при пороге предупреждения 0,04. Для внесённого наклона знак поправки верен, модуль превышает требуемый на 3°. При ограничении на время число проверенных гипотез зависит от загрузки машины. Начатая оценка и заключительные проверки могут завершиться после 180 с.

Зеркальная установка дала согласованность 0,0012 и 0,0142 соответственно, предупреждений по зеркалу нет. Без одометрии курс имеет статус `not_measured`, а проверка часов одометрии отсутствует. Для часов IMU оба профиля дали лаг −55 мс и предложили поправку +55 мс при пиковой корреляции 0,719 и 0,717. Истинная задержка неизвестна, поэтому по этому эксперименту нельзя подтвердить поправку часов. Эталон CPT7 в расчёте не использован.

Первый замер выполнялся в соседнем окружении Transport_MSK с теми же версиями Python и трёх указанных библиотек. Повтор в проектном окружении воспроизвёл предупреждения, величины поправок и оценки согласованности. В первом замере перебор для изменённого профиля охватил 22 гипотезы, время для двух профилей составило 142,7 и 186,2 с. Вывод ограничен одним окном одной миссии.
