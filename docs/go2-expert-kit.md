# Комплект эксперта · Финал Go2

Откройте `final-go2.pdf` для защиты, `video/top.mp4` для обзора карты и `video/third-person.mp4` для просмотра инспекции с движущейся камерой. Все три файла открываются локально. Ссылки презентации на командный GitHub требуют доступа к приватному репозиторию.

| Материал | Путь после распаковки |
|---|---|
| Карта, 38 251 воксель, шаг 12 см | `map.npz` |
| ROS 2 bag, карта и 1 524 позы | `bag/metadata.yaml`, `bag/result_0.mcap` |
| Время обработки каждого скана | `latency.csv` |
| Исходники и тесты версии a5ad606167c82d7d462a3de9dc2fae889315cf06 | `source.zip` |
| Профиль, команды исходного прогона и контрольные суммы его артефактов | `profile.json`, `run.json`, `official-run-index.json` |
| Узел визуализации и конфигурации RViz из c935bb13f1f8348426d3ea974417b8164c100184 | `viewer/tools`, `viewer/configs` |
| Оценка по сканам, ограничения и способ записи видео | `docs/go2-independent-check.md`, `docs/go2-video.md` |

Исходные записи обхода и инспекции можно скачать у организатора. Для повторного запуска распакуйте их в `inputs/mapping` и `inputs/inspection` так, чтобы в каждой папке лежали `metadata.yaml` и том MCAP. [Данные организаторов](https://github.com/ivr0ma/hackathon/tree/ecbeeb681f7fe1e274402623da27444f63a4c61a/hackathon)

## Контроль целостности

Из каталога распакованного комплекта в PowerShell выполните:

```powershell
$hashes = Get-Content -Raw sha256.json | ConvertFrom-Json
foreach ($item in $hashes.PSObject.Properties) {
    $actual = (Get-FileHash -LiteralPath $item.Name -Algorithm SHA256 -ErrorAction Stop).Hash
    if ($actual -ne $item.Value) { throw "SHA-256: $($item.Name)" }
}
"Проверено файлов: $($hashes.PSObject.Properties.Name.Count)"
```

## Просмотр и повторный запуск

Активируйте ROS 2 Humble с поддержкой MCAP. Подготовка Windows-окружения описана в `docs/go2-windows-ros.md`. Для просмотра выполните по одной команде в двух терминалах с одним ROS_DOMAIN_ID:

```powershell
rviz2 -d viewer/configs/go2-result.rviz
```

```powershell
ros2 bag play bag --qos-profile-overrides-path viewer/configs/go2-result-qos.yaml
```

Для вычисления поз заново на поставляемой карте выполните из каталога комплекта. Используйте новое имя выходного каталога при каждом запуске.

```powershell
$env:PYTHONPATH = (Resolve-Path source.zip).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
python -m tools.go2_live_replay inputs/inspection map.npz replay
```

Для повторного построения самой карты из обхода и обработки инспекции в офлайн-режиме в окружении с зависимостями из `requirements.txt` внутри `source.zip`:

```powershell
python -m mgts_nav.localize3d inputs/mapping inputs/inspection --out rebuilt
```

Опубликованный прогон использует штатную одометрию как начальное приближение. На первом из 1 525 сканов инспекции одометрия ещё не пришла. Время в CSV измерено от входа облака в обработчик до записи статуса после выдачи позы. Независимой эталонной траектории у команды нет.
