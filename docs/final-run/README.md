# Комплект официального прогона Go2 на 1×

[Единый комплект эксперта, ZIP 27,42 МБ](https://github.com/Ivana-banana/MGTS/releases/download/mgts-data-2026-11-26/go2-expert-a5ad606.zip) содержит финальный PDF, исполняемые исходники закреплённой версии, карту, выходной rosbag2, CSV и оба видео. SHA-256 архива `ef8d8b46f86162831cb1847b9a1bb4a227edb028470e7670fb695191ce62a294`. После распаковки откройте `README.md` для проверки 20 файлов и запуска в ROS 2. Чтобы повторить упаковку из проверенного каталога #200, выполните `python docs/final-run/build_expert_kit.py out/verify-go2/bundle out/go2-expert.zip`.

[Скачать результат для команды, ZIP 1,86 МБ](https://github.com/Ivana-banana/MGTS/releases/download/mgts-data-2026-11-26/go2-live-full-a5ad606.zip). Архив доступен участникам приватного репозитория. Доступ экспертов согласуется в #186. Оба ракурса из #191 доступны в том же релизе: [вид сверху](https://github.com/Ivana-banana/MGTS/releases/download/mgts-data-2026-11-26/go2-rviz-top-a5ad606.mp4) и [вид от третьего лица](https://github.com/Ivana-banana/MGTS/releases/download/mgts-data-2026-11-26/go2-rviz-third-person-a5ad606.mp4).

В архиве лежат выходной rosbag2 (metadata.yaml и result_0.mcap), карта map.npz, замеры latency.csv, конфигурация profile.json, команды run.json и промежуточный индекс. Полный [индекс с обоими видео](index.json) хранится в репозитории. Входные архивы перечислены с хешами в индексе и доступны отдельно из ранее опубликованного релиза. Размер ZIP с результатами 1 859 258 байт, SHA-256 `1b107f8ca9026ceb8c060e61452c85c2c87706493019a20f05ccd3c20303af89`.

## Что проверено

Скачанная копия содержит 38 251 воксель. Оба сообщения карты побайтно совпали по массивам float32 XYZ с map.npz. Из 1525 сканов инспекции получено 1524 позы в кадре map на точных метках исходных сканов. На первом скане одометрия отсутствовала. Для 1524 строк matched задержка от начала обработки скана в callback до записи результата после публикации позы составила p50 12,03 мс, p95 25,25 мс и p99 31,77 мс. Это время данного локального прогона ROS 2 на Windows.

Команды построения и локализации в [run.json](run.json). Версия кода воспроизведения a5ad606 содержит побайтно идентичную копию использованного скрипта. Отдельная ревизия профилей воспроизведения указана там же. Для просмотра откройте RViz до старта player и запустите в отдельном терминале активированного ROS 2:

```powershell
rviz2 -d configs/go2-result.rviz
```

```powershell
ros2 bag play out/verify-go2/bundle/bag --qos-profile-overrides-path configs/go2-result-qos.yaml
```

## Скачать и проверить

Из корня MGTS в PowerShell с доступом к приватному репозиторию. Входные ZIP нужны для полной сверки происхождения. Для просмотра достаточно архива с результатом.

```powershell
$dest = 'out/verify-go2'
gh release download mgts-data-2026-11-26 -D $dest -p go2-live-full-a5ad606.zip
Get-FileHash "$dest/go2-live-full-a5ad606.zip"
Expand-Archive "$dest/go2-live-full-a5ad606.zip" -DestinationPath "$dest/bundle"
gh release download mgts-data-2026-11-26 -D "$dest/bundle/inputs" -p 'bag_for_*.zip'
gh release download mgts-data-2026-11-26 -D "$dest/downloads" -p 'go2-rviz-*-a5ad606.mp4'
New-Item -ItemType Directory -Force "$dest/bundle/video" | Out-Null
foreach ($view in 'top','third-person') {
    Copy-Item "$dest/downloads/go2-rviz-$view-a5ad606.mp4" "$dest/bundle/video/$view.mp4"
}
Copy-Item docs/final-run/index.json "$dest/bundle/index.json"
python -m tools.final_run_index verify "$dest/bundle"
python -m mgts_nav.replay_audit data/official/inspection "$dest/bundle/latency.csv"
python -m mgts_nav.result_bag_audit data/official/inspection "$dest/bundle/bag" --pose-topic /mgts/pose3d --map-topic /mgts/map3d --timings "$dest/bundle/latency.csv" --allow-missing-initial-scans 1
```

Строгая приёмка включается через `--timings`: все заголовки сканов должны быть в журнале,
а точные метки записанных поз — совпадать с публикациями `matched` и `prediction_only`.
Для штатного Go2 явно разрешён пропуск только первого скана; любые другие пропуски,
расхождения с журналом и ноль поз дают ненулевой код. Успех — код 0; нарушение
приёмки bag — код 2, ноль поз в `replay_audit` — код 1.
Без `--timings` остаётся форматная сводка для намеренно неполных экспериментов:
ненулевое покрытие само по себе не подтверждает полноту результата.

Проверка всех десяти файлов скачанного комплекта прошла со статусом complete=True. Оба MP4 полностью декодированы FFmpeg. Размеры и SHA-256 совпали с файлами исполнителя #191 и данными релиза. [Описание записи и контрольные суммы видео](../go2-video.md).
