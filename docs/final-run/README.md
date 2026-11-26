# Комплект официального прогона Go2 на 1×

[Скачать результат для команды, ZIP 1,86 МБ](https://github.com/Ivana-banana/MGTS/releases/download/mgts-data-2026-11-26/go2-live-full-a5ad606.zip). Архив доступен участникам приватного репозитория. Доступ экспертов согласуется в #186. Для полной сдачи ожидаются два видео RViz из #191.

В архиве лежат выходной rosbag2 (metadata.yaml и result_0.mcap), карта map.npz, замеры latency.csv, конфигурация profile.json, команды run.json и индекс index.json. Входные архивы перечислены с хешами в индексе и доступны отдельно из ранее опубликованного релиза. Размер ZIP с результатами 1 859 258 байт, SHA-256 `1b107f8ca9026ceb8c060e61452c85c2c87706493019a20f05ccd3c20303af89`.

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
python -m tools.final_run_index verify "$dest/bundle" --allow-partial
python -m mgts_nav.result_bag_audit data/official/inspection "$dest/bundle/bag" --pose-topic /mgts/pose3d --map-topic /mgts/map3d
```

Проверка восьми файлов прошла на повторно скачанном результате с ранее проверенными входными ZIP. В [индексе](index.json) статус complete равен false. После получения обоих видео запишите их как video/top.mp4 и video/third-person.mp4 и выполните build и verify инструмента tools.final_run_index без --allow-partial.
