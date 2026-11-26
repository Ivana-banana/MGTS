# Индекс комплекта Go2

Индекс связывает заявленную версию решения и команды прогона с байтами, передаваемыми экспертам. После прогона #182 положите в отдельный каталог комплекта следующие файлы. Сохраните исходные имена томов внутри каталога bag.

| Путь в комплекте | Содержимое |
|---|---|
| run.json | Полный SHA коммита решения и команды выполненного прогона |
| inputs/bag_for_mapping.zip, inputs/bag_for_localization.zip | Два исходных архива из релиза mgts-data-2026-11-26 |
| profile.json | Конфигурация, с которой выполнен прогон |
| bag/metadata.yaml и тома | Выходной rosbag2. Список томов читается из relative_file_paths |
| latency.csv | Замеры задержки публикации позы из #182 |
| video/top.mp4, video/third-person.mp4 | Записи RViz сверху и от третьего лица из #191 |

В run.json укажите полную ревизию проверенного исходного кода в поле commit. В объекте commands сохраните строки mapping, localization и playback с фактическими командами, включая аргументы, из этого прогона. Команды сохраняются как текст. Проверяющий инструмент их не запускает.

Из корня MGTS в PowerShell соберите индекс рядом с файлами. Получатель скачивает те же байты вместе с index.json и запускает проверку. Порядок томов берётся из метаданных bag.

~~~~powershell
python -m tools.final_run_index build out/final-delivery
python -m tools.final_run_index verify out/final-delivery
~~~~

Верификация повторно вычисляет размер и SHA-256 каждого указанного файла и сравнивает весь индекс. При расхождении, отсутствии файла или команды процесс завершится с кодом 1. Воспроизведение bag, содержание видео и происхождение материалов от указанного коммита проверяются по результатам реального прогона.

## Проверенный промежуточный набор

Соседние run.json и index.json описывают только входные архивы и профиль. В каталоге inputs нужны исходные ZIP из [релиза MGTS](https://github.com/Ivana-banana/MGTS/releases/tag/mgts-data-2026-11-26). Профиль берётся из Git-объекта [configs/go2-official.json](https://github.com/Ivana-banana/MGTS/blob/815c6e9303f8de883a0dc8041edbc3d82ed436eb/configs/go2-official.json) в исходной кодировке с оригинальными переводами строк.

~~~~powershell
$bundle = 'out/verify-final-inputs'
New-Item -ItemType Directory -Force "$bundle/inputs" | Out-Null
gh release download mgts-data-2026-11-26 -D "$bundle/inputs" -p 'bag_for_*.zip'
Copy-Item docs/final-run/index.json "$bundle/index.json"
python -c "from pathlib import Path; import subprocess,sys; p=Path(sys.argv[1]); (p/'profile.json').write_bytes(subprocess.check_output(['git','show','815c6e9303f8de883a0dc8041edbc3d82ed436eb:configs/go2-official.json'])); (p/'run.json').write_bytes(Path('docs/final-run/run.json').read_bytes().replace(b'\r\n',b'\n'))" $bundle
python -m tools.final_run_index verify $bundle --allow-partial
~~~~

Поле missing перечисляет недостающие роли и команды. Без явного --allow-partial промежуточный набор проверку полного комплекта не проходит. Доступ экспертов к итоговой публикации отслеживается в #186.
