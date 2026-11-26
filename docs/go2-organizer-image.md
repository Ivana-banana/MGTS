# Сквозной прогон в образе организаторов

Нативный прогон Windows/Humble на a5ad606 подтвердил публикацию 1 524 поз по 1 525 официальным сканам, карту из 38 251 вокселя, p95 времени от входа в callback до отправки позы 25,25 мс. [Артефакты и границы измерения](https://github.com/Ivana-banana/MGTS/issues/197#issuecomment-7186009229). Для закрытия #182 нужен отдельный прогон в собранном образе организаторов. Ниже приведена процедура для ответственного на стенде, где разрешены Linux-контейнеры.

## Что входит в комплект

[Комплект организаторов](https://github.com/ivr0ma/hackathon/tree/ecbeeb681f7fe1e274402623da27444f63a4c61a/hackathon) закреплён на ecbeeb681f7fe1e274402623da27444f63a4c61a. Compose с профилем pc собирает сервис hacatonius, образ hakatonius, контейнер r_hacatonius. Единственный том связывает каталог ros2_ws возле Dockerfile с /ros2_ws. Стенд использует ROS 2 Humble, MCAP, RViz2 и сообщения Unitree. Скрипт play_viz.sh показывает входной bag. Код MGTS и его Python-зависимости в Dockerfile организаторов не включены отдельным шагом.

В статической проверке на Windows команда docker compose --profile pc config прочитала оригинальный файл; подтверждены рабочий каталог, том, профиль, имя контейнера и переменная DISPLAY. На Windows выполнен весь блок подготовки ниже. В томе появились обе metadata.yaml, MCAP обхода размером 786 513 854 байта и инспекции размером 710 308 659 байт, точка входа tools/go2_live_replay.py и профиль RViz. Из Git собран архив исходников a5ad606, проверены CRC и синтаксис трёх точек входа для Python 3.10. Фактические версии ROS, бинарных пакетов и доступ к графике внутри образа проверяются на стенде. Базовый образ указан тегом, а Unitree клонируется из текущей ветки; к отчёту приложите ID собранного образа и журнал сборки.

## Подготовка на Windows

В PowerShell из корня MGTS подготовьте свободный каталог. Команды загружают по закреплённому коммиту шесть файлов официального стенда и кладут в его том код решения и две уже распакованные официальные записи. ros2 bag info должен видеть в каждом каталоге metadata.yaml и соседний MCAP. Дополнительные профили RViz берутся из коммита f92e403.

~~~~powershell
$kit = 'out/organizer-182'
if (Test-Path $kit) {throw 'Для нового прогона укажите свободный каталог'}
$ws = "$kit/ros2_ws"
New-Item -ItemType Directory -Force "$ws/data/official", "$ws/view", "$ws/results" | Out-Null
$base = 'https://raw.githubusercontent.com/ivr0ma/hackathon/ecbeeb681f7fe1e274402623da27444f63a4c61a/hackathon'
foreach ($file in 'Dockerfile','docker-compose.yml','play_viz.sh','go2.rviz','.dockerignore','README.md') {
    Invoke-WebRequest "$base/$file" -OutFile "$kit/$file"
}
git archive --format=zip "--output=$kit/solution.zip" a5ad606167c82d7d462a3de9dc2fae889315cf06 -- configs mgts_nav tools requirements.txt
if ($LASTEXITCODE) {throw 'git archive failed'}
Expand-Archive "$kit/solution.zip" "$ws/MGTS"
git archive --format=zip "--output=$kit/view.zip" f92e403924a9b0942d4abdf349038d376ef26fd1 -- configs/go2-result.rviz configs/go2-result-qos.yaml
if ($LASTEXITCODE) {throw 'git archive failed'}
Expand-Archive "$kit/view.zip" "$ws/view" -Force
foreach ($bag in 'mapping','inspection') {
    Copy-Item -LiteralPath "data/official/$bag" -Destination "$ws/data/official/$bag" -Recurse
}
docker compose -f "$kit/docker-compose.yml" --profile pc config
~~~~

## Проверка на разрешённом контейнерном стенде

Следующие шаги требуют доступа к работающему движку Linux-контейнеров. Настройте вывод графики по официальному README для вашей машины. Сохраните журнал сборки и ID именно этого запуска. Команды PowerShell с docker exec выполняют проверку внутри контейнера.

~~~~powershell
docker compose -f "$kit/docker-compose.yml" --profile pc up -d --build
if ($LASTEXITCODE) {throw 'Official image build/start failed'}
docker inspect --format '{{.Image}}' r_hacatonius
docker exec r_hacatonius bash -lc 'source /opt/ros/humble/setup.bash && source /opt/unitree_msgs/setup.bash && printenv ROS_DISTRO && ros2 bag info /ros2_ws/data/official/mapping && ros2 bag info /ros2_ws/data/official/inspection'
~~~~

Внутри одного контейнера офлайн-расчёт выполняется отдельным Python-окружением с requirements.txt. ROS-узел работает системным Python с библиотеками Humble. Следующая установка добавляет пакеты для текущего контейнера, в том числе SciPy, NumPy и ROS Python API для облаков и TF. Для воспроизведения запишите вывод dpkg-query и pip freeze вместе с ID базового образа.

~~~~powershell
docker exec r_hacatonius bash -lc 'set -e; apt-get update; apt-get install -y python3-venv python3-scipy python3-numpy ros-humble-sensor-msgs-py ros-humble-tf2-ros-py; python3 -m venv /opt/mgts-offline; /opt/mgts-offline/bin/python -m pip install -r /ros2_ws/MGTS/requirements.txt; /opt/mgts-offline/bin/python -m pip freeze > /ros2_ws/results/offline-packages.txt; dpkg-query -W > /ros2_ws/results/system-packages.txt'
docker exec r_hacatonius bash -lc 'set -e; cd /ros2_ws/MGTS; source /opt/ros/humble/setup.bash; source /opt/unitree_msgs/setup.bash; /usr/bin/python3 -c "import rclpy, rosbag2_py, sensor_msgs_py, scipy; from mgts_nav.ros_localize3d import Localizer"'
~~~~

Сначала постройте растровую карту для исторического критерия #182 и воксельную карту для финальной локализации. Команда localize3d строит map.npz только по обходу и дополнительно сохраняет офлайн-траекторию инспекции в trajectory.csv. Следующая команда проигрывает инспекцию заново при 1× и записывает действительно опубликованные позы в live/result.

~~~~powershell
docker exec r_hacatonius bash -lc 'set -e; cd /ros2_ws/MGTS; p=/opt/mgts-offline/bin/python; d=/ros2_ws/data/official; o=/ros2_ws/results; $p -m mgts_nav map $d/mapping --config configs/go2-official.json --out $o/raster; $p -m mgts_nav.localize3d $d/mapping $d/inspection --out $o/full3d'
docker exec r_hacatonius bash -lc 'set -e; source /opt/ros/humble/setup.bash; source /opt/unitree_msgs/setup.bash; cd /ros2_ws/MGTS; /usr/bin/python3 -m tools.go2_live_replay /ros2_ws/data/official/inspection /ros2_ws/results/full3d/map.npz /ros2_ws/results/live; ros2 bag info /ros2_ws/results/live/result'
docker exec r_hacatonius bash -lc 'set -e; cd /ros2_ws/MGTS; p=/opt/mgts-offline/bin/python; i=/ros2_ws/data/official/inspection; o=/ros2_ws/results/live; $p -m mgts_nav.result_bag_audit $i $o/result --map-topic /mgts/map3d --pose-topic /mgts/pose3d; $p -m mgts_nav.replay_audit $i $o/timings.csv'
~~~~

Для проверки экспертом откройте два PowerShell. В первом запустите RViz, во втором — воспроизведение. После проверки выдачи поз повторите с поздним подключением RViz. [Порядок проверки карты и движения](go2-map-delivery.md).

~~~~powershell
docker exec r_hacatonius bash -lc 'source /opt/ros/humble/setup.bash && rviz2 -d /ros2_ws/view/configs/go2-result.rviz'
docker exec r_hacatonius bash -lc 'source /opt/ros/humble/setup.bash && ros2 bag play /ros2_ws/results/live/result --qos-profile-overrides-path /ros2_ws/view/configs/go2-result-qos.yaml'
~~~~

В каталоге Windows ros2_ws/results проверьте читаемость raster/map.yaml и указанного в нём PGM, строки full3d/trajectory.csv и файлы live/result. Приложите к #182 ID образа, вывод ROS_DISTRO, два аудита с числом пропусков, p50/p95/p99 из live/timings.csv и снимок обоих слоёв RViz. Зафиксируйте состав машины и храните файлы именно этой попытки под одним именем прогона.
