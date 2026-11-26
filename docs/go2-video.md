# Видео инспекции Go2

[Вид сверху](https://github.com/Ivana-banana/MGTS/releases/download/mgts-data-2026-11-26/go2-rviz-top-a5ad606.mp4) · [Вид от третьего лица](https://github.com/Ivana-banana/MGTS/releases/download/mgts-data-2026-11-26/go2-rviz-third-person-a5ad606.mp4) · [Исходный результат прогона и замеры](https://github.com/Ivana-banana/MGTS/releases/download/mgts-data-2026-11-26/go2-live-full-a5ad606.zip).

Оба ролика показывают один фрагмент официальной инспекции примерно с 16-й по 62-ю секунду. Они записаны одновременно из двух окон RViz при воспроизведении 1×. Сверху видно положение на карте, камера от третьего лица следует за позой робота. Серый цвет показывает фиксированную карту, оранжевый — текущий скан, голубая линия — уже пройденный путь. Розовая стрелка показывает публикуемую позу.

Полный прогон локализации — a5ad606167c82d7d462a3de9dc2fae889315cf06. Том result_0.mcap имеет SHA-256 e79e4bf23ed03e1e9ea0a32b850fca66efed3b6783912f52eaebd4384711f2e1. Для показа добавлены 1 524 скана с точными метками и визуальные слои траектории и камеры. При сравнении базового и дополненного bag совпали метки, топики и сериализованные байты всех 1 526 исходных сообщений (две карты и 1 524 позы). На первом скане поза отсутствует.

| Файл в комплекте #200 | Байт | SHA-256 |
|---|---:|---|
| video/top.mp4 | 803518 | d61b823b72edd3373b621ad5bf2c9497523dfe664cdb1ee180b5796ed7119c7a |
| video/third-person.mp4 | 24113590 | 4fef41d3ef5537425d6a2ae5377ddcb3438bc9c6f5f9af16f6f10138736186d0 |

MP4, H.264, 822 × 660, 12 кадров/с, 46 с, без звука. Захвачена только область сцены RViz. Релиз находится в закрытом репозитории команды. Порядок выдачи доступа экспертам отслеживает #186.

## Воспроизведение

В локальном проектном окружении добавьте текущий скан к результату. Для этого прогона все опубликованные позы имеют статус matched. Если в другом прогоне есть прогнозные позы, используйте отдельный топик подтверждённых поз для слоя сканов.

~~~~powershell
python -m mgts_nav.current_scan data/official/inspection out/issue197/live-full/result out/issue191/replay --pose-topic /mgts/pose3d
~~~~

В отдельных PowerShell с активированным ROS 2 Humble и одинаковым ROS_DOMAIN_ID запустите визуальный узел, откройте нужные ракурсы и затем воспроизведите bag. Узел строит отображаемую линию и TF-кадр камеры из полученных поз. Настройка окружения описана в [инструкции для Windows](go2-windows-ros.md).

~~~~powershell
python -m tools.go2_replay_view
rviz2 -d configs/go2-video-top.rviz
rviz2 -d configs/go2-video-third-person.rviz
ros2 bag play out/issue191/replay --qos-profile-overrides-path configs/go2-result-qos.yaml
~~~~

## Объёмные ракурсы RViz

[Вид сверху с поверхностями стен](https://github.com/Ivana-banana/MGTS/releases/download/mgts-data-2026-11-26/go2-rviz-surface-top-a5ad606.mp4) · [Камера за роботом](https://github.com/Ivana-banana/MGTS/releases/download/mgts-data-2026-11-26/go2-rviz-surface-third-person-a5ad606.mp4) · [Проверка и контрольные суммы](https://github.com/Ivana-banana/MGTS/issues/211).

В обоих ракурсах корпус с четырьмя ногами движется по опубликованным позам. Модель использует фиксированную стойку. Поверхности построены из наблюдаемых ячеек карты обхода. Сетка сохраняет разрывы между ячейками, поэтому часть стен выглядит фрагментарно. В панели Displays пункт `Исходное облако карты` включает оригинальные точки.

Для записи на Linux нужны собранный образ организаторов `hakatonius:latest` с ROS 2 Humble, Docker, Xvfb, xdotool, xwininfo, FFmpeg и ffprobe. Распакуйте `go2-live-full-a5ad606.zip` в `out/go2-release` и запустите запись из корня проекта.

~~~bash
bash tools/capture_go2_rviz.sh top out/go2-release out/go2-rviz-surface-top-a5ad606.mp4
bash tools/capture_go2_rviz.sh third-person out/go2-release out/go2-rviz-surface-third-person-a5ad606.mp4
~~~

Для интерактивного просмотра в терминалах с ROS 2 Humble используйте один `ROS_DOMAIN_ID`. Запустите узел показа и выбранный профиль RViz, затем воспроизведите исходный bag.

~~~bash
python -m tools.rviz_go2 out/go2-release/map.npz --wall-alpha 0.68
rviz2 -d configs/go2-surface-top.rviz
ros2 bag play out/go2-release/bag --rate 1 --qos-profile-overrides-path configs/go2-result-qos.yaml
~~~

Для вида от третьего лица откройте профиль `configs/go2-surface-third-person.rviz` и запустите узел с параметрами `--wall-top 0.60 --wall-alpha 0.45`. Низкий полупрозрачный срез сохраняет видимость корпуса возле стен.
