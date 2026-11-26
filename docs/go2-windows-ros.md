# ROS 2 Humble на Windows для команды MGTS

Проверено из корня MGTS в PowerShell на Windows x64, в локальном окружении RoboStack. Оно подходит для запуска узла, чтения MCAP и подготовки записи RViz. Проверка образа организаторов ведётся в [#182](https://github.com/Ivana-banana/MGTS/issues/182).

## Окружение

На машине команды уже есть префикс `out/issue197/ros-env`. Для новой машины один раз создайте отдельное окружение из PowerShell с доступной командой conda:

```powershell
$prefix = Join-Path (Get-Location) 'out/issue197/ros-env'
conda create --prefix $prefix --override-channels -c conda-forge -c robostack-humble --strict-channel-priority ros-humble-ros2bag ros-humble-rosbag2-storage-mcap ros-humble-rclpy ros-humble-nav-msgs ros-humble-sensor-msgs-py ros-humble-tf2-ros-py ros-humble-rviz2 scipy
conda run --prefix $prefix python -m pip install PyYAML==6.0.4
```

В каждом терминале перейдите в корень MGTS. Если команда conda ещё не определена, загрузите PowerShell hook своего дистрибутива. На машине команды он доступен так:

```powershell
(& "$env:USERPROFILE/Anaconda3/Scripts/conda.exe" shell.powershell hook) | Out-String | Invoke-Expression
```

Активируйте префикс, добавьте DLL Ogre для RViz и выберите свободный домен. Для одного запуска во всех его терминалах используйте одинаковый номер.

```powershell
conda activate (Resolve-Path out/issue197/ros-env).Path
$env:Path = (Join-Path $env:CONDA_PREFIX 'Library/opt/rviz_ogre_vendor/bin') + ';' + $env:Path
$env:ROS_DOMAIN_ID = '119'
$env:ROS_LOCALHOST_ONLY = '1'
python -c "import sys, yaml, scipy, rosbag2_py, mgts_nav.ros_localize3d; print(sys.version, yaml.__version__, scipy.__version__, rosbag2_py.get_registered_readers())"
```

На проверенном win-64 префиксе Python 3.14.10, PyYAML 6.0.4, SciPy 1.18.1, зарегистрированы mcap и sqlite3. PowerShell hook RoboStack сообщает об отсутствующем Library/local_setup.ps1. В этой сборке есть local_setup.bat, и hook устанавливает AMENT_PREFIX_PATH. При стандартной обработке ошибок PowerShell активация завершилась и команды ниже отработали. Версии пакетов из каналов могут измениться. Чистая установка на второй машине отдельно не проверялась.

## Проверка и запуск

```powershell
ros2 bag info data/official/inspection
ros2 bag play data/official/inspection --topics /utlidar/cloud_base /utlidar/robot_odom
rviz2 --help
```

Для короткой проверки на командной машине использовали `out/issue197/inspection`. В этом MCAP 38 847 сообщений за 40,499 с, включая 563 cloud_base. Команда `ros2 bag play out/issue197/inspection --topics /utlidar/cloud_base --rate 5` завершилась с кодом 0. Импорт класса узла и запуск справки rviz2.exe прошли. В этом тесте проверялся запуск программ; визуальное качество и частота кадров оцениваются при записи.

Запустите узел в первом терминале с картой из полного прогона localize3d и новым путём CSV. Во втором воспроизведите инспекцию на исходной скорости:

```powershell
python -m mgts_nav.ros_localize3d out/issue197/full3d/map.npz --timings out/windows-live-times.csv
```

```powershell
ros2 bag play data/official/inspection --topics /utlidar/cloud_base /utlidar/robot_odom
```

Для просмотра готового диагностического bag из [#198](https://github.com/Ivana-banana/MGTS/issues/198), откройте RViz до воспроизведения. В отдельном терминале того же домена проиграйте запись с начала:

```powershell
rviz2 -d configs/go2-current-scan.rviz
```

```powershell
ros2 bag play out/issue198/full-overlay
```
