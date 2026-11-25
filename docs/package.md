# Архив решения MGTS

В ZIP входят исходники навигации и инструментов, тесты, конфигурации, пользовательские руководства и файлы сборки контейнера. Перечень зафиксирован в `tools/package_solution.py` выбранной ревизии. Для работы с архивом получателю нужны Python 3.10 и зависимости из `requirements.txt`. Для тестов дополнительно нужен pytest. История Git, записи сенсоров и каталоги результатов хранятся отдельно. Доска участников и правила рабочего каталога доступны в репозитории.

Вклад [M1hairu, PR №28](https://github.com/Ivana-banana/MGTS/pull/28) включён с учётом текущего состава проекта. Проверка выпуска сверяет наличие модулей, инструментов, тестов и относительных ссылок руководств. Тест самого упаковщика выполняется в Git-репозитории, поскольку он собирает исторические ревизии. В отдельной поставке тест наследования режима UTF-8 дочерними процессами также требует Git и пропускается. Для тестов с полевыми данными и C++ действуют условия из README. Dockerfile задаёт Python 3.10; проверка этой ревизии выполнена на Windows в локальном Python.

## Собрать и проверить

Из корня MGTS в локальном Windows PowerShell выберите ревизию и создайте новый ZIP на томе NTFS. Повторная сборка той же ревизии тем же упаковщиком даёт побайтово одинаковый файл. Git отдаёт содержимое коммита, поэтому параллельные правки рабочего дерева и настройки окончаний строк не влияют на архив.

```powershell
$python = '.\.venv-mgts\Scripts\python.exe'
$revision = git rev-parse HEAD
$archive = "out\packages\mgts-$revision.zip"
& $python tools/package_solution.py $archive --revision $revision
& $python tools/package_solution.py $archive --verify --revision $revision
Get-FileHash -Algorithm SHA256 $archive
```

Упаковщик выводит полный коммит, SHA-256 и путь. Проверка заново вычисляет ожидаемый хеш из ревизии, сверяет состав и читает каждую запись для проверки CRC. Команда сборки требует новое имя назначения. Для передачи сохраните вместе с ZIP идентификатор коммита и выведенный хеш.

## Проверить запуск

### Подготовить зависимости Windows для запуска без сети

Для CPython 3.10 x64 перечень всех 20 колёс с SHA-256 хранится в `requirements-win-amd64-py310.lock`. Один раз на машине с сетью из корня MGTS выполните:

```powershell
$kit = 'out/offline-kit'
$lock = 'requirements-win-amd64-py310.lock'
& $python -m pip download --only-binary=:all: --require-hashes -r $lock -d "$kit/wheels"
if ($LASTEXITCODE) { throw 'Не удалось подготовить колёса' }
```

Передайте ZIP, файл фиксации версий и каталог wheels вместе. На машине с установленным CPython 3.10 x64 создайте новое окружение и выполните:

```powershell
& $python -m venv "$kit/env"
if ($LASTEXITCODE) { throw 'Не удалось создать окружение' }
$python = "$kit/env/Scripts/python.exe"
& $python -m pip --isolated install --no-index --no-cache-dir --find-links "$kit/wheels" --require-hashes -r $lock
if ($LASTEXITCODE) { throw 'Проверьте наличие колеса указанной версии и его SHA-256' }
& $python -m pip check
if ($LASTEXITCODE) { throw 'Проверка зависимостей завершилась ошибкой' }
```

Далее используйте команды запуска непосредственно из ZIP ниже. По умолчанию построение сетки использует Python. Для ускорения можно передать совместимую локальную DLL через `solve --grid-dll`. Вычисления на Python доступны без компилятора.

В репетиции выпуска `dc74ed1` на Windows AMD64 с CPython 3.10.10 подготовка 76 675 529 байт колёс заняла 10,24 с с доступным кешем pip. Чистое окружение приняло все 20 зависимостей из локального каталога. В отдельном пустом каталоге pip указал отсутствующий six 1.17.0, а после изменения байтов копии колеса вывел ожидаемый и фактический SHA-256. В обоих случаях установка завершилась с кодом 1.

### Запустить из рабочей копии

Для локальной приёмки проверьте ZIP указанной выше командой, затем запустите CLI в разрешённой рабочей папке MGTS. Для проверки конкретного выпуска исходники и конфигурация рабочего дерева должны соответствовать выбранному коммиту. Команда ниже проверяет изменения версионируемых файлов; при расхождении отложите прогон этой ревизии до согласования с другими сессиями.

```powershell
git diff --quiet $revision -- mgts_nav configs/example.json requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Рабочие исходники отличаются от выпуска' }
& $python -m mgts_nav --help
& $python -m mgts_nav.rehearse "out\package-rehearsal-$revision"
```

Репетиции нужен новый выходной каталог. Она создаёт синтетические записи обхода и инспекции и проверяет карту, локализацию и сравнение с одометрией.

### Проверить команды непосредственно из ZIP

Из корня MGTS запустите проверку выпуска:

```powershell
& $python -m pytest -q -s tests/test_package.py
```

Тест собирает архив из HEAD, проверяет загрузку диагностических команд из ZIP и запускает полный `solve` на синтетической паре из рабочего дерева и из архива. Сравниваются байты карты, траекторий обхода и инспекции, плотной траектории и кинематической одометрии. При запуске из ZIP все дочерние этапы используют тот же архив. Интерпретатор и внешние зависимости должны быть установлены в окружении.

Для полевого повтора подготовьте запись согласно [руководству HEAP-1](public-data.md). Переменные `$python` и `$archive` задаются при сборке выше. Обработка и проверка могут выполняться непосредственно из ZIP.

Для первого запуска получателю достаточно архива и окружения с установленными зависимостями. Из локального PowerShell передайте путь к ZIP и новому выходному каталогу. Репетиция создаёт обе записи, читает профиль из архива и запускает обход, инспекцию и базовую одометрию из той же версии программы.

```powershell
$launch = "import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module(sys.argv.pop(1),run_name='__main__',alter_sys=True)"
& $python -I -c $launch $archive mgts_nav.rehearse out/packed-first-demo
```

```powershell
$launch = "import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module(sys.argv.pop(1),run_name='__main__',alter_sys=True)"
$d = 'out/heap1-revisit-v2'
$o = 'out/packed-heap1-solve'
& $python -I -c $launch $archive mgts_nav.solve "$d/mapping" "$d/inspection" --config "$d/profile.json" --start-search delayed --out $o
if ($LASTEXITCODE -eq 0) { & $python -I -c $launch $archive mgts_nav.solve_audit "$d/mapping" "$d/inspection" $o }
```

### Повторить сохранённый расчёт из ZIP

Передайте получателю архив, обе исходные записи, каталог завершённого решения с `solve-options.json` и исходный манифест. Для решения с зонами нужен в том числе `profiles/zones.json` внутри решения. Задайте пути к полученным файлам и новый каталог результата. Переменные `$python`, `$archive` и `$launch` определены выше.

```powershell
$previous = 'out/solution'
$mapping = 'data/mapping'
$inspection = 'data/inspection'
$manifest = 'out/manifest.json'
$repeat = 'out/repeated-from-zip'
& $python -I -X utf8 -c $launch $archive mgts_nav.solve_manifest verify $mapping $inspection $previous $manifest
if ($LASTEXITCODE -ne 0) { throw 'Исходные файлы расходятся с манифестом' }
& $python -I -X utf8 -c $launch $archive tools.repeat_solve $previous $mapping $inspection $repeat
if ($LASTEXITCODE -ne 0) { throw 'Повтор solve завершился ошибкой' }
& $python -I -X utf8 -c $launch $archive mgts_nav.solve_manifest verify $mapping $inspection $repeat $manifest
```

Команда читает параметры и профили из предыдущего решения. Для ядра C++ передайте также исходную DLL через `--grid-dll`. Последняя проверка сверяет хеши входов и продуктов повтора с тем же манифестом, включая отчёт посещений зон.
