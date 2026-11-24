# Архив решения MGTS

Архив содержит модули навигации, тесты, README, зависимости, Dockerfile, пример профиля, конфигурацию RViz, руководства по запуску и оценке и короткий ответ квалификации. Тест самого упаковщика читает историю Git и остаётся в репозитории. Точный перечень задают `FILES`, `MODULES` и `TESTS` в `tools/package_solution.py`. При подготовке выпуска проверяйте изменения в разрешённых файлах. Локальные профили, сырые записи, журналы участников, окружения и результаты запусков в список не входят. Ссылки README на доску участников и правила агентов ведут к файлам полного репозитория. Новый модуль, тест или документ из ссылок нужно добавить в списки упаковщика, иначе `tests/test_package.py` не пройдёт.

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

В Linux те же шаги:

```bash
revision=$(git rev-parse HEAD)
archive=out/packages/mgts-$revision.zip
.venv/bin/python tools/package_solution.py "$archive" --revision "$revision"
.venv/bin/python tools/package_solution.py "$archive" --verify --revision "$revision"
sha256sum "$archive"
```

Упаковщик выводит полный коммит, SHA-256 и путь. Проверка заново вычисляет ожидаемый хеш из ревизии, сверяет состав и читает каждую запись для проверки CRC. Команда сборки требует новое имя назначения. Для передачи сохраните вместе с ZIP идентификатор коммита и выведенный хеш.

## Проверить запуск

Для локальной приёмки проверьте ZIP указанной выше командой, затем запустите CLI в разрешённой рабочей папке MGTS. Для проверки конкретного выпуска исходники и конфигурация рабочего дерева должны соответствовать выбранному коммиту. Команда ниже проверяет изменения версионируемых файлов; при расхождении отложите прогон этой ревизии до согласования с другими сессиями.

```powershell
git diff --quiet $revision -- mgts_nav configs/example.json requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Рабочие исходники отличаются от выпуска' }
& $python -m mgts_nav --help
& $python -m mgts_nav.rehearse "out\package-rehearsal-$revision"
```

Репетиции нужен новый выходной каталог. Она создаёт синтетические записи обхода и инспекции и проверяет карту, локализацию и сравнение с одометрией.

## Проверить только по архиву

Для приёмки без репозитория распакуйте ZIP в пустую папку и подготовьте окружение Python 3.10 по README. Тесты, репетиция и образ Docker используют только файлы архива. Пример для Linux:

```bash
unzip -q mgts-$revision.zip -d mgts && cd mgts
python3.10 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q
.venv/bin/python -m mgts_nav.rehearse out/rehearsal
docker build -t mgts . && docker run --rm mgts
```

Образ основан на `python:3.10-slim` и по умолчанию печатает справку CLI. Для репетиции в контейнере передайте `python -m mgts_nav.rehearse /tmp/rehearsal`.
