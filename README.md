# Воспроизводимый вычислительный конвейер предварительной диагностики рациональных эллиптических кривых

Репозиторий реализует воспроизводимый конвейер для целочисленных коротких моделей Вейерштрасса `y^2 = x^3 + ax + b` над `Q`. Stage A — целочисленный диагностический классификатор A1–A6. Stage B — точный вызов SageMath `torsion_subgroup()` на стратифицированной референсной выборке.

## Среда и проверка

Зафиксированная среда результатов: SageMath 10.8, Sage Python 3.13.7, CMake 4.2.3, C++17 и Apple clang 21.0.0. Для сборки текущего C++-кода нужен Clang или GCC: реализация использует `__int128`.

```bash
conda env create -f environment.yml
conda activate rational-elliptic-diagnostics
python -m pip install -e .
python -m pytest -q
sage -python -m compileall -q src scripts tests
```

## Воспроизведение

Полная последовательность Stage A → Stage B → проверка → бенчмарк запускается из корня репозитория:

```bash
python scripts/run_stage_a.py --config configs/conference.yml --output-dir output
sage -python scripts/run_sage_reference.py --config configs/conference.yml --input output/stage_a_rows.csv --output-dir output
python scripts/verify_results.py --config configs/conference.yml --input-dir output --results-dir results
sage -python scripts/run_benchmark.py --config configs/conference.yml --sizes 10000 30000 100000 300000 --sample-per-size 1000 --repeats 3 --output-dir results
```

Stage A и верификатор поддерживают обычный `python`; бенчмарк также можно запустить через `python scripts/run_benchmark.py` при наличии `sage` в `PATH`.

| Файл | Содержание |
|---|---|
| `results/stage_a_summary.csv` | Численности и доли классов A1–A6 в Stage A. |
| `results/stage_b_reference.csv` | Точные порядки, инварианты и генераторы кручения для референсной выборки. |
| `results/calibration.csv` | Метрики на сбалансированной калибровке и эквивалентность Python/C++. |
| `results/benchmark_scaling.csv` | Медианные wall/CPU-времена и счётчики проверок для четырёх диапазонов. |
| `results/environment.json` | Обезличенная конфигурация машины и версии инструментов. |

## Границы интерпретации и лицензии

Stage A выполняет предварительную диагностику, а не заменяет точное вычисление Stage B. Референс из 2000 строк не охватывает все 200000 кривых. Точность на сбалансированной калибровке 300+300 не является оценкой распространённости или вероятности в генеральной совокупности. Нулевая наблюдаемая частота редкого класса не означает невозможность этого класса. Измеренные времена зависят от машины и методики запуска.

Код распространяется под MIT (`LICENSE-CODE`), данные в `data/` и `results/` — под CC BY 4.0 (`LICENSE-DATA`).
