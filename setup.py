#!/usr/bin/env python3
import os
import subprocess
import sys

def get_python39_command():
    """
    Ищет команду для запуска Python 3.9.
    На Windows сначала пытается ["py", "-3.9"], затем ["python3.9"].
    На остальных системах – ["python3.9"], затем ["python"].
    Если Python 3.9 не найден, предлагает продолжить с текущей версией.
    """
    candidates = []
    if os.name == 'nt':
        candidates = [["py", "-3.9"], ["python3.9"]]
    else:
        candidates = [["python3.9"], ["python"]]
    for candidate in candidates:
        try:
            result = subprocess.run(candidate + ["--version"],
                                      capture_output=True, text=True)
            version_output = (result.stdout + result.stderr).strip()
            if result.returncode == 0 and "3.9.0" in version_output:
                return candidate
        except Exception:
            continue
    print("Для работы требуется Python версии 3.9.0. Нажмите \'y\' чтобы продолжить с вашей версией (работа не гарантируется),"
          " нажмите \'n\' чтобы прекратить установку")
    answer = input().strip().lower()
    if answer == 'y':
        fallback = [sys.executable]
        print("Продолжаем с Python версии:", " ".join(fallback))
        return fallback
    else:
        sys.exit("Установка прервана пользователем.")

def create_virtualenv(python_cmd):
    """Создаёт виртуальное окружение в папке 'venv'."""
    venv_dir = "venv"
    if os.path.exists(venv_dir):
        print(f"Виртуальное окружение '{venv_dir}' уже существует. Пропускаем создание.")
    else:
        print("Создаём виртуальное окружение...")
        cmd = python_cmd + ["-m", "venv", venv_dir]
        subprocess.check_call(cmd)
        print("Виртуальное окружение создано.")

def activate_venv():
    """
    «Активирует» виртуальное окружение для subprocess-вызовов,
    изменяя PATH и устанавливая переменную VIRTUAL_ENV.
    """
    venv_dir = os.path.abspath("venv")
    if os.name == "nt":
        bin_dir = os.path.join(venv_dir, "Scripts")
    else:
        bin_dir = os.path.join(venv_dir, "bin")
    # Добавляем директорию с интерпретатором в начало PATH
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["VIRTUAL_ENV"] = venv_dir
    print("Виртуальное окружение активировано для subprocess-вызовов.")
    # Теперь вызов "python" будет ссылаться на интерпретатор внутри venv.
    return "python"

def install_requirements(python_cmd):
    """Устанавливает зависимости из requirements.txt, если такой файл найден."""
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        print("Устанавливаем зависимости из requirements.txt...")
        subprocess.check_call([python_cmd, "-m", "pip", "install", "-r", req_file],
                              env=os.environ)
    else:
        print("Файл requirements.txt не найден, пропускаем установку зависимостей.")

def install_fast_kan(python_cmd):
    """
    Устанавливает локальную библиотеку fast-kan из директории fast-kan.
    Эквивалентно выполнению:
        cd fast-kan
        pip install .
    """
    fast_kan_dir = "fast-kan"
    if os.path.exists(fast_kan_dir):
        print("Устанавливаем библиотеку fast-kan из локальной директории...")
        try:
            subprocess.check_call([python_cmd, "-m", "pip", "install", "."],
                                  cwd=fast_kan_dir, env=os.environ)
        except subprocess.CalledProcessError as e:
            print("Ошибка при установке fast-kan. Убедитесь, что в папке fast-kan есть корректный setup.py или pyproject.toml.")
            sys.exit(e.returncode)
    else:
        print("Папка fast-kan не найдена, пропускаем установку fast-kan.")

def main():
    python_cmd = get_python39_command()
    print("Используем команду для Python:", " ".join(python_cmd))
    create_virtualenv(python_cmd)
    # «Активируем» виртуальное окружение для последующих вызовов pip
    venv_python = activate_venv()
    install_requirements(venv_python)
    install_fast_kan(venv_python)
    print("Настройка завершена.")



if __name__ == "__main__":
    main()