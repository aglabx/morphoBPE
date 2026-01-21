import os
import subprocess
import glob

# Папка с файлами
folder_path = "/media/eternus1/nfs/projects/users/ichelombitko/texts"  # Укажите путь к папке, если файлы находятся в другом месте

# Шаблон файлов
file_pattern = os.path.join(folder_path, "wikipedia_*.text")

# Найти все файлы, соответствующие шаблону
files = glob.glob(file_pattern)

# Путь к скрипту
script_path_process_files = "~/Dropbox/workspace/story/morphoBPE/process_files"
script_path_tf_df = "~/Dropbox/workspace/story/morphoBPE/tf_df"

# Запуск команд для каждого файла
for ii, file in enumerate(files):
    print(f"Processing file {ii+1}/{len(files)}: {file}")
    for script_arg in ["latin", "cyrillic"]:
        # command = f"time {script_path_process_files} {file} {script_arg}"
        # print(f"Executing: {command}")
        # subprocess.run(command, shell=True)
        step1_file = file.replace(".text", f".{script_arg}.step1")
        if os.path.getsize(file) > 100:
            command = f"time {script_path_tf_df} {step1_file}"
            # print(f"Executing: {command}")
            subprocess.run(command, shell=True)
    