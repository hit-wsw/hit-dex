with open('logs/log.txt', 'r') as file:
    lines = file.readlines()

filtered_lines = [line for line in lines if not line.rstrip().endswith('it/s]')]

with open('logs/pro_log1-14.txt', 'w') as file:
    file.writelines(filtered_lines)