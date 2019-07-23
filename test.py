import json
write_data = open("tianmaojingling", "w")
with open("data/tianmao", "r") as data:
    for line in data:
        value = json.loads(line.strip())
        answer = value["comment"]
        write_data.write(answer + "\n")
write_data.close()