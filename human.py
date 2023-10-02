import json

file = "examples_v2"

with open(f'{file}.jsonl', 'r') as in_file:
    with open(f'{file}_strong.jsonl', 'w') as strong_file:
        with open(f'{file}_weak.jsonl', 'w') as weak_file:
            while True:

                line = in_file.readline()

                if not line:
                    break

                data = json.loads(line)

                prompt = data["prompt"]

                print(f'\nPrompt:\n\n{prompt}')

                meaningful = None

                while True:
                    print(f'\nIs this a meaningful prompt (y/n):')
                    i = input()
                    if i == "n":
                        meaningful = False
                        break
                    elif i == "y":
                        meaningful = True
                        break
                    else:
                        continue

                if meaningful:
                    strong_file.write(line)
                else:
                    weak_file.write(line)
