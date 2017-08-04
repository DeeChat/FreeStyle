import json
import random

if __name__ == '__main__':
    with open('chunks.json') as f:
        chunks = json.load(f)
    lines = [line for chunk in chunks for line in chunk[:-1] if len(''.join(line)) >= 5]
    print('sample from', len(lines))
    random.seed(614)
    lines = random.sample(lines, 10000)
    print('max sentence lengths:', max(map(len, lines)))
    with open('quiz.json', 'w') as f:
        json.dump(lines, f, ensure_ascii=False, indent=0)
    with open('quiz.txt', 'w') as f:
        for line in lines:
            f.write('{}\n'.format(''.join(line)))
