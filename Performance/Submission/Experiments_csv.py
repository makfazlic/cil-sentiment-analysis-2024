import csv

with open('sample_submission.csv', 'r') as f:
    data = list(csv.reader(f, delimiter=","))

with open('out.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)


with open('sample_submission.csv', 'r') as f:
    data1= f.read()

with open('out.csv', 'r') as f:
    data2= f.read()

print(data1==data2)
