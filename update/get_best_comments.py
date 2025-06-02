import json
import pandas as pd

def has_duplicates(seq):
    return len(seq) != len(set(seq))


def get_best_comments(author):
    comment_scores = author.get('comment_classifications')
    q1_scores = []
    q2_scores = []
    q3_scores = []
    for i, comment_score in enumerate(comment_scores):
        q1_scores.append(comment_score.get('Q1_score'))
        q2_scores.append(comment_score.get('Q2_score'))
        q3_scores.append(comment_score.get('Q3_score'))
    q1_max = max(q1_scores)
    q2_max = max(q2_scores)
    q3_max = max(q3_scores)

    q1_index = q1_scores.index(q1_max)
    q2_index = q2_scores.index(q2_max)
    q3_index = q3_scores.index(q3_max)

    comments = author.get('comments')

    best_comments = [comments[q1_index],comments[q2_index],comments[q3_index]]
    return best_comments

with open('..\shared task\data\humility_comments_reduced.json','r',encoding='utf-8') as f:
    data = json.read(f) 
for i, author in enumerate(data):
    print(f'{i+1}/{len(data)}')
    best_comments = get_best_comments(author)
    if has_duplicates(best_comments):
        dupes +=1
    author['comments'] = best_comments
print(dupes)
with open('..\shared task\data\best_comments_only.json','w',encoding='utf-8') as f:
    json.dump(data)