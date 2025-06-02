import json
import pandas as pd
import transformers
from transformers import pipeline
import torch
import accelerate

# adapt to test and val

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

classifier = pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli',
    device_map='auto'
)

labels = [
    'an answer to: "Please describe a situation where you were presented with a problem outside of your comfort zone and where you were able to come up with a creative solution."', #q1
    'an answer to: "Tell us about a time when you have failed or made a mistake. What happened? What did you learn from this experience?"', #q2
    'an answer to: "Describe a situation in which you got a group of people to work together as a team. Did you encounter any issues? What was the end result?"' #q3
]


# data
with open(r'..\shared task\data\humility_comments_reduced.json', 'r') as f:
    data = json.load(f)





for i, user_data in enumerate(data):
    print(f'{i+1}/{len(data)}')
    user_comments = user_data.get('comments')
    if user_comments and isinstance(user_comments, list) and len(user_comments) > 0:
        try:
            results = classifier(user_comments, labels, multi_label=True)

            user_results = []
            for i, result in enumerate(results):
                result = {label: round(score,4) for label, score in zip(result['labels'], result['scores'])}
                result_dict = {
                    'Q1_score': result['an answer to: "Please describe a situation where you were presented with a problem outside of your comfort zone and where you were able to come up with a creative solution."'],
                    'Q2_score': result['an answer to: "Tell us about a time when you have failed or made a mistake. What happened? What did you learn from this experience?"'],
                    'Q3_score': result['an answer to: "Describe a situation in which you got a group of people to work together as a team. Did you encounter any issues? What was the end result?"']
                }
                user_results.append(result_dict)

        except:
            print('somethign wrong~!!!t-t')
    user_data['comment_classifications'] = user_results

with open('q_scored_val.json','w',encoding='utf-8') as f:
    json.dump(data,f)



