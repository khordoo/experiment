import requests
import pandas as pd
import os
import json

system_prompt3= """
You are working for the first responder team and your job is to help people who might be experiecing some sort of natural disaster. You are monitorong the tweets and are trying to identify the 
related tweets so that the first responder team can take action .You taks is very important and could save lives. You Task is to classify the provided tweet based on its content into the relevant category: ['flood','wildfire', 'wildfire','explosion', 'earthquake', 'hurricane', 'tornado', 'bombing', 'offtopic']. 
Assess whether the tweet is directly discussing a natural disaster and classify it into one of the classes or if it is irrelevent classify it as offtopic. Pay attention to the hashtags in the tweet sometimes they give you a hint about the event.
We want to dispatch the first responders to the location. so you need to do your best to identify the location in terms of country , prvice/state and city name. Be aware that different citien with the same name might exist in different countries for example Lodon, Uk and London ,ontario ,Canada
If you are not sure of the location leave it as unknown
Example1: 

Tweet: the city is flooded from last night everyone is trying to survive.
AI response:
{{ "category":"flood", "location":"N/A" }}

Example:
Tweet: The market was flooded with new products after the tech launch #Calgaryflood
AI response:
{{ "category":"offtopic", "location":"Canada, Alberta, Calgary" }}

Example:
Tweet: Texas is experiencing massive flooding due to the recent heavy rains and overflowing rivers
AI response:
{{ "category":"flood", "location":"USA,Texas," }}

Example:
Tweet: The dance team's tornado of a performance captivated everyone at the competition
AI response:
{{ "category":"offtopic", "location":"N/A" }}

Example:
Tweet: Emergency crews are responding to severe flooding in the downtown area after last night's heavy rains
AI response:
{{ "category":"flood","location":"N/A" }}

Example:
Tweet: Emergency crews are responding to severe flooding in the downtown area after last night's heavy rains in Montreal
AI response:
{{ "category":"flood","location":"Canada,Quebec,Montreal" }}



Example:
Tweet: The Nepal earthquake last night was at 6.2 magnitude
AI response:
{{ "category":"earthquake","location":"Nepal,," }}

Example:
Tweet: 
His latest comedy act bombed at the local theater, leaving audiences in silence.
AI response:
{{ "category":"offtopic","location":"N/A" }}

Example:
Tweet: 
Hurricane Marina has made landfall near Miami with winds up to 120 mph, causing widespread damage.
AI response:
{{ "category":"hurricane","location":"N/A" }}


Example:
Tweet: 
The sudden explosion of colors in the sky during the festival was breathtaking.
AI response:
{{ "category":"offtopic" }}

Example:
Tweet: Firefighters are battling a large wildfire in California that has already consumed thousands of acres.
AI response:
{{ "category":"wildfire" ,"location":"USA,California,"}}

Example:
Tweet: Authorities are urging residents to evacuate due to an approaching bushfire in New South Wales.
AI response:
{{ "category":"wildfire","location":"Australia,New South Wales," }}


Example:
Tweet: 
A tornado warning has been issued for our county; everyone should seek shelter immediately
AI response:
{{ "category":"tornado","location":"N/A" }}

Example:
Tweet: 
An explosion at a chemical plant has forced evacuations within a five-mile radius of the site
AI response:
{{ "category":"explosion", "location":"N/A" }}


Example:
Tweet: 
News update: A bombing in the city center has resulted in several injuries, and emergency services are on the scene.
AI response:
{{ "category":"bombing" ,"location":"N/A"}}

Example:
Tweet: 
Unions claim job losses in Queensland public sector have hindered flood response.
AI response:
{{ "category":"flood" ,"location":"Australia,Queensland" }}



Tweet:
{input}
AI response:

"""
questions =[
    "Emergency crews are responding to severe flooding in the downtown area after last night's heavy rains",
    "The new policy has truly shaken the foundations of our approach to healthcare.",
    "Her latest novel is a hurricane of emotions, sweeping you off your feet.",
    "His thoughts were a tornado, twisting through ideas at breakneck speed.",
    "The surprise quiz bombed, with half the class scoring below average.",
    "Enjoying the peaceful scenery at the park, watching squirrels chase each other.",
   "Emergency crews are responding to severe flooding in the downtown area after last night's heavy rains",
"An explosion at a chemical plant has forced evacuations within a five-mile radius of the site.",
  "The surprise quiz bombed, with half the class scoring below average.",
"Huge fire in bc woods near vancouver",
 "Her latest novel is a hurricane of emotions, sweeping you off your feet.",
"Her latest novel is spreading like a wildfire across book clubs, igniting discussions everywhere.",
"The latest political scandal has sent shockwaves through the government, shaking foundations",
"The hurricane of support for the new charity swept through our community.",
"His latest comedy act bombed at the local theater, leaving audiences in silence.",
"A tornado warning has been issued for our county; everyone should seek shelter immediately.",
"I spent the day learning about the history of medieval castles and their lasting impact.",
"I'm drowning in work this week, it feels like a flood of assignments.",
"The sudden explosion of colors in the sky during the festival was breathtaking.",
"After the breakup, her emotions were a bushfire, wild and untamed.",
"The tech startup's growth was a wildfire, burning through initial capital at an alarming rate."
]
#phi3:3.8b-mini-instruct-4k-fp16
def classify(tweet):
    try:
        payload = {
        "model": "llama3:instruct",
        "prompt":system_prompt3.format(input=tweet),
        "format":"json",
        "stream": False
        }
        res=requests.post('http://localhost:11434/api/generate',json=payload)
        # print('res:',res)
        # print('-----------------------------------------')
        # print('Q:',tweet)
        answer= res.json()['response']
        # print(answer)
        return json.loads(answer)
    except Exception as e:
        print('Exception happende:',e, 'response:',response)
        raise e
data_path = os.path.join(os.path.dirname(__file__),'clean.csv')

print('data path:',data_path)
df = pd.read_csv(data_path).sample(frac=1)
print('data loaded from file.')
Size = df.shape[0]
index=0
for i,row in df[:10].iterrows():
    index+=1
    origical_category=row['category']
    try:
       tweet =row['tweet']
       response =classify(tweet)
       class_pred=response['category']
    except Exception as e:
       class_pred='EXCEPTION' 
    
    df.at[i,'llm_category']= class_pred
    print()
    print(f'{index*100/Size:0.2f}-----------------------------------------')
    if origical_category=='floods':
        origical_category='flood'
    result = 'MATCH' if class_pred.strip() == origical_category.strip() else 'MISMATCH'
    print(result )
    print('tweet:',row['tweet'])
    print('Original class:',origical_category)
    print(i,"prediction:",class_pred)
    print('location:',response['location'])
    print('-----------------------------------------------')
df.to_csv('clean_with_pred.csv')
   



# for question in questions:
#     payload = {
#     "model": "phi3:3.8b-mini-instruct-4k-fp16",
#     "prompt":system_prompt3.format(input=question),
#     "format":"json",
#     "stream": False
#     }
#     res=requests.post('http://localhost:11434/api/generate',json=payload)
#     print('-----------------------------------------')
#     print('Q:',question)
#     answer= res.json()['response']
#     print(answer)

