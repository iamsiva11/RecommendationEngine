import numpy as np

from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

data = fetch_movielens(min_rating=5.0)

#print repr(data)
print repr(data['train'])
print repr(data['test'])

"""
#Model Building
"""
#WARP (Weighted Approximate-Rank Pairwise) model
model= LightFM(loss='warp')
#%time model.fit(data['train'], epochs=30, num_threads=2)
model.fit(data['train'], epochs=30 , num_threads=2)

#Checking Precision
print("Train precision: %.2f" % precision_at_k(model, data['train'], k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, data['test'], k=5).mean())
#Unsurprisingly, the model fits the train set better than the test set.

"""
Making predictions
"""

def sample_recommendation(model, data, user_ids):
    n_users,n_items= data['train'].shape
    
    for user_id in  user_ids:
        known_positives= data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]
        
        print "User %s" % user_id
        print "known Positives"        
        for x in known_positives[:3]:
            print "%s" % x
        
        print "Recommended"        
        for x in  top_items[:3]:
            print "%s" % x            
        print " "        
            
sample_recommendation(model, data, [3,34,455])