import requests
from lxml import html
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

# (optional) Disable FutureWarning of Scikit-learn
from warnings import simplefilter


main_subject = 'Machine learning'

url = 'https://en.wikipedia.org/w/api.php'
params = {
        'action': 'query',
        'format': 'json',
        'generator':'links',
        'titles': main_subject,
        'prop':'pageprops',
        'ppprop':'wikibase_item',
        'gpllimit':1000,
        'redirects':1
        }

r = requests.get(url, params=params)
r_json = r.json()
linked_pages = r_json['query']['pages']

page_titles = [p['title'] for p in linked_pages.values()]


# select first X articles
num_articles = 200
pages = page_titles[:num_articles]

# make sure to keep the main subject on the list
pages += [main_subject]

# make sure there are no duplicates on the list
pages = list(set(pages))


text_db = []
for page in tqdm(pages):
    response = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={
                'action': 'parse',
                'page': page,
                'format': 'json',
                'prop': 'text',
                'redirects': ''
            }
        ).json()

    raw_html = response['parse']['text']['*']
    document = html.document_fromstring(raw_html)
    text = ''
    for p in document.xpath('//p'):
        text += p.text_content()
    text_db.append(text)
print('Done')





# Create a list of English stopwords
stop_words = stopwords.words('english')

# Instantiate the class
vec = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=(2,2), # bigrams
    use_idf=True
    )

# Train the model and transform the data
tf_idf =  vec.fit_transform(text_db)

# Create a pandas DataFrame
df = pd.DataFrame(
    tf_idf.toarray(),
    columns=vec.get_feature_names(),
    index=pages
    )

# Show the first lines of the DataFrame
df.head()

idf_df = pd.DataFrame(
    vec.idf_,
    index=vec.get_feature_names(),
    columns=['idf_weigths']
)

idf_df.sort_values(by=['idf_weigths']).head(10)

simplefilter(action='ignore', category=FutureWarning)

# select number of topic clusters
n_topics = 25

# Create an NMF instance
nmf = NMF(n_components=n_topics)

# Fit the model to the tf_idf
nmf_features = nmf.fit_transform(tf_idf)

# normalize the features
norm_features = normalize(nmf_features)


# Create clustered dataframe the NMF clustered df
components = pd.DataFrame(
    nmf.components_,
    columns=[df.columns]
    )

clusters = {}

# Show top 25 queries for each cluster
for i in range(len(components)):
    clusters[i] = []
    loop = dict(components.loc[i,:].nlargest(25)).items()
    for k,v in loop:
        clusters[i].append({'q':k[0],'sim_score': v})

# Create dataframe using the clustered dictionary
grouping = pd.DataFrame(clusters).T
grouping['topic'] = grouping[0].apply(lambda x: x['q'])
grouping.drop(0, axis=1, inplace=True)
grouping.set_index('topic', inplace=True)


def show_queries(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x['q'])
    return df


# Only display the query in the dataframe
clustered_queries = show_queries(grouping)

# compute cosine similarities of each cluster
data = {}
# create dataframe
norm_df = pd.DataFrame(norm_features, index=pages)
for page in pages:
    # select page recommendations
    recommendations = norm_df.loc[page,:]

    # Compute cosine similarity
    similarities = norm_df.dot(recommendations)

    data[page] = []
    loop = dict(similarities.nlargest(20)).items()
    for k, v in loop:
        if k != page:
            data[page].append({'q':k,'sim_score': v})

# convert dictionary to dataframe
recommender = pd.DataFrame(data).T

def show_queries(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x['q'])
    return df

show_queries(recommender).head()