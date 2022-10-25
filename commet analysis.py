import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from PIL import Image
import numpy as np

file = "PsjBGqTjWu4.xlsx"
df = pd.read_excel(file, usecols=[1, 2, 3, 4, 5])
print(df.head(10))


v_cmt_list = df['text'].values.tolist()
print('length of v_cmt_list is:{}'.format(len(v_cmt_list)))


score_list = []
tag_list = []
for comment in v_cmt_list:
	tag = ''
	judge = TextBlob(comment)
	sentiments_score = judge.sentiment.polarity
	score_list.append(sentiments_score)
	if sentiments_score < 0:
		tag = 'negative'
	elif sentiments_score == 0:
		tag = 'neutral'
	else:
		tag = 'positive'
	tag_list.append(tag)
df['sentiment score'] = score_list
df['analysis result'] = tag_list
df.to_excel('Sentiment analysis results.xlsx', index=None)


print(df.head(10))


print(df.groupby(by=['analysis result']).count()['text'])


v_cmt_str = ".".join(v_cmt_list)

# Map Wordcloud
stopwords = ['the', 'a', 'and', 'of', 'it', 'her', 'she', 'if', 'I', 'is', 'not', 'your', 'there', 'this', 'how','just','got','I'                       ,
             'that', 'to', 'you', 'in', 'as', 'for', 'are', 'so', 'was', 'but', 'with', 'they','can', 'have','my','be']  # 停用词
coloring = np.array(Image.open("lzq.jpeg"))
backgroud_Image = coloring
wc = WordCloud(
	# scale=3,
	background_color="white",
	max_words=1000,
	font_path='/System/Library/Fonts/simhei.ttf',
	stopwords=stopwords,
	mask=backgroud_Image,
	color_func=ImageColorGenerator(coloring),
	max_font_size=100,
	random_state=240
)
wc.generate(v_cmt_str)
wc.to_file('wordcloud.png')

