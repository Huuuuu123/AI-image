import requests
from bs4 import BeautifulSoup

with open('stock_news.txt', 'w', encoding='utf-8') as f:
    a = 1
    
    for i in range(1, 21):
        url = 'https://wap.stockstar.com/list/' + str(i)
        
        response = requests.get(url)
        # 设置response.encoding为网页编码，防止出现中文乱码
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        title_selectors = soup.select('  div > div > div.left_text > a')
        
        # 遍历每一个标题元素，提取标题文本并打印输出
        for selector in title_selectors:
            title = selector.get_text()
            print('{}. {},{}'.format(a, title, "王瑞"))
            
            f.write('{}. {},{}'.format(a, title, "王瑞") + '\n')
            a += 1
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
with open('stock_news.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    words = jieba.cut(text)

# 统计词频
freq = {}
for word in words:
    if len(word) > 1:  # 只统计长度大于1的词语
        
        freq[word] = freq.get(word, 0) + 1
mask = plt.imread('china.jpg')
# 生成词云图
wc = WordCloud(mask=mask,font_path='msyh',background_color='white', width=800, height=600,mode ="RGB")
wc.generate_from_frequencies(freq)
plt.axis('off')
plt.imshow(wc)
wc.to_file('证券之星词云（王瑞）.png')