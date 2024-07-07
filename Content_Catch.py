import requests
import re
from bs4 import BeautifulSoup
import sqlite3
import NSFWJS_API as NSFWJS

conn = sqlite3.connect('posts.db')
                                

# 创建 posts 表
conn.execute('''
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    created_at TEXT NOT NULL,
    content TEXT,
    image_review BLOB,
    review TEXT
);
''')
conn.commit()

# API URL用于获取最新帖子
url = 'https://discourse.know-cnu.wiki/posts.json'

# API密钥和用户名
api_token = '91a46c71d6497e17a026a0063245bb9f20f4cbdf58f3ea8efae73a505189b96e'
headers = {
    'Api-Key': api_token,
    'Api-Username': "sansan",
}

base_url_1 = 'https://discourse.know-cnu.wiki/uploads/default/original/1X/'
base_url_2 = 'https://discourse-source.cn-sy1.rains3.com/original/1X/'

params = {
    'before': 100  # 获取ID小于100的帖子
}

response = requests.get(url, headers=headers, params=params)

posts = response.json()

img_review = True  # 默认为True，假设没有不适当的图片
for post in posts['latest_posts']:
    content = BeautifulSoup(post['cooked'], 'html.parser').get_text()
    username = post['username']
    created_at = post['created_at']
    # 尝试匹配两种正则表达式
    img_patterns = re.findall(r'href="/uploads/default/original/1X/(.*?)"|href="//discourse-source.cn-sy1.rains3.com/original/1X/(.*?)"', post['cooked'])
    if img_patterns:
        for img_pattern in img_patterns:
            # 检查匹配的是哪种模式，并选择相应的base_url
            if img_pattern[0]:  # 如果第一种模式匹配
                img_url = base_url_1 + img_pattern[0]
            else:  # 如果第二种模式匹配
                img_url = base_url_2 + img_pattern[1]
            print(f"正在检查图片：{img_url}")
            result = NSFWJS.classify_image(img_url)
            print(f"图片 {img_url} 的分类结果：", result)
            if not NSFWJS.classify_image(img_url):  # 如果有任何一张图片被判定为不适当
                img_review = False
                break  # 退出循环
    else:
        img_review = True
    
    # 在循环内部打开游标
    cursor = conn.cursor()
    
    # 使用 created_at 作为唯一标识符来检查记录是否已存在
    cursor.execute('SELECT * FROM posts WHERE created_at = ?', (created_at,))
    existing_record = cursor.fetchone()
    
    # 如果不存在相同的记录，则插入新记录
    if not existing_record:
        cursor.execute('INSERT INTO posts (username, created_at, content, image_review, review) VALUES (?, ?, ?, ?, ?)',
                       (username, created_at, content, img_review, None))
        conn.commit()
    
    # 在每次循环结束时关闭游标
    cursor.close()

# 在所有操作完成后关闭数据库连接
conn.close()