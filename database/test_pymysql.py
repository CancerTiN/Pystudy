# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import pymysql

db = pymysql.connect('localhost', 'root', 'root', 'mrsoft')
cursor = db.cursor()
cursor.execute('SELECT VERSION()')
data = cursor.fetchone()
print('Database version: {}'.format(data))

cursor.execute('DROP TABLE IF EXISTS books')
sql = '''
CREATE TABLE books (
  id int(8) NOT NULL AUTO_INCREMENT,
  name varchar(50) NOT NULL,
  category varchar(50) NOT NULL,
  price decimal(10,2) DEFAULT NULL,
  publish_time date DEFAULT NULL,
  PRIMARY KEY (id)
) ENGINE=MyISAM AUTO_INCREMENT=1 DEFAULT CHARSET=UTF8MB4;
'''
cursor.execute(sql)

data = [('零基础学Python', 'Python', '79.80', '2018-5-20'),
        ('Python从入门到精通', 'Python', '69.80', '2018-6-18'),
        ('零基础学PHP', 'PHP', '69.80', '2017-5-21'),
        ('PHP项目开发实战入门', 'PHP', '79.80', '2016-5-21'),
        ('零基础学Java', 'Java', '69.80', '2017-5-21')]

cursor.executemany('insert into books (name, category, price, publish_time) values (%s, %s, %s, %s)', data)
db.commit()

db.close()
