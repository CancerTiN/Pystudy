# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import os
import sqlite3

os.remove('mrsoft.db')
conn = sqlite3.connect('mrsoft.db')

cursor = conn.cursor()
cursor.execute('create table user (id int(10) primary key, name varchar (20))')

cursor.execute('insert into user (id, name) values ("1", "MRSOFT")')
cursor.execute('insert into user (id, name) values ("2", "Andy")')
cursor.execute('insert into user (id, name) values ("3", "Little Helper")')

cursor.execute('select * from user')
result1 = cursor.fetchone()
print(result1)

cursor.execute('select * from user')
result2 = cursor.fetchmany(2)
print(result2)

cursor.execute('select * from user')
result3 = cursor.fetchall()
print(result3)

cursor.execute('select * from user where id > ?', (1,))
result4 = cursor.fetchall()
print(result4)

cursor.execute('update user set name = ? where id = ?', ('MR', 1))
cursor.execute('select * from user')
result5 = cursor.fetchall()
print(result5)

cursor.execute('delete from user where id = ?', (1,))
cursor.execute('select * from user')
result6 = cursor.fetchall()
print(result6)

cursor.close()
conn.close()
