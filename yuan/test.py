import datetime

a = input("你是哪一年出生的，填一个四位数年份")
nowyear = datetime.datetime.now().year
age = nowyear - int(a)
print("您今年是：" + str(age) + "years old")
if age < 18:
    print("未成年")
if age >= 18:
    print("已成年")
