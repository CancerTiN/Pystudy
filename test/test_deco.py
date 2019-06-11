# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import datetime

def pkgsfuncdeco(func):
    def wrapper(*args, **kwargs):
        getime = lambda: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{}\tINFO: begin of the function ({}) at ({}) at'.format(
            getime(), func.__name__, func.__module__
        ))
        try:
            dct = dict()
            result = func(*args, **kwargs)
        except Exception as e:
            print('{}\tERROR: exception occurred when calling ({}) at ({})'.format(
                getime(), func.__name__, func.__module__
            ))
            print('{}\tDEBUG: args -> {}'.format(getime(), args))
            print('{}\tDEBUG: kwargs -> {}'.format(getime(), kwargs))
            raise Exception(e)
        print('{}\tINFO: final of the function ({}) at ({}) at'.format(
            getime(), func.__name__, func.__module__
        ))
        return result
    return wrapper


@pkgsfuncdeco
def main():
    print(type(dct))

if __name__ == '__main__':
    main()
