# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import wx
import this

class App(wx.App):
    def OnInit(self):
        frame = wx.Frame(parent=None, title='Hello wxPython')
        frame.Show()
        return True

class MyFrame(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title='My Frame', pos=(100, 100), size=(300, 300))

if __name__ == '__main__':
    '''
    app = App()
    app.MainLoop()
    '''
    app = wx.App()
    frame = MyFrame(parent=None, id=-1)
    frame.Show()
    app.MainLoop()
