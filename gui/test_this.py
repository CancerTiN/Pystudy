# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import wx
import this

class MyFrame(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title='Creat StaticText Class', pos=(100, 100), size=(600, 600))
        panel = wx.Panel(self)
        title = wx.StaticText(panel, label='The Zen of Python, by Tim Peters', pos=(100, 20))
        font = wx.Font(16, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        title.SetFont(font)
        for n, line in enumerate(''.join(this.d.get(c, c) for c in this.s).split('\n')[2:]):
            wx.StaticText(panel, label=line, pos=(50, 50 + n * 20))

def main():
    app = wx.App()
    frame = MyFrame(parent=None, id=-1)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
