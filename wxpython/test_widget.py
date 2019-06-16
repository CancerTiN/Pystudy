# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import wx

class MyFrame(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title='Creat TextCtrl Class', size=(400, 300))
        panel = wx.Panel(self)
        self.title = wx.StaticText(panel, label='Please enter username and password', pos=(90, 40))
        self.label_usr = wx.StaticText(panel, label='Username: ', pos=(50, 100))
        self.text_usr = wx.TextCtrl(panel, pos=(120, 100), size=(200, 25), style=wx.TE_LEFT)
        self.label_pwd = wx.StaticText(panel, label='Password: ', pos=(50, 140))
        self.text_pwd = wx.TextCtrl(panel, pos=(120, 140), size=(200, 25), style=wx.TE_LEFT)
        self.bt_confirm = wx.Button(panel, label='Confirm', pos=(105, 180))
        self.bt_cancel = wx.Button(panel, label='Cancel', pos=(195, 180))

def main():
    app = wx.App()
    frame = MyFrame(parent=None, id=-1)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
