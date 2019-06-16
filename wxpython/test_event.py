# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import wx

class MyFrame(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title='Log in', size=(400, 300))
        panel = wx.Panel(self)
        self.title = wx.StaticText(panel, label='Please enter username and password')
        self.label_usr = wx.StaticText(panel, label='Username: ')
        self.text_usr = wx.TextCtrl(panel, style=wx.TE_LEFT)
        self.label_pwd = wx.StaticText(panel, label='Password: ')
        self.text_pwd = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
        self.bt_confirm = wx.Button(panel, label='Confirm')
        self.bt_cancel = wx.Button(panel, label='Cancel')

        hsizer_usr = wx.BoxSizer(wx.HORIZONTAL)
        hsizer_usr.Add(self.label_usr, proportion=0, flag=wx.ALL, border=5)
        hsizer_usr.Add(self.text_usr, proportion=1, flag=wx.ALL, border=5)
        hsizer_pwd = wx.BoxSizer(wx.HORIZONTAL)
        hsizer_pwd.Add(self.label_pwd, proportion=0, flag=wx.ALL, border=5)
        hsizer_pwd.Add(self.text_pwd, proportion=1, flag=wx.ALL, border=5)
        hsizer_but = wx.BoxSizer(wx.HORIZONTAL)
        hsizer_but.Add(self.bt_confirm, proportion=0, flag=wx.ALIGN_CENTRE, border=5)
        hsizer_but.Add(self.bt_cancel, proportion=0, flag=wx.ALIGN_CENTRE, border=5)

        vsizer_all = wx.BoxSizer(wx.VERTICAL)
        vsizer_all.Add(self.title, proportion=0, flag=wx.BOTTOM|wx.TOP|wx.ALIGN_CENTRE, border=15)
        vsizer_all.Add(hsizer_usr, proportion=0, flag=wx.EXPAND|wx.LEFT|wx.RIGHT, border=45)
        vsizer_all.Add(hsizer_pwd, proportion=0, flag=wx.EXPAND|wx.LEFT|wx.RIGHT, border=45)
        vsizer_all.Add(hsizer_but, proportion=0, flag=wx.ALIGN_CENTRE, border=15)
        panel.SetSizer(vsizer_all)

        self.bt_confirm.Bind(wx.EVT_BUTTON, self.OnlickSubmit)
        self.bt_cancel.Bind(wx.EVT_BUTTON, self.OnlickCancel)

    def OnlickSubmit(self, event):
        username = self.text_usr.GetValue()
        password = self.text_pwd.GetValue()
        if username == str() or password == str():
            msg = 'The username or password cannot be empty'
        elif username == 'TiN' and password == '123456':
            msg = 'Login successful'
        else:
            msg = 'The username or password do not match'
        wx.MessageBox(msg)

    def OnlickCancel(self, event):
        self.text_usr.SetValue(str())
        self.text_pwd.SetValue(str())

def main():
    app = wx.App()
    frame = MyFrame(parent=None, id=-1)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
