import numpy as np
from tkinter import *
from tkinter import filedialog
import tkinter as tk
from tkinter import Tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb

import tkinter.ttk as ttk
from converterdnafile import converter

import os
import logging
from progress.bar import Bar


class App():
     def __init__(self, master):
       
        self.master = master
        self.use_save_csv=BooleanVar(value=False)
        self.use_save_xlsx=BooleanVar(value=False)
        self.fileme=StringVar(value='dnafile.xlsx')
        self.master.title(u"Конвертор файлов")
        Label(self.master, text="1:").grid(row=2, column=0)

        # CREATE A TEXTBOX
        self.filename1 = Entry(self.master, width=64)
        self.filename1.focus_set()
        self.filename1.grid(row=2, column=1)

        # CREATE A BUTTON WITH "ASK TO OPEN A FILE"
        open_file_a = Button(self.master, text=u"Выбор файла 1",
                             command=self.browse_file_1)
        open_file_a.grid(row=2, column=2)

        Label(self.master, text=u"2:").grid(row=3, column=0)

        # CREATE B TEXTBOX
        self.filename2 = Entry(self.master, width=64)
        self.filename2.grid(row=3, column=1)

        # CREATE B BUTTON WITH "ASK TO OPEN B FILE"
        open_file_b = Button(self.master, text=u"Выбор файла 2",
                             command=self.browse_file_2)
        open_file_b.grid(row=3, column=2)

        Label(self.master, text=u"2:").grid(row=3, column=0)

        # CREATE C TEXTBOX
        self.filename3 = Entry(self.master, width=64)
        self.filename3.grid(row=4, column=1)

        # CREATE B BUTTON WITH "ASK TO OPEN C FILE"
        open_file_с = Button(self.master, text=u"Выбор файла 3",
                             command=self.browse_file_3)
        open_file_с.grid(row=4, column=2)


        Label(self.master, text=u"3:").grid(row=4, column=0)
        chk1 = Checkbutton(self.master, text=u"Сохранить в csv",
                            var=self.use_save_csv)
        chk1.grid(row=10, column=3, sticky=W)

        chk2 = Checkbutton(self.master, text=u"Сохранить в xlsx",
                            var=self.use_save_xlsx)
        chk2.grid(row=10, column=4, sticky=W)
    

        self.submit = Button(self.master, text = 'Выход', width = 12 , command = self.close_window,bg="red", fg="black")
        self.submit.grid(row=19, column=4)
        self.submit = Button(self.master, text="Конвертировать",
                             command=self.start_processing, width=12, height=3,bg="red", fg="black")
    
def start_processing(self):
    converter(
        file1=self.filename1.get(),  
        file2=self.filename2.get(),
        file3=self.filename3.get(),
        )





def close_window(self):
    print('Вышли из программы')
    self.master.destroy()

def browse_file_1(self):

    f = self.filename1.get()
    if not f:
        f = __file__

    filename = filedialog.askopenfilename(
        title=u"Выбор первого файла",
        filetypes=(("bcl", "*.bcl"), ("all files", "*.*")),
        initialdir=os.path.dirname(f)
        )
    self.filename1.delete(0, END)
    self.filename1.insert(0, filename)

def browse_file_2(self):
    f = self.filename2.get()
    if not f:
        f = __file__

    filename = filedialog.askopenfilename(
        title=u"Выбор второго файла",
        filetypes=(("locs", "*.locs") , ("all files", "*.*")),
        initialdir=os.path.dirname(f)
    )
    self.filename2.delete(0, END)
    self.filename2.insert(0, filename)

def browse_file_3(self):
    f = self.filename3.get()
    if not f:
        f = __file__

    filename = filedialog.askopenfilename(
        title=u"Выбор третьего файла",
        filetypes=(("cif", "*.cif"), ("all files", "*.*")),
        initialdir=os.path.dirname(f)
        )
    self.filename3.delete(0, END)
    self.filename3.insert(0, filename)




def main(): 
    root = tk.Tk()
    app = App(root)
    try:
        current_directory = fd.askdirectory()
        file_name = "car.ico"
        file_path = os.path.join(current_directory,file_name)
        root.iconbitmap(file_path)
    except:
        pass

    root.mainloop()

if __name__ == '__main__':
    main()


    

    