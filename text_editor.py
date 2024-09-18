from tkinter import *
from tkinter import filedialog, messagebox


filename=None

def newFile():
    global filename
    filename="Untitled"
    text.delete(0.0,END)

def saveFile():
    global filename
    t=text.get(0.0,END)
    f=open(filename, 'w')
    f.write(t)
    f.close()

def saveas():
    f=filedialog.asksaveasfile(mode='w', defaultextension='.text')
    t=text.get(0.0,END)
    try:
        f.write(t.rstrip)
    except:
        messagebox.showerror(title="Oops", message="Unable to save the file.")
def openFile():
    f=filedialog.askopenfile(mode='r')
    t=f.read()
    text.delete(0.0,END)
    text.insert(0.0,t)

root=Tk()
root.title("my python text editor")
root.minsize(width=400,height=400)


text=Text(root,width=400,height=400)
text.pack()

menubar=Menu(root)
filemenu=Menu(menubar)
filemenu.add_command(label="New", command=newFile)
filemenu.add_command(label="Open", command=openFile)
filemenu.add_command(label="Save", command=saveFile)
filemenu.add_command(label="Save As...", command=saveas)
filemenu.add_separator()
filemenu.add_command(label="Quit", command=root.quit)
menubar.add_cascade(label="Menu", menu=filemenu)

root.config(menu=menubar)
root.mainloop()