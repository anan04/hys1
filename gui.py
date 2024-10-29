import tkinter as tk  
from tkinter import messagebox  
from tkinter import ttk  
from com08 import *   
def on_submit():  
    # 获取选择选项  
    selected_option = combo_box.get()  
    # 获取输入文本  
    input_text = entry.get()  
      
    # 显示选择结果  
    messagebox.showinfo("提交结果", f"你选择的选项是: {selected_option}\n你输入的文本是: {input_text}")  
  
# 创建主窗口  
root = tk.Tk()  
root.title("多模态星载LiDAR估测林下地形精度评估")  
  
# 设置窗口大小  
root.geometry("400x600")  
  
# 创建标签和输入框  
label = tk.Label(root, text="选择星载LiDAR数据:")  
label.pack(pady=50)  
  
# 创建选项列表  
options = ["ICESat-2", "GEDI"]  
  
# 创建组合框（下拉菜单）  
combo_box = ttk.Combobox(root, values=options)  
combo_box.pack(pady=20)  
combo_box.current(0)  # 设置默认选项为第一个  
  
# 创建标签和输入框  
label_input1 = tk.Label(root, text="输入星载LiDAR数据的路径:")  
label_input1.pack(pady=5)  
entry = tk.Entry(root, width=30) 
entry.pack(pady=10) 

label_input2 = tk.Label(root, text="输入验证数据的路径:")  
label_input2.pack(pady=5)   
entry = tk.Entry(root, width=30)  
entry.pack(pady=10)  

label_input3 = tk.Label(root, text="输入储存数据的路径:")  
label_input3.pack(pady=5)   
entry = tk.Entry(root, width=30)  
entry.pack(pady=10)  
  
# 创建确定按钮  
submit_button = tk.Button(root, text="精度验证", command=on_submit)  
submit_button.pack(pady=20)  
  
# 运行主循环  
root.mainloop()