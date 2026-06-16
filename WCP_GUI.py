import tkinter as tk

ws = tk.Tk()
ws.title("World Cup Simulator")
ws.geometry("400x400")
ws.resizable(False, False)

title_label = tk.Label(ws, text="World Cup Simulator", font=("Arial", 16))
title_label.pack(pady=20)

test_button = tk.Button(ws, text="Click me")
test_button.pack(pady=10)

ws.mainloop()