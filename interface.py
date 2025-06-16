import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Animation function for fading text
def fade_in(widget, text, fg_target, delay=30):
    widget.config(text="")
    steps = 10
    for i in range(steps + 1):
        def step_opacity(i=i):
            color = blend_colors("#121212", fg_target, i / steps)
            widget.config(fg=color, text=text)
        root.after(i * delay, step_opacity)

# Blend color (simple interpolation)
def blend_colors(bg, fg, factor):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return "#%02x%02x%02x" % rgb

    bg_rgb = hex_to_rgb(bg)
    fg_rgb = hex_to_rgb(fg)

    blended = tuple(int(bg + (fg - bg) * factor) for bg, fg in zip(bg_rgb, fg_rgb))
    return rgb_to_hex(blended)

# Prediction function
def predict_news():
    text = entry.get("1.0", tk.END).strip()
    if len(text.split()) < 3:
        messagebox.showwarning("⚠️ Wait", "Please enter more than 3 words for meaningful results.")
        return
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    if prediction[0] == "REAL":
        fade_in(result_label, "✅ This news appears REAL.", "#00FFAA")
    else:
        fade_in(result_label, "❌ This news is likely FAKE.", "#FF5555")

# GUI Setup
root = tk.Tk()
root.title("📰 DeFake Fake News Check")
root.geometry("700x500")
root.configure(bg="#121212")

FONT_HEADER = ("Helvetica Neue", 20, "bold")
FONT_LABEL = ("Helvetica Neue", 12)
FONT_RESULT = ("Helvetica Neue", 14, "bold")
FONT_INPUT = ("Helvetica Neue", 11)

header = tk.Label(root, text="🧠 Fake News Classifier", font=FONT_HEADER, bg="#121212", fg="#ffffff")
header.pack(pady=20)

instruction = tk.Label(root, text="Paste news content below to analyze:", font=FONT_LABEL, bg="#121212", fg="#cccccc")
instruction.pack()

frame = tk.Frame(root, bg="#1e1e1e", bd=1)
frame.pack(pady=15, padx=20)

entry = tk.Text(frame, height=8, width=75, wrap="word", font=FONT_INPUT, bg="#1e1e1e", fg="#ffffff", insertbackground="white", relief="flat")
entry.pack(padx=10, pady=10)
entry.insert(tk.END, "Tesla launches humanoid robot that can cook and clean.")

check_button = tk.Button(root, text="🧪 Analyze", command=predict_news, font=FONT_LABEL,
                         bg="#0f62fe", fg="white", activebackground="#0051cc",
                         padx=20, pady=8, relief="flat", cursor="hand2")
check_button.pack(pady=10)

result_label = tk.Label(root, text="", font=FONT_RESULT, bg="#121212", fg="#00FFAA")
result_label.pack(pady=15)

footer = tk.Label(root, text="Made by Team DeFake 🖤", font=("Helvetica Neue", 10), bg="#121212", fg="#444444")
footer.pack(side="bottom", pady=20)

root.mainloop()
