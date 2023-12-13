from tkinter import *
from chat import get_response

BG_BOTTOM_LABEL = "#240E16"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
HEAD_LABEL_COLOR = "#27405D"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


class ChatApp:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def _setup_main_window(self):
        self.window.title("ChatBot")
        self.window.resizable(width=True, height=True)
        self.window.configure(width=600, height=800, bg=BG_COLOR)

        head_label = Label(self.window, bg=HEAD_LABEL_COLOR, fg=TEXT_COLOR, text="Hello human!",
                           font=FONT_BOLD, pady=20)
        head_label.place(relwidth=1)

        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        bottom_label = Label(self.window, bg=BG_BOTTOM_LABEL, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self.on_enter_pressed)

        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=HEAD_LABEL_COLOR,
                             command=lambda: self.on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self.insert_message(msg, "USER")

    def insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"BOT: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ChatApp()
    app.run()
