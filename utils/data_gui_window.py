import tkinter as tk
from tkinter import ttk

from threading import Thread

# GUI class that runs in its own thread
class DataWindow(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.precision = 2
        self.latest_data = None
        self.start()

    def run(self):
        self.root = tk.Tk()
        self.root.title("Stream Info")

        # Precision control
        self.precision_var = tk.IntVar(value=self.precision)
        tk.Label(self.root, text="Precision:").grid(row=0, column=0, sticky='w')
        tk.Spinbox(self.root, from_=0, to=10, textvariable=self.precision_var, width=5).grid(row=0, column=1, sticky='w')

        # Labels for variables
        self.labels = {}
        fields = [
            "timestamp", "focal_length", "principal_point", "exposure_time", "exposure_compensation",
            "lens_position", "focus_state", "iso_speed", "white_balance", "iso_gains",
            "white_balance_gains", "resolution"
        ]
        for i, key in enumerate(fields, start=1):
            tk.Label(self.root, text=f"{key}:").grid(row=i, column=0, sticky='w')
            label = tk.Label(self.root, text="")
            label.grid(row=i, column=1, sticky='w')
            self.labels[key] = label

        # Pose matrix
        tk.Label(self.root, text="Pose:").grid(row=1, column=2, sticky='w')
        self.pose_labels = []
        for i in range(4):
            row = []
            for j in range(4):
                lbl = tk.Label(self.root, text="", width=10, anchor='e')
                lbl.grid(row=2 + i, column=2 + j, padx=2)
                row.append(lbl)
            self.pose_labels.append(row)

        self.root.after(100, self._update_gui)
        self.root.mainloop()

    def update_data(self, data):
        self.latest_data = data

    def _update_gui(self):
        data = self.latest_data
        if data:
            p = self.precision_var.get()

            def fmt(v):
                try:
                    return f"{float(v):.{p}f}"
                except:
                    return str(v)

            self.labels["timestamp"].config(text=str(data.timestamp))
            self.labels["focal_length"].config(text=str(tuple(fmt(x) for x in data.payload.focal_length)))
            self.labels["principal_point"].config(text=str(tuple(fmt(x) for x in data.payload.principal_point)))
            self.labels["exposure_time"].config(text=fmt(data.payload.exposure_time))
            self.labels["exposure_compensation"].config(text=str(tuple(fmt(x) for x in data.payload.exposure_compensation)))
            self.labels["lens_position"].config(text=fmt(data.payload.lens_position))
            self.labels["focus_state"].config(text=str(data.payload.focus_state))
            self.labels["iso_speed"].config(text=str(data.payload.iso_speed))
            self.labels["white_balance"].config(text=str(data.payload.white_balance))
            self.labels["iso_gains"].config(text=str(tuple(fmt(x) for x in data.payload.iso_gains)))
            self.labels["white_balance_gains"].config(text=str(tuple(fmt(x) for x in data.payload.white_balance_gains)))
            self.labels["resolution"].config(text=f"{data.payload.resolution[0]}x{data.payload.resolution[1]}")

            for i in range(4):
                for j in range(4):
                    self.pose_labels[i][j].config(text=fmt(data.pose[i, j]))

        self.root.after(100, self._update_gui)  # keep updating

