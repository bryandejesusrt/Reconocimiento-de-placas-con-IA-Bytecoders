import tkinter as tk
from tkinter import ttk
import pandas as pd

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detection Results")

        # Predetermined file path
        self.file_path = './test.csv'

        
        self.file_label = tk.Label(self.root, text=self.file_path)
        self.file_label.pack(pady=10)

        
        self.tree = ttk.Treeview(self.root, columns=("Frame", "Car ID", "Car BBox", "Plate BBox", "Plate BBox Score", "Plate Number", "Plate Number Score"), show="headings")
        self.tree.pack(pady=20)

        # Define the column headings
        self.tree.heading("Frame", text="Frame")
        self.tree.heading("Car ID", text="Car ID")
        self.tree.heading("Car BBox", text="Car BBox")
        self.tree.heading("Plate BBox", text="Plate BBox")
        self.tree.heading("Plate BBox Score", text="Plate BBox Score")
        self.tree.heading("Plate Number", text="Plate Number")
        self.tree.heading("Plate Number Score", text="Plate Number Score")

        # Define column widths
        self.tree.column("Frame", width=80)
        self.tree.column("Car ID", width=80)
        self.tree.column("Car BBox", width=200)
        self.tree.column("Plate BBox", width=200)
        self.tree.column("Plate BBox Score", width=150)
        self.tree.column("Plate Number", width=150)
        self.tree.column("Plate Number Score", width=150)

        # Load the predetermined CSV file
        self.load_csv(self.file_path)

    def load_csv(self, file_path):
        df = pd.read_csv(file_path)
        self.display_data(df)

    def display_data(self, df):
        # Clear the current data in the Treeview
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Insert new data into the Treeview
        for index, row in df.iterrows():
            self.tree.insert("", "end", values=(
                row["frame_nmr"], row["car_id"], row["car_bbox"], row["license_plate_bbox"],
                row["license_plate_bbox_score"], row["license_number"], row["license_number_score"]
            ))

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
