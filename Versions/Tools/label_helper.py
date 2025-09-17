# Cardboard QC Labeling Tool for Workers
# Usage: python label_helper.py
import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import threading

class CardboardLabeler:
    def __init__(self):
        self.csv_path = Path("out/qc_labels.csv")
        self.overlay_dir = Path("out/overlays_overlay")
        self.current_index = 0
        self.rows = []
        self.labeled_count = 0
        
        self.setup_gui()
        self.load_data()
        self.show_current_image()
    
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Cardboard QC Labeling Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame, text="Loading...", font=('Arial', 12, 'bold'))
        self.progress_label.pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=400)
        self.progress_bar.pack(side=tk.RIGHT)
        
        # Image frame
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="Loading image...", font=('Arial', 14))
        self.image_label.pack(expand=True)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X)
        
        # Navigation buttons
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(nav_frame, text="← Previous", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next →", command=self.next_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Skip", command=self.skip_image).pack(side=tk.LEFT, padx=10)
        
        # Labeling buttons
        label_frame = ttk.Frame(controls_frame)
        label_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(label_frame, text="Quality:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        pass_btn = ttk.Button(label_frame, text="✓ PASS (Flat)", command=lambda: self.label_image("Pass"))
        pass_btn.pack(side=tk.LEFT, padx=2)
        pass_btn.configure(style='Pass.TButton')
        
        fail_btn = ttk.Button(label_frame, text="✗ FAIL (Warped)", command=lambda: self.label_image("Fail"))
        fail_btn.pack(side=tk.LEFT, padx=2)
        fail_btn.configure(style='Fail.TButton')
        
        # Save/Exit buttons
        action_frame = ttk.Frame(controls_frame)
        action_frame.pack(side=tk.RIGHT)
        
        ttk.Button(action_frame, text="💾 Save Progress", command=self.save_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🚪 Exit", command=self.exit_app).pack(side=tk.LEFT, padx=2)
        
        # Instructions
        instructions = ttk.Label(main_frame, 
            text="Instructions: Look at the cardboard bundle (green box). If it appears FLAT → PASS. If WARPED/BENT → FAIL.", 
            font=('Arial', 10), foreground='blue')
        instructions.pack(pady=5)
        
        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<p>', lambda e: self.label_image("Pass"))
        self.root.bind('<f>', lambda e: self.label_image("Fail"))
        self.root.bind('<s>', lambda e: self.skip_image())
        self.root.bind('<Control-s>', lambda e: self.save_data())
        
        # Setup button styles
        style = ttk.Style()
        style.configure('Pass.TButton', foreground='green')
        style.configure('Fail.TButton', foreground='red')
    
    def load_data(self):
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.rows = list(reader)
            
            # Count labeled images
            self.labeled_count = sum(1 for row in self.rows if row["label"].strip())
            
            # Find first unlabeled image
            for i, row in enumerate(self.rows):
                if not row["label"].strip():
                    self.current_index = i
                    break
            
            self.update_progress()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def show_current_image(self):
        if not self.rows:
            return
            
        row = self.rows[self.current_index]
        img_path = self.overlay_dir / row["file"]
        
        if not img_path.exists():
            self.image_label.configure(text=f"Image not found: {row['file']}")
            return
        
        try:
            # Load and resize image
            img = Image.open(img_path)
            img.thumbnail((800, 600), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
            # Update window title
            status = "✓ LABELED" if row["label"].strip() else "⏳ UNLABELED"
            self.root.title(f"Cardboard QC Labeling - {row['file']} ({status})")
            
        except Exception as e:
            self.image_label.configure(text=f"Error loading image: {e}")
    
    def update_progress(self):
        total = len(self.rows)
        self.progress_label.configure(text=f"Progress: {self.labeled_count}/{total} labeled ({self.current_index + 1}/{total} viewing)")
        self.progress_bar.configure(maximum=total, value=self.labeled_count)
    
    def label_image(self, label):
        if not self.rows:
            return
            
        row = self.rows[self.current_index]
        was_labeled = bool(row["label"].strip())
        
        row["label"] = label
        row["reason"] = "Bundle appears flat" if label == "Pass" else "Bundle appears warped"
        
        if not was_labeled:
            self.labeled_count += 1
        
        self.update_progress()
        self.next_image()
    
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
            self.update_progress()
    
    def next_image(self):
        if self.current_index < len(self.rows) - 1:
            self.current_index += 1
            self.show_current_image()
            self.update_progress()
    
    def skip_image(self):
        self.next_image()
    
    def save_data(self):
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["file", "label", "reason"])
                writer.writeheader()
                writer.writerows(self.rows)
            messagebox.showinfo("Saved", f"Progress saved! {self.labeled_count}/{len(self.rows)} images labeled.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
    
    def exit_app(self):
        if messagebox.askyesno("Exit", "Save progress before exiting?"):
            self.save_data()
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

def show_and_label():
    """Fallback simple labeling for systems without tkinter"""
    csv_path = Path("out/qc_labels.csv")
    overlay_dir = Path("out/overlays_overlay")
    
    # Read current CSV
    labeled_files = set()
    rows = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if row["label"].strip():
                labeled_files.add(row["file"])
    
    print(f"Found {len(labeled_files)} already labeled images")
    print(f"Total images to label: {len(rows)}")
    
    # Process unlabeled images
    for i, row in enumerate(rows):
        if row["file"] in labeled_files:
            continue
            
        img_path = overlay_dir / row["file"]
        if not img_path.exists():
            continue
            
        # Show image info instead of actual image
        print(f"\n{'='*50}")
        print(f"Image {i+1}/{len(rows)}: {row['file']}")
        print(f"Path: {img_path}")
        print("Please open the image manually to view it.")
        print("GREEN BOX shows the cardboard bundle detection")
        print("='*50}")
        
        # Get user input
        print("LABELING INSTRUCTIONS:")
        print("- If cardboard bundle appears FLAT → type 'p' (PASS)")  
        print("- If cardboard bundle appears WARPED → type 'f' (FAIL)")
        print("- To skip this image → type 's'")
        print("- To quit and save → type 'q'")
        
        while True:
            label_input = input("Your label (p/f/s/q): ").lower().strip()
            
            if label_input == 'q':
                save_csv(csv_path, rows)
                print("✅ Progress saved! Exiting...")
                return
            elif label_input == 's':
                print("⏭️ Skipped")
                break
            elif label_input in ['p', 'pass']:
                row["label"] = "Pass"
                row["reason"] = "Bundle appears flat"
                print("✅ Labeled as PASS")
                break
            elif label_input in ['f', 'fail']:
                row["label"] = "Fail" 
                row["reason"] = "Bundle appears warped"
                print("❌ Labeled as FAIL")
                break
            else:
                print("Invalid input. Please use p/f/s/q")
    
    save_csv(csv_path, rows)
    print("🎉 All images labeled!")

def save_csv(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "reason"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    try:
        # Try GUI first (better for workers)
        app = CardboardLabeler()
        app.run()
    except Exception as e:
        print(f"GUI not available ({e}), falling back to text interface...")
        show_and_label()