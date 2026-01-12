import customtkinter as ctk
import tkinter as tk
import numpy as np
import math

class HandwritingTester(ctk.CTkToplevel):
    def __init__(self, parent, model):
        super().__init__(parent)
        
        self.model = model
        self.title("✍️ Handwriting Tester")
        self.geometry("400x500")
        self.resizable(False, False)
        
        # Grid setup
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header = ctk.CTkLabel(self, text="Draw a Digit (0-9)", font=ctk.CTkFont(size=18, weight="bold"))
        self.header.grid(row=0, column=0, pady=10)
        
        # Canvas Frame
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.grid(row=1, column=0, padx=20, pady=10)
        
        # Canvas (280x280 for easier drawing, effectively 10x scale of 28x28)
        self.canvas_size = 280
        self.grid_size = 28
        self.scale = self.canvas_size // self.grid_size
        
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg='black', 
            highlightthickness=0
        )
        self.canvas.pack(padx=2, pady=2)
        
        # Internal grid representation (28x28)
        self.grid_data = np.zeros((self.grid_size, self.grid_size))
        
        # Bindings
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<Button-1>", self._paint)
        
        # Control Buttons
        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.btn_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.btn_clear = ctk.CTkButton(
            self.btn_frame, 
            text="Clear", 
            fg_color="#E74C3C", 
            hover_color="#C0392B",
            command=self._clear
        )
        self.btn_clear.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_predict = ctk.CTkButton(
            self.btn_frame, 
            text="Predict", 
            fg_color="#27AE60", 
            hover_color="#229954",
            command=self._predict
        )
        self.btn_predict.grid(row=0, column=1, padx=5, pady=5)
        
        # Result Label
        self.result_label = ctk.CTkLabel(
            self, 
            text="Prediction: ?", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.result_label.grid(row=3, column=0, pady=20)
        
    def _paint(self, event):
        x, y = event.x, event.y
        r = 8  # Brush radius
        
        # Draw on canvas
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        
        # Update internal grid (simple splatting)
        # Map canvas coordinates to grid coordinates
        # We need to simulate the 'brush' on the 28x28 grid
        
        # Center in grid coords
        gx_center = x / self.scale
        gy_center = y / self.scale
        
        # Brush radius in grid coords
        gr = r / self.scale
        
        # Bounding box in grid
        g_min_x = max(0, int(gx_center - gr - 1))
        g_max_x = min(self.grid_size, int(gx_center + gr + 1))
        g_min_y = max(0, int(gy_center - gr - 1))
        g_max_y = min(self.grid_size, int(gy_center + gr + 1))
        
        for gy in range(g_min_y, g_max_y):
            for gx in range(g_min_x, g_max_x):
                # Distance from center
                dist = math.sqrt((gx + 0.5 - gx_center)**2 + (gy + 0.5 - gy_center)**2)
                if dist < gr:
                    # Strong activation to match MNIST
                    self.grid_data[gy, gx] = 1.0

    def _clear(self):
        self.canvas.delete("all")
        self.grid_data.fill(0)
        self.result_label.configure(text="Prediction: ?")
        
    def _predict(self):
        if self.model is None:
            self.result_label.configure(text="No Model!")
            return
        
        # Apply MNIST-style preprocessing
        img = self.grid_data.copy()
        
        # Normalize to [0, 1] range
        if img.max() > 0:
            img = img / img.max()
        
        # Center the image (like MNIST does)
        # Find bounding box of non-zero pixels
        rows = np.any(img > 0.1, axis=1)
        cols = np.any(img > 0.1, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Extract the digit
            digit = img[rmin:rmax+1, cmin:cmax+1]
            
            # Resize to 20x20 (MNIST uses 20x20 centered in 28x28)
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            target_size = 20
            zoom_y = target_size / digit.shape[0]
            zoom_x = target_size / digit.shape[1]
            zoom_factor = min(zoom_y, zoom_x)  # Keep aspect ratio
            
            # Resize
            digit_resized = zoom(digit, zoom_factor, order=1)
            
            # Create 28x28 canvas and center the digit
            img_centered = np.zeros((28, 28))
            y_offset = (28 - digit_resized.shape[0]) // 2
            x_offset = (28 - digit_resized.shape[1]) // 2
            
            img_centered[
                y_offset:y_offset+digit_resized.shape[0],
                x_offset:x_offset+digit_resized.shape[1]
            ] = digit_resized
            
            input_vector = img_centered.flatten()
        else:
            # Empty canvas
            input_vector = img.flatten()
        
        # Predict
        try:
            prediction = self.model.predict([input_vector])
            
            if isinstance(prediction, (list, np.ndarray)):
                pred_class = prediction[0]
            else:
                pred_class = prediction
                
            self.result_label.configure(text=f"Prediction: {pred_class}")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            self.result_label.configure(text="Error")
