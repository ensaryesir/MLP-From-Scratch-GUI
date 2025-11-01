"""
Visualization panel with matplotlib plots.
Contains three tabs: Training (interactive), Test, and Error graph.
"""

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class VisualizationFrame(ctk.CTkFrame):
    """
    Tabbed visualization panel with matplotlib plots.
    Training tab allows interactive point addition via mouse clicks.
    """

    def __init__(self, master, on_point_added_callback=None, **kwargs):
        super().__init__(master, **kwargs)

        self.on_point_added_callback = on_point_added_callback
        self.loss_history = []

        # create tabbed view
        self.tabview = ctk.CTkTabview(self, width=700, height=600)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # add tabs
        self.tabview.add("🎯 Training")
        self.tabview.add("📊 Test")
        self.tabview.add("📈 Error Graph")

        # setup matplotlib figures for each tab
        self._setup_train_tab()
        self._setup_test_tab()
        self._setup_loss_tab()

    def _setup_train_tab(self):
        """Setup training tab with clickable matplotlib canvas."""
        tab = self.tabview.tab("🎯 Training")

        # create matplotlib figure
        self.train_fig = Figure(figsize=(7, 6), dpi=100)
        self.train_ax = self.train_fig.add_subplot(111)
        self.train_ax.set_xlim(-1, 11)
        self.train_ax.set_ylim(-1, 11)
        self.train_ax.set_xlabel('X')
        self.train_ax.set_ylabel('Y')
        self.train_ax.set_title('Training Data - Click to Add Points')
        self.train_ax.grid(True, alpha=0.3)

        # embed in tkinter
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, tab)
        self.train_canvas.draw()
        self.train_canvas.get_tk_widget().pack(fill="both", expand=True)

        # bind mouse click event
        self.train_canvas.mpl_connect('button_press_event', self._on_train_click)

    def _setup_test_tab(self):
        """Setup test tab with matplotlib plot."""
        tab = self.tabview.tab("📊 Test")

        # create matplotlib figure
        self.test_fig = Figure(figsize=(7, 6), dpi=100)
        self.test_ax = self.test_fig.add_subplot(111)
        self.test_ax.set_xlim(-1, 11)
        self.test_ax.set_ylim(-1, 11)
        self.test_ax.set_xlabel('X')
        self.test_ax.set_ylabel('Y')
        self.test_ax.set_title('Test Data and Model Performance')
        self.test_ax.grid(True, alpha=0.3)

        # embed in tkinter
        self.test_canvas = FigureCanvasTkAgg(self.test_fig, tab)
        self.test_canvas.draw()
        self.test_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _setup_loss_tab(self):
        """Setup loss tab for training curve."""
        tab = self.tabview.tab("📈 Error Graph")

        # create matplotlib figure
        self.loss_fig = Figure(figsize=(7, 6), dpi=100)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Error')
        self.loss_ax.set_title('Error During Training')
        self.loss_ax.grid(True, alpha=0.3)

        # embed in tkinter
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, tab)
        self.loss_canvas.draw()
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_train_click(self, event):
        """Handle clicks on training plot to add points."""
        # only handle clicks inside the plot
        if event.inaxes == self.train_ax and event.xdata and event.ydata:
            x, y = event.xdata, event.ydata
            if self.on_point_added_callback:
                self.on_point_added_callback(x, y)

    def plot_data_points(self, data_handler, ax=None):
        """Plot all data points on the given axes."""
        if ax is None:
            ax = self.train_ax

        # plot each class
        for class_id in range(data_handler.get_num_classes()):
            points = data_handler.get_data_by_class(class_id)
            if len(points) > 0:
                color = data_handler.get_color(class_id)
                # Extract x and y coordinates
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                ax.scatter(x_coords, y_coords,
                          c=color, s=100, alpha=0.7,
                          edgecolors='black', linewidth=1.5,
                          label=data_handler.classes[class_id])

        # add legend only if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right')

    def update_train_view(self, data_handler):
        """Refresh the training plot with current data."""
        self.train_ax.clear()
        self.train_ax.set_xlim(-1, 11)
        self.train_ax.set_ylim(-1, 11)
        self.train_ax.set_xlabel('X')
        self.train_ax.set_ylabel('Y')
        self.train_ax.set_title('Training Data - Click to Add Points')
        self.train_ax.grid(True, alpha=0.3)

        # draw points
        self.plot_data_points(data_handler, self.train_ax)

        self.train_canvas.draw()

    def clear_test_view(self):
        """Clear the test plot and reset it."""
        self.test_ax.clear()
        self.test_ax.set_xlim(-1, 11)
        self.test_ax.set_ylim(-1, 11)
        self.test_ax.set_xlabel('X')
        self.test_ax.set_ylabel('Y')
        self.test_ax.set_title('Test Data and Model Performance')
        self.test_ax.grid(True, alpha=0.3)
        self.test_canvas.draw()

    def update_decision_boundary(self, model, X, y, data_handler, tab_name='train'):
        """Draw decision boundary using trained model predictions on a meshgrid."""
        # select which plot to update
        if tab_name == 'train':
            ax = self.train_ax
            canvas = self.train_canvas
            title = 'Training Data - Decision Boundaries'
        else:
            ax = self.test_ax
            canvas = self.test_canvas
            title = 'Test Data - Model Performance'
        
        ax.clear()
        
        # create meshgrid manually
        x_min, x_max = -1, 11
        y_min, y_max = -1, 11
        h = 0.1  # Grid resolution
        
        # Create ranges manually
        x_range = []
        x_val = x_min
        while x_val < x_max:
            x_range.append(x_val)
            x_val += h
        
        y_range = []
        y_val = y_min
        while y_val < y_max:
            y_range.append(y_val)
            y_val += h
        
        # Create meshgrid manually
        xx = []
        yy = []
        for y_val in y_range:
            xx_row = []
            yy_row = []
            for x_val in x_range:
                xx_row.append(x_val)
                yy_row.append(y_val)
            xx.append(xx_row)
            yy.append(yy_row)
        
        # Flatten the meshgrid and create input for prediction
        grid_points = []
        for i in range(len(xx)):
            for j in range(len(xx[0])):
                grid_points.append([xx[i][j], yy[i][j]])
        
        # predict on grid
        Z = model.predict(grid_points)
        
        # Reshape Z back to grid shape
        Z_grid = []
        idx = 0
        for i in range(len(xx)):
            Z_row = []
            for j in range(len(xx[0])):
                Z_row.append(Z[idx])
                idx += 1
            Z_grid.append(Z_row)
        
        # draw decision regions with contourf
        n_classes = data_handler.get_num_classes()
        colors = [data_handler.get_color(i) for i in range(n_classes)]
        
        # Create levels for contourf
        levels = [i - 0.5 for i in range(n_classes + 1)]
        ax.contourf(xx, yy, Z_grid, alpha=0.3, levels=levels, colors=colors)
        
        # overlay data points
        if len(X) > 0:
            for class_id in range(n_classes):
                # Create mask manually
                mask = [y[i] == class_id for i in range(len(y))]
                if any(mask):
                    color = data_handler.get_color(class_id)
                    # Extract points matching this class
                    x_coords = [X[i][0] for i in range(len(X)) if mask[i]]
                    y_coords = [X[i][1] for i in range(len(X)) if mask[i]]
                    ax.scatter(x_coords, y_coords,
                             c=color, s=100, alpha=0.8,
                             edgecolors='black', linewidth=1.5,
                             label=data_handler.classes[class_id])
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right')
        
        canvas.draw()
    
    def update_loss_plot(self, epoch, loss):
        """Add new loss value and refresh the loss curve."""
        self.loss_history.append((epoch, loss))
        self.loss_ax.clear()
        
        if len(self.loss_history) > 0:
            epochs, losses = zip(*self.loss_history)
            self.loss_ax.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4)
            self.loss_ax.set_xlabel('Epoch')
            self.loss_ax.set_ylabel('Error')
            self.loss_ax.set_title('Training Error')
            self.loss_ax.grid(True, alpha=0.3)
        
        self.loss_canvas.draw()
    
    def clear_loss_history(self):
        """Clear loss history and reset plot."""
        self.loss_history = []
        self.loss_ax.clear()
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Error')
        self.loss_ax.set_title('Error During Training')
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_canvas.draw()
    
    def switch_to_tab(self, tab_name):
        """Switch to a specific tab by name."""
        tab_mapping = {
            'train': "🎯 Training",
            'test': "📊 Test",
            'loss': "📈 Error Graph"
        }
        if tab_name in tab_mapping:
            self.tabview.set(tab_mapping[tab_name])
