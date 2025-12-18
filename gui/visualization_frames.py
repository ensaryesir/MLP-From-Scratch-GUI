"""
Visualization panel with matplotlib plots.
Contains three tabs: Training (interactive), Test, and Error graph.
"""

import customtkinter as ctk
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import COLOR_PALETTE as COLOR_DIGITS, COLOR_DATA_POINTS, COLOR_REGRESSION_LINE


class VisualizationFrame(ctk.CTkFrame):
    """
    Tabbed visualization panel with matplotlib plots.
    Training tab allows interactive point addition via mouse clicks.
    """

    def __init__(self, master, on_point_added_callback=None, **kwargs):
        super().__init__(master, **kwargs)

        self.on_point_added_callback = on_point_added_callback
        self.loss_history = []  # For backward compatibility with regular MLP
        self.ae_loss_history = []  # Separate history for autoencoder
        self.mlp_loss_history = []  # Separate history for MLP (when using AutoencoderMLP)
        self.current_task = 'classification'  # Track current task type
        self.click_enabled = True  # Track if clicking is enabled
        self.click_handler_id = None  # Store click event handler ID

        # create tabbed view
        self.tabview = ctk.CTkTabview(self, width=700, height=600)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # add tabs (manual mode by default)
        self.current_dataset_mode = 'manual'
        self._add_tabs_for_mode('manual')

        # setup matplotlib figures for manual mode tabs only
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
        self.click_handler_id = self.train_canvas.mpl_connect('button_press_event', self._on_train_click)

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
        self.loss_ax.set_title('Training Error')
        self.loss_ax.grid(True, alpha=0.3)

        # embed in tkinter
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, tab)
        self.loss_canvas.draw()
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _setup_ae_loss_tab(self):
        """Setup autoencoder loss tab for training curve."""
        tab = self.tabview.tab("📈 Autoencoder Error")

        # create matplotlib figure
        self.ae_loss_fig = Figure(figsize=(7, 6), dpi=100)
        self.ae_loss_ax = self.ae_loss_fig.add_subplot(111)
        self.ae_loss_ax.set_xlabel('Epoch')
        self.ae_loss_ax.set_ylabel('Error')
        self.ae_loss_ax.set_title('Autoencoder Training Error')
        self.ae_loss_ax.grid(True, alpha=0.3)

        # embed in tkinter
        self.ae_loss_canvas = FigureCanvasTkAgg(self.ae_loss_fig, tab)
        self.ae_loss_canvas.draw()
        self.ae_loss_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _setup_mlp_loss_tab(self):
        """Setup MLP loss tab for training curve."""
        tab = self.tabview.tab("📈 MLP Error")

        # create matplotlib figure
        self.mlp_loss_fig = Figure(figsize=(7, 6), dpi=100)
        self.mlp_loss_ax = self.mlp_loss_fig.add_subplot(111)
        self.mlp_loss_ax.set_xlabel('Epoch')
        self.mlp_loss_ax.set_ylabel('Error')
        self.mlp_loss_ax.set_title('MLP Training Error')
        self.mlp_loss_ax.grid(True, alpha=0.3)

        # embed in tkinter
        self.mlp_loss_canvas = FigureCanvasTkAgg(self.mlp_loss_fig, tab)
        self.mlp_loss_canvas.draw()
        self.mlp_loss_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_train_click(self, event):
        """Handle clicks on training plot to add points."""
        # Ignore clicks if disabled (during training)
        if not self.click_enabled:
            return
        
        # only handle clicks inside the plot
        if event.inaxes == self.train_ax and event.xdata and event.ydata:
            x, y = event.xdata, event.ydata
            if self.on_point_added_callback:
                self.on_point_added_callback(x, y)

    def plot_data_points(self, data_handler, ax=None, task='classification'):
        """Plot all data points on the given axes."""
        if ax is None:
            ax = self.train_ax

        if task == 'regression':
            # Regression: Plot all points as a single color (no class distinction)
            all_points = data_handler.get_all_points()
            if len(all_points) > 0:
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                ax.scatter(x_coords, y_coords,
                          c=COLOR_DATA_POINTS, s=100, alpha=0.7,
                          edgecolors='black', linewidth=1.5,
                          label='Data Points')
        else:
            # Classification: Plot each class with different colors
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

        # draw points (use current_task if available)
        task = getattr(self, 'current_task', 'classification')
        self.plot_data_points(data_handler, self.train_ax, task=task)

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

    def _create_meshgrid(self, x_min=-1, x_max=11, y_min=-1, y_max=11, h=0.1):
        """Create a 2D meshgrid for visualization."""
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
        
        xx = []
        yy = []
        for y_v in y_range:
            xx_row = []
            yy_row = []
            for x_v in x_range:
                xx_row.append(x_v)
                yy_row.append(y_v)
            xx.append(xx_row)
            yy.append(yy_row)
        return xx, yy

    def update_decision_boundary(self, model, X, y, data_handler, tab_name='train', task='classification'):
        """Draw decision boundary (classification) or regression surface (regression)."""
        # select which plot to update
        if tab_name == 'train':
            ax = self.train_ax
            canvas = self.train_canvas
            title = 'Training Data - Decision Boundaries' if task == 'classification' else 'Training Data - Regression'
        else:
            ax = self.test_ax
            canvas = self.test_canvas
            title = 'Test Data - Model Performance' if task == 'classification' else 'Test Data - Regression'
        
        ax.clear()
        
        if task == 'regression':
            # Regression visualization
            if len(X) == 0:
                canvas.draw()
                return
            
            # Check input dimensionality
            input_dim = len(X[0]) if len(X) > 0 else 0
            
            if input_dim == 1:
                # 1D regression: X -> Y (draw regression line)
                x_vals = [x[0] for x in X]  # X is [[x1], [x2], ...]
                y_vals = y  # y is [y1, y2, ...] (target values)
                
                # Sort by x for smooth line plotting
                sorted_pairs = sorted(zip(x_vals, y_vals))
                x_sorted = [p[0] for p in sorted_pairs]
                y_sorted = [p[1] for p in sorted_pairs]
                
                # Plot actual data points
                ax.scatter(x_vals, y_vals, c=COLOR_DATA_POINTS, s=100, alpha=0.7,
                          edgecolors='black', linewidth=1.5, label='Data Points', zorder=3)
                
                # Create prediction line
                x_min, x_max = -1, 11
                x_pred_range = []
                x_val = x_min
                h = 0.1
                while x_val <= x_max:
                    x_pred_range.append(x_val)
                    x_val += h
                
                # Predict on x range
                grid_points = [[x] for x in x_pred_range]
                y_pred = model.predict(grid_points)
                
                # Handle case where predict returns list of lists
                if isinstance(y_pred[0], (list, tuple)):
                    y_pred = [p[0] if len(p) > 0 else 0.0 for p in y_pred]
                
                # Plot regression line
                ax.plot(x_pred_range, y_pred, COLOR_REGRESSION_LINE, linewidth=2, label='Model Prediction', zorder=2)
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(-1, 11)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
                
            else:
                # 2D regression: (X, Y) -> continuous value (draw surface heatmap)
                # Create meshgrid manually
                x_min, x_max = -1, 11
                y_min, y_max = -1, 11
                h = 0.1  # Grid resolution
                
                xx, yy = self._create_meshgrid(x_min, x_max, y_min, y_max, h)
                
                # Flatten the meshgrid and create input for prediction
                grid_points = []
                for i in range(len(xx)):
                    for j in range(len(xx[0])):
                        grid_points.append([xx[i][j], yy[i][j]])
                
                # Predict on grid
                Z = model.predict(grid_points)
                
                # Handle case where predict returns list of lists
                if isinstance(Z[0], (list, tuple)):
                    Z = [z[0] if len(z) > 0 else 0.0 for z in Z]
                
                # Reshape Z back to grid shape
                Z_grid = []
                idx = 0
                for i in range(len(xx)):
                    Z_row = []
                    for j in range(len(xx[0])):
                        Z_row.append(Z[idx])
                        idx += 1
                    Z_grid.append(Z_row)
                
                # Draw continuous heatmap (Purple=Low, Yellow=High)
                # Create custom colormap from purple to yellow
                colors_list = ['#8B00FF', '#4B0082', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFD700']
                n_bins = 100
                cmap = mcolors.LinearSegmentedColormap.from_list('purple_yellow', colors_list, N=n_bins)
                
                im = ax.contourf(xx, yy, Z_grid, levels=50, cmap=cmap, alpha=0.6)
                ax.figure.colorbar(im, ax=ax, label='Predicted Value')
                
                # Overlay data points
                x_coords = [x[0] for x in X]
                y_coords = [x[1] for x in X]
                ax.scatter(x_coords, y_coords, c='black', s=100, alpha=0.8,
                          edgecolors='white', linewidth=1.5, label='Data Points', zorder=3)
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
            
        else:
            # Classification: 2D input -> discrete class predictions
            # create meshgrid manually
            x_min, x_max = -1, 11
            y_min, y_max = -1, 11
            h = 0.1  # Grid resolution
            
            xx, yy = self._create_meshgrid(x_min, x_max, y_min, y_max, h)
            
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

    def update_loss_plot(self, epoch, train_loss, model_type='MLP'):
        """Add new train loss values and refresh the loss curve.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            model_type: 'MLP', 'Autoencoder', or 'AutoencoderMLP' to determine which graph to update
        """
        if model_type == 'Autoencoder':
            # Update autoencoder-specific graph
            self.ae_loss_history.append((epoch, train_loss))
            self.ae_loss_ax.clear()

            if len(self.ae_loss_history) > 0:
                epochs = [e for e, _ in self.ae_loss_history]
                train_losses = [tl for _, tl in self.ae_loss_history]

                # Training loss (blue)
                self.ae_loss_ax.plot(
                    epochs,
                    train_losses,
                    'b-',
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label='Autoencoder Error',
                )

                self.ae_loss_ax.grid(True, alpha=0.3)
                handles, _ = self.ae_loss_ax.get_legend_handles_labels()
                if handles:
                    self.ae_loss_ax.legend(loc='upper right')
                self.ae_loss_ax.set_title("Autoencoder Training Loss")
                self.ae_loss_ax.set_xlabel("Epoch")
                self.ae_loss_ax.set_ylabel("Loss")

            self.ae_loss_canvas.draw()
            self.ae_loss_canvas.flush_events()
        
        elif model_type in ['AutoencoderMLP']:
            # Update MLP-specific graph (when using AutoencoderMLP)
            self.mlp_loss_history.append((epoch, train_loss))
            self.mlp_loss_ax.clear()

            if len(self.mlp_loss_history) > 0:
                epochs = [e for e, _ in self.mlp_loss_history]
                train_losses = [tl for _, tl in self.mlp_loss_history]

                # Training loss (blue)
                self.mlp_loss_ax.plot(
                    epochs,
                    train_losses,
                    'b-',
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label='MLP Error',
                )

                self.mlp_loss_ax.grid(True, alpha=0.3)
                handles, _ = self.mlp_loss_ax.get_legend_handles_labels()
                if handles:
                    self.mlp_loss_ax.legend(loc='upper right')
                self.mlp_loss_ax.set_title("MLP Training Loss")
                self.mlp_loss_ax.set_xlabel("Epoch")
                self.mlp_loss_ax.set_ylabel("Loss")

            self.mlp_loss_canvas.draw()
            self.mlp_loss_canvas.flush_events()
        
        else:
            # Default: use regular loss graph (for backward compatibility)
            self.loss_history.append((epoch, train_loss))
            self.loss_ax.clear()

            if len(self.loss_history) > 0:
                epochs = [e for e, _ in self.loss_history]
                train_losses = [tl for _, tl in self.loss_history]

                # Training loss (blue)
                self.loss_ax.plot(
                    epochs,
                    train_losses,
                    'b-',
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label='Training Error',
                )

                self.loss_ax.grid(True, alpha=0.3)
                handles, _ = self.loss_ax.get_legend_handles_labels()
                if handles:
                    self.loss_ax.legend(loc='upper right')
                self.loss_ax.set_title("Training Loss")
                self.loss_ax.set_xlabel("Epoch")
                self.loss_ax.set_ylabel("Loss")

            self.loss_canvas.draw()

    def clear_loss_history(self, model_type=None):
        """Clear loss history and reset plot.
        
        Args:
            model_type: 'Autoencoder', 'MLP', 'AutoencoderMLP', or None for all
        """
        if model_type == 'Autoencoder' or model_type is None:
            self.ae_loss_history = []
            if hasattr(self, 'ae_loss_ax'):
                self.ae_loss_ax.clear()
                self.ae_loss_ax.set_xlabel('Epoch')
                self.ae_loss_ax.set_ylabel('Error')
                self.ae_loss_ax.set_title('Autoencoder Training Error')
                self.ae_loss_ax.grid(True, alpha=0.3)
                self.ae_loss_canvas.draw()
        
        if model_type in ['MLP', 'AutoencoderMLP'] or model_type is None:
            self.mlp_loss_history = []
            if hasattr(self, 'mlp_loss_ax'):
                self.mlp_loss_ax.clear()
                self.mlp_loss_ax.set_xlabel('Epoch')
                self.mlp_loss_ax.set_ylabel('Error')
                self.mlp_loss_ax.set_title('MLP Training Error')
                self.mlp_loss_ax.grid(True, alpha=0.3)
                self.mlp_loss_canvas.draw()
        
        if model_type is None or model_type not in ['Autoencoder', 'AutoencoderMLP']:
            # Also clear regular loss history for backward compatibility
            self.loss_history = []
            if hasattr(self, 'loss_ax'):
                self.loss_ax.clear()
                self.loss_ax.set_xlabel('Epoch')
                self.loss_ax.set_ylabel('Error')
                self.loss_ax.set_title('Training Error')
                self.loss_ax.grid(True, alpha=0.3)
                self.loss_canvas.draw()

    def switch_to_tab(self, tab_name):
        """Switch to a specific tab by name."""
        tab_mapping = {
            'train': "🎯 Training",
            'test': "📊 Test",
            'loss': "📈 Error Graph",
            'ae_loss': "📈 Autoencoder Error",
            'mlp_loss': "📈 MLP Error",
        }
        if tab_name in tab_mapping:
            self.tabview.set(tab_mapping[tab_name])

    def enable_clicking(self, enabled=True):
        """Enable or disable clicking on the training plot."""
        self.click_enabled = enabled
    
    def _add_tabs_for_mode(self, mode, model_type='MLP'):
        """Add tabs based on dataset mode and model type."""
        if mode == 'mnist':
            # MNIST mode: Separate error graphs for AutoencoderMLP
            if model_type == 'AutoencoderMLP':
                self.tabview.add("📈 Autoencoder Error")
                self.tabview.add("📈 MLP Error")
                self.tabview.add("🔍 Reconstruction")
            else:
                # Regular MLP: single error graph
                self.tabview.add("📈 Error Graph")
        else:
            # Manual mode: Training, Test, Error Graph
            self.tabview.add("🎯 Training")
            self.tabview.add("📊 Test")
            self.tabview.add("📈 Error Graph")
    
    def configure_for_dataset_mode(self, mode, model_type='MLP'):
        """Reconfigure tabs for dataset mode (manual vs mnist) and model type."""
        if self.current_dataset_mode == mode:
            return  # No change needed
        
        self.current_dataset_mode = mode
        
        # Get current tab before clearing
        try:
            current_tab = self.tabview.get()
        except:
            current_tab = None
        
        # Clear all tabs
        for tab_name in self.tabview._tab_dict.copy():
            self.tabview.delete(tab_name)
        
        # Re-add tabs for new mode (with model_type for MNIST)
        self._add_tabs_for_mode(mode, model_type)
        
        # Re-setup figures
        if mode == 'manual':
            self._setup_train_tab()
            self._setup_test_tab()
            self._setup_loss_tab()
            # Set to training tab by default
            self.tabview.set("🎯 Training")
        else:  # mnist
            # Setup loss tabs based on model type
            if model_type == 'AutoencoderMLP':
                self._setup_ae_loss_tab()
                self._setup_mlp_loss_tab()
                self._setup_reconstruction_tab()
                # Set to autoencoder error tab by default
                self.tabview.set("📈 Autoencoder Error")
            else:
                self._setup_loss_tab()
                # Set to error graph by default
                self.tabview.set("📈 Error Graph")
            
            # Force canvas update to prevent black screen - AGGRESSIVE FIX
            # Immediate draw (attempt 1)
            self.update()  # Force full update
            
            if model_type == 'AutoencoderMLP':
                if hasattr(self, 'ae_loss_canvas'):
                    try:
                        self.ae_loss_canvas.draw()
                        self.ae_loss_canvas.flush_events()
                    except:
                        pass
                if hasattr(self, 'mlp_loss_canvas'):
                    try:
                        self.mlp_loss_canvas.draw()
                        self.mlp_loss_canvas.flush_events()
                    except:
                        pass
            else:
                if hasattr(self, 'loss_canvas'):
                    try:
                        self.loss_canvas.draw()
                        self.loss_canvas.flush_events()
                    except:
                        pass
            
            # Delayed redraw (attempt 2 - 200ms)
            def force_redraw_1():
                try:
                    self.update_idletasks()
                    if model_type == 'AutoencoderMLP':
                        if hasattr(self, 'ae_loss_canvas'):
                            self.ae_loss_canvas.draw_idle()
                            self.ae_loss_canvas.flush_events()
                        if hasattr(self, 'mlp_loss_canvas'):
                            self.mlp_loss_canvas.draw_idle()
                            self.mlp_loss_canvas.flush_events()
                        if hasattr(self, 'recon_canvas'):
                            self.recon_canvas.draw_idle()
                    else:
                        if hasattr(self, 'loss_canvas'):
                            self.loss_canvas.draw_idle()
                            self.loss_canvas.flush_events()
                except:
                    pass
            
            # Second delayed redraw (attempt 3 - 500ms)
            def force_redraw_2():
                try:
                    if model_type == 'AutoencoderMLP':
                        if hasattr(self, 'ae_loss_canvas'):
                            self.ae_loss_canvas.draw()
                            self.ae_loss_canvas.flush_events()
                        if hasattr(self, 'mlp_loss_canvas'):
                            self.mlp_loss_canvas.draw()
                            self.mlp_loss_canvas.flush_events()
                    else:
                        if hasattr(self, 'loss_canvas'):
                            self.loss_canvas.draw()
                            self.loss_canvas.flush_events()
                except:
                    pass
            
            # Schedule multiple redraws for reliability
            self.after(200, force_redraw_1)
            self.after(500, force_redraw_2)
    
    def _setup_reconstruction_tab(self):
        """Setup reconstruction tab for autoencoder visualizations."""
        tab = self.tabview.tab("🔍 Reconstruction")
        
        # create matplotlib figure
        self.recon_fig = Figure(figsize=(7, 6), dpi=100)
        self.recon_ax = self.recon_fig.add_subplot(111)
        self.recon_ax.axis('off')
        self.recon_ax.set_title('Original vs Reconstructed Digits')
        
        # embed in tkinter
        self.recon_canvas = FigureCanvasTkAgg(self.recon_fig, tab)
        self.recon_canvas.draw()
        self.recon_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def update_reconstruction(self, original_images, reconstructed_images, mse_per_sample=None):
        """
        Update reconstruction tab with original vs reconstructed digits.
        
        Args:
            original_images: List of original images (flattened 784-dim)
            reconstructed_images: List of reconstructed images (flattened 784-dim)
            mse_per_sample: Optional list of MSE values per sample
        """
        self.recon_ax.clear()
        self.recon_ax.axis('off')
        
        n_samples = min(len(original_images), 10)  # Show max 10 samples
        if n_samples == 0:
            self.recon_canvas.draw()
            return
        
        # Create subplots grid: 2 rows (original, reconstructed) x n_samples columns
        self.recon_fig.clear()
        
        for i in range(n_samples):
            # Original image (top row)
            ax_orig = self.recon_fig.add_subplot(2, n_samples, i + 1)
            img_orig = [original_images[i][j:j+28] for j in range(0, 784, 28)]  # Reshape to 28x28
            ax_orig.imshow(img_orig, cmap='gray')
            ax_orig.axis('off')
            if i == 0:
                ax_orig.set_ylabel('Original', rotation=0, labelpad=40, fontsize=10)
            
            # Reconstructed image (bottom row)
            ax_recon = self.recon_fig.add_subplot(2, n_samples, n_samples + i + 1)
            img_recon = [reconstructed_images[i][j:j+28] for j in range(0, 784, 28)]  # Reshape to 28x28
            ax_recon.imshow(img_recon, cmap='gray')
            ax_recon.axis('off')
            if i == 0:
                ax_recon.set_ylabel('Reconstructed', rotation=0, labelpad=40, fontsize=10)
            
            # Show MSE if provided
            if mse_per_sample and i < len(mse_per_sample):
                ax_recon.set_title(f'MSE: {mse_per_sample[i]:.4f}', fontsize=8)
        
        self.recon_fig.tight_layout()
        self.recon_canvas.draw()
